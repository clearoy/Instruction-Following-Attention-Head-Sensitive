"""Does a BOS-dominated calibration geometry make GPTQ overfit the sink token and
mis-compensate the deployment template tokens?

Mechanism chain under test (W51):

    BOS-dominated calibration  ->  BOS-favoured GPTQ compensation
                               ->  large template-token residual

Per model x calibration condition this measures, at the model's sink-forming
matrix:

  A. Hessian geometry -- is H still near-rank-1 / sink-aligned?
     R1_trace = lam1/tr(H), R1_frob = lam1^2/sum lam_i^2, lam1/lam2, and the
     cosine between the top eigenvector and the mean sink activation.

  B. Compensation residual (the core measurement). THEORY_BRIEF_v2 §2 defines
         r_lam(z) = z_S - H_SF (H_FF + lam I)^-1 z_F,
     the error of predicting z's rounded coordinates from its free ones with the
     calibration-optimal ridge regressor, so that e_GPTQ(z) = delta^T r_lam(z)
     while e_RTN(z) = delta^T z_S. We evaluate it on the split GPTQ actually
     uses -- sequential, in act-order: at step i, S = {i}, F = {i+1..n}.

     That split has an exact closed form in the matrix GPTQ already builds. With
     A = H + lam I and U the upper Cholesky of A^-1 (gptq_core's `Hinv`),

         r_i = (U z)_i / U_ii

     (verified against the direct block formula to 3e-16). So the whole residual
     costs one triangular matvec, and the prediction term is B_lam z_F = z - r.

     Reported: L_sink = ||r(z_sink)||^2, L_template = E_{z in T} ||r(z)||^2 over
     the non-sink chat-template positions, their ratio, plus ||B z_F||/||z_S||
     and cos(B z_F, z_S) -- which separate "overshoot" from "wrong direction".

  C. Causal intervention: rebuild the regressor from H^-B = H - sum_{t in sink}
     x_t x_t^T (exact removal of the sink token's outer products, matched to the
     accumulator's 2/N scaling -- not positional dropping, which on c4 would also
     remove an ordinary token) and recompute every residual. If L_template falls
     sharply once the sink is removed, the first arrow of the chain is causal.

"Sink" rather than "BOS" throughout: for Llama and Mistral the dominant token IS
BOS at position 0 (and position 1 too, under the double-BOS pipeline), but Qwen's
sink is the first newline at position 2 of layer 4 -- so the dominant position is
detected from the calibration norms rather than assumed, and recorded.

  python src/comp_residual.py --model meta-llama/Llama-3.1-8B-Instruct \
      --target 1:down_proj --calib c4 --calib-file data/calib_cache/c4.jsonl \
      --deploy data/ifeval_input_data.jsonl --out runs/comp_residual_l_c4.csv
"""
import argparse
import csv
import gc
import json
import os

import torch

from common import DEFAULT_MODEL, load_model
from quantize_gptq import load_calib
from quantize_protected import ATTN, capture_layer0_inputs

P = 8                     # first P positions of each sequence are the template window
CSV_COLS = ["model", "layer", "matrix", "calib", "hessian", "n_calib_tokens", "dim",
            "sink_pos", "sink_tokens", "xnorm_sink", "xnorm_template",
            "lambda_1", "lambda_2", "lam1_over_lam2", "trace",
            "R1_trace", "R1_frob", "cos_v1_sink",
            "L_sink", "L_template", "L_ratio",
            "amp_sink", "amp_template", "pred_ratio_template", "cos_pred_template",
            "percdamp", "n_deploy"]


def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def walk(model, layers, inps, kws, li, mod, hook):
    """Run every calibration sample through layers up to `li`, hooking `mod`."""
    h = mod.register_forward_pre_hook(hook)
    for j in range(len(inps)):
        layers[li](inps[j], **kws[j])
    h.remove()


@torch.no_grad()
def collect(args):
    """-> H, n_tokens, X_calib [n,P,d], X_deploy [m,P,d], deploy token strings."""
    from gptq_core import MaskedGPTQ
    model, tok = load_model(args.model)
    li, proj = args.target.split(":")
    li = int(li)

    if args.calib_file:
        with open(args.calib_file, encoding="utf-8") as f:
            calib = [json.loads(l)["text"] for l in f][: args.n_calib]
    else:
        calib = load_calib(args.calib, tok, args.n_calib, args.seqlen)

    deploy_rows = [json.loads(l) for l in open(args.deploy, encoding="utf-8")][: args.n_deploy]
    deploy = [chat(tok, r.get("prompt") or r["text"]) for r in deploy_rows]
    ids0 = tok(deploy[0], truncation=True, max_length=args.seqlen)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids0[:P])
    print(f"[resid] deploy prompt-0 positions 0..{P - 1}: {toks}")

    layers = model.model.layers
    mod = getattr(layers[li].self_attn if proj in ATTN else layers[li].mlp, proj)
    g = MaskedGPTQ(mod, name=f"layers.{li}.{proj}")

    def grab(store):
        def hook(_m, a):
            x = a[0].reshape(-1, a[0].shape[-1]).float()
            store.append(x[:P].clone().cpu())
        return hook

    Xc, Xd = [], []
    inps, kws = capture_layer0_inputs(model, tok, calib, args.seqlen)
    for l in range(li + 1):
        if l == li:
            def hook(_m, a):                      # H over ALL tokens, plus the window
                g.add_batch(a[0])
                grab(Xc)(_m, a)
            walk(model, layers, inps, kws, l, mod, hook)
        for j in range(len(inps)):
            out = layers[l](inps[j], **kws[j])
            inps[j] = out[0] if isinstance(out, tuple) else out
    H, n_tok = g.H.clone(), g.nsamples
    g.free()
    del inps
    gc.collect()
    torch.cuda.empty_cache()

    inps, kws = capture_layer0_inputs(model, tok, deploy, args.seqlen)
    for l in range(li + 1):
        if l == li:
            walk(model, layers, inps, kws, l, mod, grab(Xd))
        for j in range(len(inps)):
            out = layers[l](inps[j], **kws[j])
            inps[j] = out[0] if isinstance(out, tuple) else out

    H = H.cpu()
    del model, inps
    gc.collect()
    torch.cuda.empty_cache()
    return H, n_tok, torch.stack(Xc), torch.stack(Xd), toks


def detect_sink(Xc: torch.Tensor, forced: str | None) -> list[int]:
    """Positions whose mean input norm is at least half the largest. Llama/Mistral
    give {0} on raw text and {0,1} under the double-BOS chat pipeline; Qwen gives
    {2}, its first newline -- which is why this is measured, not assumed."""
    if forced:
        return [int(p) for p in forced.split(",")]
    n = Xc.norm(dim=-1).mean(0)                                  # [P]
    return [int(p) for p in torch.where(n >= 0.5 * n.max())[0]]


@torch.no_grad()
def residuals(H: torch.Tensor, Z: torch.Tensor, percdamp: float, actorder: bool):
    """r_i = (Uz)_i / U_ii with U the upper Cholesky of (H+lam I)^-1, in act-order
    -- the sequential split GPTQ uses. Returns per-row ||r||^2, ||Bz||/||z||,
    cos(Bz, z)."""
    A = H.clone()
    dead = torch.diag(A) == 0
    A[dead, dead] = 1.0
    Z = Z.clone()
    Z[:, dead] = 0
    if actorder:
        perm = torch.argsort(torch.diag(A), descending=True)
        A = A[perm][:, perm]
        Z = Z[:, perm]
    idx = torch.arange(A.shape[0], device=A.device)
    A[idx, idx] += percdamp * torch.mean(torch.diag(A))
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(A)), upper=True)
    R = (Z @ U.t()) / torch.diag(U)                              # [k, n]
    B = Z - R                                                    # the prediction B_lam z_F
    zn = Z.norm(dim=1).clamp(min=1e-12)
    return ((R ** 2).sum(1), R.norm(dim=1) / zn, B.norm(dim=1) / zn,
            torch.nn.functional.cosine_similarity(B, Z, dim=1))


@torch.no_grad()
def analyse(args, H, n_tok, Xc, Xd, toks, sink, dev):
    """One CSV row per Hessian condition (full, and sink-removed)."""
    d = H.shape[0]
    tpl = [p for p in range(P) if p not in sink]
    b = Xc[:, sink].reshape(-1, d).mean(0).double()
    b = (b / b.norm().clamp(min=1e-12)).float()

    # H^-B: exact removal of the sink outer products, at the accumulator's 2/N scale
    Xs = Xc[:, sink].reshape(-1, d).to(dev).float()
    H_minus = H.to(dev) - (2.0 / n_tok) * (Xs.t() @ Xs)

    Zs = Xd[:, sink].reshape(-1, d).to(dev).float()
    Zt = Xd[:, tpl].reshape(-1, d).to(dev).float()
    rows = []
    for name, Hx in (("full", H.to(dev)), ("sink_removed", H_minus)):
        ev, evec = torch.linalg.eigh(Hx)
        lam = ev.flip(0).double()
        v1 = evec[:, -1].float().cpu()
        Ls, amp_s, _, _ = residuals(Hx, Zs, args.percdamp, not args.no_actorder)
        Lt, amp_t, pred_t, cos_t = residuals(Hx, Zt, args.percdamp, not args.no_actorder)
        rows.append({
            "model": args.model, "layer": args.target.split(":")[0],
            "matrix": args.target.split(":")[1], "calib": args.calib, "hessian": name,
            "n_calib_tokens": n_tok, "dim": d,
            "sink_pos": "|".join(map(str, sink)),
            "sink_tokens": "|".join(toks[p] for p in sink),
            "xnorm_sink": round(float(Xc[:, sink].norm(dim=-1).mean()), 3),
            "xnorm_template": round(float(Xc[:, tpl].norm(dim=-1).mean()), 3),
            "lambda_1": f"{float(lam[0]):.6g}", "lambda_2": f"{float(lam[1]):.6g}",
            "lam1_over_lam2": f"{float(lam[0] / lam[1].clamp(min=1e-30)):.6g}",
            "trace": f"{float(lam.sum()):.6g}",
            "R1_trace": round(float(lam[0] / lam.sum()), 6),
            "R1_frob": round(float(lam[0] ** 2 / (lam ** 2).sum()), 6),
            "cos_v1_sink": round(float(torch.dot(v1, b).abs()), 6),
            "L_sink": f"{float(Ls.mean()):.6g}", "L_template": f"{float(Lt.mean()):.6g}",
            "L_ratio": f"{float(Lt.mean() / Ls.mean().clamp(min=1e-30)):.6g}",
            "amp_sink": round(float(amp_s.mean()), 4),
            "amp_template": round(float(amp_t.mean()), 4),
            "pred_ratio_template": round(float(pred_t.mean()), 4),
            "cos_pred_template": round(float(cos_t.mean()), 4),
            "percdamp": args.percdamp, "n_deploy": args.n_deploy,
        })
        print(f"[resid] {name}: R1_trace={rows[-1]['R1_trace']} lam1/lam2={rows[-1]['lam1_over_lam2']} "
              f"cos_v1_sink={rows[-1]['cos_v1_sink']} | L_sink={rows[-1]['L_sink']} "
              f"L_template={rows[-1]['L_template']} ratio={rows[-1]['L_ratio']} "
              f"amp_tpl={rows[-1]['amp_template']} cos_pred={rows[-1]['cos_pred_template']}", flush=True)
        del ev, evec
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--target", required=True, help='"1:down_proj" (Qwen-14B: "4:down_proj")')
    ap.add_argument("--calib", default="c4",
                    choices=["c4", "instruct", "wikitext", "ultrachat", "c4chat", "c4wrongchat"])
    ap.add_argument("--calib-file", help="prefetched calibration jsonl (offline compute nodes)")
    ap.add_argument("--deploy", default="data/ifeval_input_data.jsonl")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--n-deploy", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--percdamp", type=float, default=0.05)
    ap.add_argument("--no-actorder", action="store_true")
    ap.add_argument("--sink-pos", help="force the sink positions, e.g. \"0,1\" (default: detect)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    H, n_tok, Xc, Xd, toks = collect(args)
    sink = detect_sink(Xc, args.sink_pos)
    norms = Xc.norm(dim=-1).mean(0)
    print(f"[resid] calibration norms by position: "
          f"{[round(float(v), 1) for v in norms]} -> sink positions {sink}")

    dev = args.device if torch.cuda.is_available() else "cpu"
    rows = analyse(args, H, n_tok, Xc, Xd, toks, sink, dev)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[resid] -> {args.out}")


if __name__ == "__main__":
    main()
