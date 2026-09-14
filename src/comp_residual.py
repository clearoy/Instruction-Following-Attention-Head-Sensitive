"""Does a BOS-dominated calibration geometry make GPTQ overfit the sink token and
mis-compensate the deployment template tokens?

Mechanism chain under test (W51/W52):

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

     so the whole residual is one triangular matvec, and the prediction term is
     B_lam z_F = z - r. `--self-test` checks this against the direct block
     formula (and two other invariants) with no model required.

     Reported: L_sink = ||r(z_sink)||^2, L_template = E_{z in T} ||r(z)||^2 over
     the non-sink chat-template positions, their ratio, plus ||B z_F||/||z_S||
     and cos(B z_F, z_S) -- which separate "overshoot" from "wrong direction".
     Every deployment token's raw numbers are written to <out>.tokens.csv; the
     aggregate row is a summary of that file, not a substitute for it.

  C. Causal intervention: rebuild the regressor from H^-B = H - sum_{t in sink}
     x_t x_t^T and recompute every residual. If L_template falls sharply once the
     sink is removed, the first arrow of the chain is causal.

Three things this file is careful about:

  * **float64.** The intervention subtracts the sink outer products from H, and
    in float32 that is catastrophic cancellation: the sink eigenvalue is ~1e3
    while the bulk eigenvalues are ~1e-4, and eps32 * 1e3 ~ 6e-5 is the same
    order as the bulk -- the subtraction would return numerical noise. So H is
    accumulated in float64 by `Accum`, which reproduces the running update in
    gptq_core.MaskedGPTQ._accum exactly (it telescopes to H = (2/N) sum_t x_t
    x_t^T over the total token count N). `--self-test` checks it against the
    pipeline's own accumulator. Note r_lam is invariant to a global rescaling of
    H (lam = rho * mean diag H scales with it), so subtracting the outer products
    rather than renormalising over the surviving tokens changes nothing.

  * **Identical deployment z across calibration conditions.** z is the target
    module's input on deployment prompts under the *unquantized* model, so it
    cannot depend on the calibration corpus -- but rather than rely on that, the
    z window is computed once per model and cached (`--z-cache`); every
    calibration condition for that model loads the same tensor.

  * **"Sink", not "BOS", and detected on BOTH sides.** For Llama and Mistral the
    dominant token IS BOS, but Qwen's sink is the first newline at position 2 of
    layer 4 -- so the dominant position is detected from activation norms rather
    than assumed. It is detected separately for calibration (which builds H^-B)
    and for deployment (which splits L_sink from L_template), because the two
    need not agree: raw c4 carries one BOS at position 0, while the chat-formatted
    deployment prompts carry Llama's double-BOS at positions 0 AND 1. Inheriting
    the calibration positions would file that second BOS under "template", and
    L_template would then be reporting BOS.

  python src/comp_residual.py --self-test        # no model needed
  python src/comp_residual.py --model meta-llama/Llama-3.1-8B-Instruct \
      --target 1:down_proj --calib c4 --calib-file data/calib_cache/c4.jsonl \
      --z-cache runs/zcache_llama.pt --out runs/comp_residual_l_c4.csv
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
            "sink_pos", "sink_pos_deploy", "sink_tokens", "xnorm_sink", "xnorm_template",
            "lambda_1", "lambda_2", "lam1_over_lam2", "trace",
            "R1_trace", "R1_frob", "cos_v1_sink",
            "L_sink", "L_template", "L_ratio",
            "amp_sink", "amp_template", "pred_ratio_template", "cos_pred_template",
            "percdamp", "n_deploy", "tokens_csv"]
TOK_COLS = ["hessian", "prompt", "pos", "token", "is_sink", "znorm",
            "r_sq", "amp", "pred_ratio", "cos_pred"]


class Accum:
    """float64 reproduction of gptq_core.MaskedGPTQ._accum (see module docstring)."""

    def __init__(self, cols: int, dev):
        self.H = torch.zeros((cols, cols), dtype=torch.float64, device=dev)
        self.n = 0

    @torch.no_grad()
    def add(self, inp: torch.Tensor):
        x = inp.reshape(-1, self.H.shape[0]).t().double()
        n = x.shape[1]
        self.H *= self.n / (self.n + n)
        self.n += n
        x *= (2.0 / self.n) ** 0.5
        self.H += x @ x.t()


@torch.no_grad()
def residuals(H: torch.Tensor, Z: torch.Tensor, percdamp: float, actorder: bool):
    """r_i = (Uz)_i / U_ii with U the upper Cholesky of (H+lam I)^-1, in act-order
    -- the sequential split GPTQ uses. Returns per-row ||r||^2, ||r||/||z||,
    ||B z_F||/||z_S|| and cos(B z_F, z_S)."""
    A = H.clone().double()
    dead = torch.diag(A) == 0
    A[dead, dead] = 1.0
    Z = Z.clone().double()
    Z[:, dead] = 0
    if actorder:
        perm = torch.argsort(torch.diag(A), descending=True)
        A = A[perm][:, perm]
        Z = Z[:, perm]
    idx = torch.arange(A.shape[0], device=A.device)
    A[idx, idx] += percdamp * torch.mean(torch.diag(A))
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(A)), upper=True)
    R = (Z @ U.t()) / torch.diag(U)
    B = Z - R
    zn = Z.norm(dim=1).clamp(min=1e-30)
    return ((R ** 2).sum(1), R.norm(dim=1) / zn, B.norm(dim=1) / zn,
            torch.nn.functional.cosine_similarity(B, Z, dim=1))


# ------------------------------------------------------------------ self-test

def self_test(dev="cpu"):
    """Three invariants, no model needed:
      1. the sequential r_lam equals the direct block formula z_S - H_SF(H_FF+lam I)^-1 z_F
      2. Accum reproduces gptq_core.MaskedGPTQ's accumulated H
      3. H minus the sink outer products equals a direct sink-free accumulation
         (up to the global rescaling r_lam is invariant to)
    """
    from gptq_core import MaskedGPTQ
    torch.manual_seed(0)
    n, N = 64, 700
    ok = True

    # --- 1. sequential residual == block formula (with act-order, as in residuals())
    M = torch.randn(n, n, dtype=torch.float64)
    H = M @ M.t() + 0.3 * torch.eye(n, dtype=torch.float64)
    z = torch.randn(n, dtype=torch.float64)
    rho = 0.05
    A = H + rho * H.diag().mean() * torch.eye(n, dtype=torch.float64)
    perm = torch.argsort(torch.diag(A), descending=True)
    Ap, zp = A[perm][:, perm], z[perm]
    direct = torch.empty(n, dtype=torch.float64)
    for i in range(n):
        F = list(range(i + 1, n))
        direct[i] = zp[i] if not F else zp[i] - Ap[i, F] @ torch.linalg.solve(
            Ap[F][:, F], zp[F])
    r_sq, _, _, _ = residuals(H, z.unsqueeze(0), rho, True)
    d1 = abs(float(r_sq[0]) - float((direct ** 2).sum())) / float((direct ** 2).sum())
    print(f"[self-test] 1 sequential r_lam vs block formula: rel diff {d1:.3e} "
          f"{'OK' if d1 < 1e-10 else 'FAIL'}")
    ok &= d1 < 1e-10

    # --- 2. Accum (float64) == MaskedGPTQ._accum (float32)
    lin = torch.nn.Linear(n, 8, bias=False)
    g = MaskedGPTQ(lin, name="t")
    acc = Accum(n, "cpu")
    batches = [torch.randn(1, N // 7, n) for _ in range(7)]
    for b in batches:
        g.add_batch(b)
        acc.add(b)
    d2 = float((acc.H.float() - g.H).abs().max() / g.H.abs().max())
    print(f"[self-test] 2 Accum vs MaskedGPTQ._accum:        rel diff {d2:.3e} "
          f"{'OK' if d2 < 1e-5 else 'FAIL'}")
    ok &= d2 < 1e-5

    # --- 3. outer-product removal == sink-free accumulation, up to global scale
    X = torch.cat([b.reshape(-1, n) for b in batches], 0).double()
    sink = torch.arange(0, X.shape[0], 50)           # pretend these are sink tokens
    keep = torch.tensor([i for i in range(X.shape[0]) if i not in set(sink.tolist())])
    Xs = X[sink]
    H_minus = acc.H - (2.0 / acc.n) * (Xs.t() @ Xs)
    H_direct = (2.0 / keep.numel()) * (X[keep].t() @ X[keep])
    scale = keep.numel() / acc.n
    d3 = float((H_minus - scale * H_direct).abs().max() / H_direct.abs().max())
    print(f"[self-test] 3 H - sink outer products:           rel diff {d3:.3e} "
          f"{'OK' if d3 < 1e-10 else 'FAIL'}")
    ok &= d3 < 1e-10

    # r_lam invariance to the global rescaling that distinguishes the two
    a, _, _, _ = residuals(H_minus, z.unsqueeze(0), rho, True)
    b_, _, _, _ = residuals(H_minus / scale, z.unsqueeze(0), rho, True)
    d4 = abs(float(a[0]) - float(b_[0])) / float(a[0])
    print(f"[self-test] 4 r_lam invariant to H rescaling:    rel diff {d4:.3e} "
          f"{'OK' if d4 < 1e-8 else 'FAIL'}")
    ok &= d4 < 1e-8
    print(f"[self-test] {'ALL PASS' if ok else 'FAILURE'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ collection

def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def collect(args):
    """-> H (float64), n_tokens, X_calib [n,P,d], Z_deploy [m,P,d], deploy ids [m,P]."""
    model, tok = load_model(args.model)
    li, proj = args.target.split(":")
    li = int(li)
    layers = model.model.layers
    mod = getattr(layers[li].self_attn if proj in ATTN else layers[li].mlp, proj)
    dev = mod.weight.device

    if args.calib_file:
        with open(args.calib_file, encoding="utf-8") as f:
            calib = [json.loads(l)["text"] for l in f][: args.n_calib]
    else:
        calib = load_calib(args.calib, tok, args.n_calib, args.seqlen)

    acc = Accum(mod.weight.shape[1], dev)
    Xc: list = []

    def window(store):
        def hook(_m, a):
            x = a[0].reshape(-1, a[0].shape[-1]).float()
            store.append(x[:P].clone().cpu())
        return hook

    def run(texts, at_target):
        inps, kws = capture_layer0_inputs(model, tok, texts, args.seqlen)
        for l in range(li + 1):
            if l == li:
                h = mod.register_forward_pre_hook(at_target)
                for j in range(len(inps)):
                    layers[l](inps[j], **kws[j])
                h.remove()
            for j in range(len(inps)):
                out = layers[l](inps[j], **kws[j])
                inps[j] = out[0] if isinstance(out, tuple) else out
        del inps
        gc.collect()
        torch.cuda.empty_cache()

    def calib_hook(_m, a):
        acc.add(a[0])
        window(Xc)(_m, a)

    run(calib, calib_hook)
    H, n_tok = acc.H.cpu(), acc.n
    del acc
    gc.collect()
    torch.cuda.empty_cache()

    # Deployment z: identical across calibration conditions by construction (the
    # model is unquantized), and cached so that it is identical by inspection too.
    if args.z_cache and os.path.exists(args.z_cache):
        d = torch.load(args.z_cache)
        Z, ids = d["Z"], d["ids"]
        assert d["model"] == args.model and d["target"] == args.target, \
            f"z-cache {args.z_cache} was built for {d['model']} {d['target']}"
        print(f"[resid] deployment z loaded from {args.z_cache} ({tuple(Z.shape)})")
    else:
        rows = [json.loads(l) for l in open(args.deploy, encoding="utf-8")][: args.n_deploy]
        texts = [chat(tok, r.get("prompt") or r["text"]) for r in rows]
        ids = torch.tensor([tok(t, truncation=True, max_length=args.seqlen)["input_ids"][:P]
                            for t in texts])
        Zl: list = []
        run(texts, window(Zl))
        Z = torch.stack(Zl)
        if args.z_cache:
            # Atomic: sibling tasks for the same model may race here, and they
            # compute identical z (the model is unquantized), so last-writer-wins
            # is correct -- but a reader must never see a half-written file.
            tmp = f"{args.z_cache}.{os.getpid()}.tmp"
            torch.save({"Z": Z, "ids": ids, "model": args.model, "target": args.target}, tmp)
            os.replace(tmp, args.z_cache)
            print(f"[resid] deployment z cached -> {args.z_cache}")
    print(f"[resid] deploy prompt-0 window: {tok.convert_ids_to_tokens(ids[0].tolist())}")
    toks = [tok.convert_ids_to_tokens(r.tolist()) for r in ids]

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return H, n_tok, torch.stack(Xc), Z, toks


def detect_sink(Xc: torch.Tensor, forced: str | None) -> list[int]:
    """Positions whose mean input norm is at least half the largest."""
    if forced:
        return [int(p) for p in forced.split(",")]
    n = Xc.norm(dim=-1).mean(0)
    return [int(p) for p in torch.where(n >= 0.5 * n.max())[0]]


@torch.no_grad()
def analyse(args, H, n_tok, Xc, Z, toks, sink, dev):
    d = H.shape[0]
    tpl = [p for p in range(P) if p not in sink]
    b = Xc[:, sink].reshape(-1, d).double().mean(0)
    b = (b / b.norm().clamp(min=1e-30)).to(dev)

    Xs = Xc[:, sink].reshape(-1, d).to(dev).double()
    H = H.to(dev)
    H_minus = H - (2.0 / n_tok) * (Xs.t() @ Xs)

    m = Z.shape[0]
    Zf = Z.reshape(-1, d).to(dev)                       # [m*P, d]
    pos = torch.arange(P).repeat(m)
    prompt = torch.arange(m).repeat_interleave(P)
    # The sink must be identified on the DEPLOYMENT side by the same criterion,
    # not inherited from the calibration positions: c4 is raw text with one BOS
    # at position 0, while the deployment prompts are chat-formatted and Llama's
    # double-BOS puts a second one at position 1. Inheriting `sink` would file
    # that second BOS (norm ~481) under "template" and the template average would
    # be measuring BOS.
    sink_dep = detect_sink(Z, args.sink_pos_deploy)
    is_sink = torch.tensor([p in sink_dep for p in pos.tolist()])

    rows, trows = [], []
    for name, Hx in (("full", H), ("sink_removed", H_minus)):
        try:
            ev, evec = torch.linalg.eigh(Hx)
        except torch.OutOfMemoryError:
            # ~18 s on a 13824^2 float64 matrix with a threaded BLAS; far cheaper
            # than failing the arm.
            print(f"[resid] eigh OOM on {dev}; retrying on cpu", flush=True)
            torch.cuda.empty_cache()
            ev, evec = torch.linalg.eigh(Hx.cpu())
            ev, evec = ev.to(Hx.device), evec.to(Hx.device)
        lam = ev.flip(0)
        v1 = evec[:, -1]
        r_sq, amp, pred, cosp = residuals(Hx, Zf, args.percdamp, not args.no_actorder)
        r_sq, amp, pred, cosp = (t.cpu() for t in (r_sq, amp, pred, cosp))
        zn = Zf.norm(dim=1).cpu()
        for k in range(r_sq.numel()):
            trows.append({"hessian": name, "prompt": int(prompt[k]), "pos": int(pos[k]),
                          "token": toks[int(prompt[k])][int(pos[k])],
                          "is_sink": int(bool(is_sink[k])),
                          "znorm": f"{float(zn[k]):.6g}", "r_sq": f"{float(r_sq[k]):.6g}",
                          "amp": f"{float(amp[k]):.6g}", "pred_ratio": f"{float(pred[k]):.6g}",
                          "cos_pred": f"{float(cosp[k]):.6g}"})
        s, t = is_sink, ~is_sink
        rows.append({
            "model": args.model, "layer": args.target.split(":")[0],
            "matrix": args.target.split(":")[1], "calib": args.calib, "hessian": name,
            "n_calib_tokens": n_tok, "dim": d,
            "sink_pos": "|".join(map(str, sink)),
            "sink_pos_deploy": "|".join(map(str, sink_dep)),
            "sink_tokens": "|".join(toks[0][p] for p in sink_dep),
            "xnorm_sink": round(float(Xc[:, sink].norm(dim=-1).mean()), 3),
            "xnorm_template": round(float(Xc[:, tpl].norm(dim=-1).mean()), 3),
            "lambda_1": f"{float(lam[0]):.6g}", "lambda_2": f"{float(lam[1]):.6g}",
            "lam1_over_lam2": f"{float(lam[0] / lam[1].clamp(min=1e-30)):.6g}",
            "trace": f"{float(lam.sum()):.6g}",
            "R1_trace": round(float(lam[0] / lam.sum()), 6),
            "R1_frob": round(float(lam[0] ** 2 / (lam ** 2).sum()), 6),
            "cos_v1_sink": round(float(torch.dot(v1, b).abs()), 6),
            "L_sink": f"{float(r_sq[s].mean()):.6g}",
            "L_template": f"{float(r_sq[t].mean()):.6g}",
            "L_ratio": f"{float(r_sq[t].mean() / r_sq[s].mean().clamp(min=1e-30)):.6g}",
            "amp_sink": round(float(amp[s].mean()), 4),
            "amp_template": round(float(amp[t].mean()), 4),
            "pred_ratio_template": round(float(pred[t].mean()), 4),
            "cos_pred_template": round(float(cosp[t].mean()), 4),
            "percdamp": args.percdamp, "n_deploy": m,
            "tokens_csv": args.out.replace(".csv", "") + ".tokens.csv",
        })
        r = rows[-1]
        print(f"[resid] {name}: R1_trace={r['R1_trace']} lam1/lam2={r['lam1_over_lam2']} "
              f"cos_v1_sink={r['cos_v1_sink']} | L_sink={r['L_sink']} "
              f"L_template={r['L_template']} ratio={r['L_ratio']} | amp_sink={r['amp_sink']} "
              f"amp_tpl={r['amp_template']} pred={r['pred_ratio_template']} "
              f"cos_pred={r['cos_pred_template']}", flush=True)
        del ev, evec
        torch.cuda.empty_cache()
    return rows, trows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--target", help='"1:down_proj" (Qwen-14B: "4:down_proj")')
    ap.add_argument("--calib", default="c4",
                    choices=["c4", "instruct", "wikitext", "ultrachat", "c4chat", "c4wrongchat"])
    ap.add_argument("--calib-file", help="prefetched calibration jsonl (offline compute nodes)")
    ap.add_argument("--deploy", default="data/ifeval_input_data.jsonl")
    ap.add_argument("--z-cache", help="cache the deployment z window here; every calibration "
                                      "condition for a model must share one file")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--n-deploy", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--percdamp", type=float, default=0.05)
    ap.add_argument("--no-actorder", action="store_true")
    ap.add_argument("--sink-pos", help='force the calibration sink positions, e.g. "0,1" (default: detect)')
    ap.add_argument("--sink-pos-deploy", help="force the deployment sink positions (default: detect)")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self-test", action="store_true", help="check the invariants; no model needed")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())
    assert args.target and args.out, "--target and --out are required unless --self-test"

    H, n_tok, Xc, Z, toks = collect(args)
    # collect()'s nested run()/hook closures hold the model in their cells, so the
    # `del model` inside it does not release anything until that frame exits. Only
    # here is the memory actually reclaimable -- and the analysis below needs it,
    # since eigh on a 13824^2 float64 Hessian wants ~9 GB of workspace.
    gc.collect()
    torch.cuda.empty_cache()
    sink = detect_sink(Xc, args.sink_pos)
    print(f"[resid] calibration norms by position: "
          f"{[round(float(v), 1) for v in Xc.norm(dim=-1).mean(0)]} -> sink {sink}")

    dev = args.device if torch.cuda.is_available() else "cpu"
    rows, trows = analyse(args, H, n_tok, Xc, Z, toks, sink, dev)
    for path, cols, data in ((args.out, CSV_COLS, rows),
                             (args.out.replace(".csv", "") + ".tokens.csv", TOK_COLS, trows)):
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if new:
                w.writeheader()
            for r in data:
                w.writerow(r)
    print(f"[resid] -> {args.out} (+ .tokens.csv, {len(trows)} per-token rows)")


if __name__ == "__main__":
    main()
