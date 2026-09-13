"""Is the calibration Hessian of a sink-forming matrix effectively rank one?

Hypothesis under test: at the matrix where the model manufactures its
attention sink (Llama-3.1-8B layer-1 down_proj), the BOS position carries an
activation norm two orders of magnitude above every other token (~481 vs
1-2.5). Because H_c = sum_t x_t x_t^T is a sum of outer products weighted by
||x_t||^2, one token then contributes essentially the whole matrix: H_c is
dominated by a single direction, and GPTQ's compensation -- which solves
against H_c -- degenerates.

This script measures that dominance directly. For a target matrix it

  1. collects H_c with the pipeline's own accumulator (gptq_core.MaskedGPTQ.
     add_batch, the exact code path used during quantization) and saves it,
  2. eigendecomposes it (torch.linalg.eigh; H_c is symmetric PSD),
  3. reports how much of ||H_c||_F^2 the leading eigenpair explains, and
     whether the leading eigenvector points along the mean BOS activation.

Columns written per (model, layer, matrix, data_source):
  lambda_1, lambda_2    two largest eigenvalues of H_c
  R1                    lambda_1^2 / sum_i lambda_i^2  -- Frobenius-squared
                        share of the leading eigenpair. Equivalently the
                        rank-one approximation explains R1 of ||H_c||_F^2.
  residual_ratio        sqrt(1 - R1) = ||H_c - lambda_1 v1 v1^T||_F/||H_c||_F
  cos_align             |<v1, b>|, b = normalized MEAN BOS activation over the
                        whole calibration set (not a single sample)
  xnorm_bos/_rest       mean activation norm at position 0 vs positions >= 1,
                        the premise of the hypothesis, measured not assumed

Scope and caveats, deliberately:
  * H_c is measured under the input distribution the pipeline ACTUALLY feeds
    this matrix, not an idealized clean one. With --quantize-preceding the
    preceding layers are quantized first exactly as in a real GPTQ run (so
    the target sees already-quantized upstream activations); by default they
    are left in fp16, which isolates the claim being tested -- that the
    dominance is an intrinsic property of the fp16 model (attention sink),
    not something quantization creates. Both are legitimate; they answer
    different questions, so the choice is recorded in the CSV.
  * This is a single-matrix, static analysis. No cross-layer coupling is
    modelled and none should be read into it.
  * BOS is never dropped or edited. Its outlier activation is a property of
    the POSITION (first token, nothing to attend to under the causal mask),
    not of the token's content, so removing it would merely promote the next
    position to sink and answer a different question.
  * lambda_i inherit the GPTQ accumulator's scaling, H = (2/N) sum_t x_t x_t^T
    (a running mean, so comparable across corpora of different length).
    R1, residual_ratio and cos_align are scale-free.

  python src/hessian_rank1.py --model meta-llama/Llama-3.1-8B-Instruct \
      --targets "1:down_proj" --calib c4 --data-source calib \
      --hess-dir $STORE/hessians/llama31-8b-c4 --out runs/hess_rank1_l1_calib.csv
"""
import argparse
import csv
import gc
import json
import os

import torch

from common import DEFAULT_MODEL, load_model
from gptq_core import MaskedGPTQ
from quantize_gptq import load_calib
from quantize_protected import ATTN, capture_layer0_inputs

CSV_COLS = ["model", "layer", "matrix", "data_source", "calib", "n_samples",
            "n_tokens", "dim", "lambda_1", "lambda_2", "R1", "residual_ratio",
            "cos_align", "xnorm_bos", "xnorm_rest", "xnorm_ratio",
            "quantize_preceding", "hess_path"]


def parse_targets(spec: str) -> dict[int, list[str]]:
    """"1:down_proj;16:down_proj,up_proj" -> {1: [down_proj], 16: [...]}."""
    out: dict[int, list[str]] = {}
    for part in spec.split(";"):
        lay, _, projs = part.partition(":")
        for p in projs.split(","):
            out.setdefault(int(lay), []).append(p.strip())
    return out


class BosProbe:
    """Mean activation at position 0 and mean norms, alongside the Hessian.

    Position 0 is BOS for every corpus here: the raw-text arms get it from the
    tokenizer, the chat arms from the template (plus the tokenizer's own, the
    documented double-BOS protocol deviation -- either way position 0 is BOS).
    """

    def __init__(self, cols: int):
        self.sum = torch.zeros(cols, dtype=torch.float64)
        self.n = 0
        self.norm_bos = 0.0
        self.norm_rest = 0.0
        self.n_tokens = 0

    @torch.no_grad()
    def add(self, inp: torch.Tensor):
        x = inp.reshape(-1, inp.shape[-1]).float()
        self.sum += x[0].double().cpu()
        self.n += 1
        self.norm_bos += float(x[0].norm())
        if x.shape[0] > 1:
            self.norm_rest += float(x[1:].norm(dim=-1).mean())
        self.n_tokens += x.shape[0]

    def bos_vec(self) -> torch.Tensor:
        b = self.sum / max(self.n, 1)
        return (b / b.norm().clamp(min=1e-12)).float()


def get_calib(args, tok) -> list[str]:
    """c4 and ultrachat are streamed from the Hub, which needs outbound network.
    Compute nodes that lack it read a jsonl prefetched by --dump-calib on a
    login node instead; the texts are identical, only the fetch moves."""
    if args.calib_file:
        with open(args.calib_file, encoding="utf-8") as f:
            return [json.loads(l)["text"] for l in f][: args.n_calib]
    return load_calib(args.calib, tok, args.n_calib, args.seqlen, seed=args.calib_seed)


@torch.no_grad()
def collect(args, targets: dict[int, list[str]]):
    """Walk the layers with the pipeline's own hooks; dump H for the targets."""
    model, tok = load_model(args.model)
    calib = get_calib(args, tok)
    ids0 = tok(calib[0], truncation=True, max_length=args.seqlen)["input_ids"]
    print(f"[rank1] {args.calib}: {len(calib)} samples; "
          f"prompt-0 tokens 0..7: {tok.convert_ids_to_tokens(ids0[:8])}")

    inps, kws = capture_layer0_inputs(model, tok, calib, args.seqlen)
    layers = model.model.layers
    last = max(targets)
    os.makedirs(args.hess_dir, exist_ok=True)
    found = []

    for li in range(last + 1):
        layer = layers[li]
        mods = {p: getattr(layer.self_attn if p in ATTN else layer.mlp, p)
                for p in targets.get(li, [])}
        if args.quantize_preceding and li < last:
            mods = {p: getattr(layer.self_attn if p in ATTN else layer.mlp, p)
                    for p in ATTN + ("gate_proj", "up_proj", "down_proj")}
        gptq = {p: MaskedGPTQ(m, name=f"layers.{li}.{p}") for p, m in mods.items()}
        probes = {p: BosProbe(m.weight.shape[1]) for p, m in mods.items()}

        if mods:
            def _hook(g, pr):
                def h(_m, a):        # must return None, or the args are replaced
                    g.add_batch(a[0])
                    pr.add(a[0])
                return h
            handles = [m.register_forward_pre_hook(_hook(gptq[p], probes[p]))
                       for p, m in mods.items()]
            for j in range(len(inps)):
                layer(inps[j], **kws[j])
            for h in handles:
                h.remove()

        for p in targets.get(li, []):
            g, pr = gptq[p], probes[p]
            path = os.path.join(args.hess_dir, f"H_L{li}_{p}.pt")
            torch.save({"H": g.H.cpu(), "bos_vec": pr.bos_vec(),
                        "n_samples": pr.n, "n_tokens": pr.n_tokens,
                        "xnorm_bos": pr.norm_bos / max(pr.n, 1),
                        "xnorm_rest": pr.norm_rest / max(pr.n, 1)}, path)
            print(f"[rank1] L{li} {p}: H {tuple(g.H.shape)} -> {path} "
                  f"(|x_BOS|={pr.norm_bos / max(pr.n, 1):.1f}, "
                  f"|x_rest|={pr.norm_rest / max(pr.n, 1):.2f})", flush=True)
            found.append((li, p, path))

        if args.quantize_preceding and li < last:
            for p, g in gptq.items():
                g.quantize(bits=args.bits, group_size=args.group_size,
                           percdamp=args.percdamp)
        for g in gptq.values():
            g.free()
        for j in range(len(inps)):
            out = layer(inps[j], **kws[j])
            inps[j] = out[0] if isinstance(out, tuple) else out
        print(f"[rank1] layer {li}/{last} forwarded", flush=True)

    del model, inps
    gc.collect()
    torch.cuda.empty_cache()
    return found


@torch.no_grad()
def analyze(path: str, device: str):
    d = torch.load(path, map_location="cpu")
    H, b = d["H"], d["bos_vec"]
    try:
        evals, evecs = torch.linalg.eigh(H.to(device))
    except Exception as e:  # noqa: BLE001  (OOM / no cuda -> cpu fallback)
        print(f"[rank1] eigh on {device} failed ({e}); falling back to cpu")
        evals, evecs = torch.linalg.eigh(H)
    evals, evecs = evals.cpu(), evecs.cpu()
    # eigh returns ascending eigenvalues; the leading pair is last.
    lam = evals.flip(0).double()
    v1 = evecs[:, -1].float()
    r1 = float(lam[0] ** 2 / (lam ** 2).sum())
    return {"dim": H.shape[0], "n_samples": d["n_samples"], "n_tokens": d["n_tokens"],
            "lambda_1": f"{float(lam[0]):.6g}", "lambda_2": f"{float(lam[1]):.6g}",
            "R1": round(r1, 6), "residual_ratio": round(max(0.0, 1 - r1) ** 0.5, 6),
            "cos_align": round(float(torch.dot(v1, b).abs()), 6),
            "xnorm_bos": round(d["xnorm_bos"], 3),
            "xnorm_rest": round(d["xnorm_rest"], 3),
            "xnorm_ratio": round(d["xnorm_bos"] / max(d["xnorm_rest"], 1e-9), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--targets", help='"1:down_proj" or "1:down_proj;16:down_proj"')
    ap.add_argument("--calib", default="c4",
                    choices=["c4", "instruct", "wikitext", "ultrachat", "c4chat", "c4wrongchat"])
    ap.add_argument("--data-source", default="calib",
                    help="label for the CSV, e.g. calib / deploy")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--hess-dir", help="where the H .pt files go (large: dim^2 fp32)")
    ap.add_argument("--out", help="CSV (appended if it exists)")
    ap.add_argument("--calib-file",
                    help="read calibration texts from this jsonl (one {'text': ...} per line) "
                         "instead of fetching them -- for compute nodes without network")
    ap.add_argument("--dump-calib",
                    help="write the calibration texts to this jsonl and exit. Run on a login "
                         "node (which has network), then pass the file back via --calib-file.")
    ap.add_argument("--eig-device", default="cuda")
    ap.add_argument("--quantize-preceding", action="store_true",
                    help="quantize the layers before the target first, so H is measured on "
                         "already-quantized upstream activations (a real GPTQ run). Default off: "
                         "the hypothesis is about an fp16-intrinsic property.")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.05)
    args = ap.parse_args()

    if args.dump_calib:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)   # tokenizer only: no GPU needed
        texts = load_calib(args.calib, tok, args.n_calib, args.seqlen, seed=args.calib_seed)
        with open(args.dump_calib, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t}) + "\n")
        print(f"[rank1] {len(texts)} {args.calib} texts -> {args.dump_calib}")
        return

    assert args.hess_dir and args.out and args.targets, \
        "--targets, --hess-dir and --out are required unless --dump-calib"
    targets = parse_targets(args.targets)
    found = collect(args, targets)

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for li, p, path in found:
            row = analyze(path, args.eig_device)
            row.update({"model": args.model, "layer": li, "matrix": p,
                        "data_source": args.data_source, "calib": args.calib,
                        "quantize_preceding": int(args.quantize_preceding),
                        "hess_path": path})
            w.writerow(row)
            print(f"[rank1] L{li} {p} ({args.data_source}): R1={row['R1']} "
                  f"residual={row['residual_ratio']} cos_align={row['cos_align']} "
                  f"lambda_1={row['lambda_1']} |x_BOS|/|x_rest|={row['xnorm_ratio']}",
                  flush=True)
    print(f"[rank1] -> {args.out}")


if __name__ == "__main__":
    main()
