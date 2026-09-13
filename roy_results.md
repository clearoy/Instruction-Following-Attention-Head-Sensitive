# W50 — the calibration Hessian at the sink-forming matrix is rank one

Branch `roy_experiments`. Run 2026-09-13 on CRC (`gpu@qa-a10-*`, Llama-3.1-8B-Instruct,
pytorch/2.9.1 module, transformers 5.15.0 / datasets 5.0.1). Data: `runs/hess_rank1.csv`.

## Question

RESULTS §9.10n established positionally that at Llama-3.1-8B layer-1 `down_proj` the
token-averaged layer-wise objective is ~99.9% one token, and §9.10u argued the decisive
quantity is the *eigenvalue ratio* rather than the energy share. This run states the same
mechanism spectrally and measures it directly:

> Is `H_c = Σ_t x_t x_tᵀ` at the sink-forming matrix effectively rank one, and is the
> direction it spans the BOS direction?

If so, that is the structural reason GPTQ's compensation — which solves against `H_c` —
degenerates exactly there, and it is an fp16-intrinsic property (attention sink), not
something quantization creates.

## Method

`src/hessian_rank1.py` collects `H` through the pipeline's own accumulator
(`gptq_core.MaskedGPTQ.add_batch`, the same code path used during quantization), saves it,
then eigendecomposes it (`torch.linalg.eigh`, symmetric PSD).

- `R1 = λ₁² / Σᵢ λᵢ²` — Frobenius-squared share of the leading eigenpair
- `residual_ratio = √(1 − R1) = ‖H − λ₁v₁v₁ᵀ‖_F / ‖H‖_F`
- `cos_align = |⟨v₁, b⟩|`, where `b` is the **mean** BOS activation over the whole
  calibration set, normalised (not a single sample)
- `xnorm_bos` / `xnorm_rest` — mean activation norm at position 0 vs positions ≥ 1

Preceding layers are left in fp16 (`quantize_preceding=0`), which isolates the claim under
test. BOS is never dropped or edited: its outlier norm is a property of the position (first
token, nothing to attend to under the causal mask), so removing it would merely promote the
next position to sink. λᵢ carry the accumulator's `H = (2/N) Σ_t x_t x_tᵀ` scaling — a
running mean, comparable across corpora; `R1`, `residual_ratio` and `cos_align` are scale-free.

## Results

| Arm | λ₁ | λ₂ | λ₁/λ₂ | R1 | residual | cos_align | \|x_BOS\| | \|x_rest\| |
|---|---|---|---|---|---|---|---|---|
| **L1 c4** (target) | 1102.5 | 0.0456 | **24,200×** | **1.000000** | 5.5e-05 | **1.000** | 480.9 | 0.98 |
| L1 instruct | 19350.5 | 0.2054 | 94,200× | 1.000000 | 1.7e-05 | 1.000 | 480.9 | 11.28 |
| L1 ultrachat | 788.8 | 0.0470 | 16,800× | 1.000000 | 8.6e-05 | 1.000 | 480.9 | 1.37 |
| **L16 c4** (control) | 5.20 | 0.785 | 6.6× | 0.820 | 0.424 | **0.063** | 2.12 | 7.07 |

Chance floor for `cos_align` at dim 14336 is 1/√14336 ≈ 0.0084. All Hessians have
`n_tokens` ≫ `dim` (24.5k–150k vs 14336), so none is rank-deficient by construction.

## Findings

1. **The premise reproduces independently.** `|x_BOS|` = 480.9 against `|x_rest|` = 0.98 at
   L1 `down_proj`, measured here from scratch; §9.10n recorded 481 vs 1–2.5.

2. **`H_c` is rank one to five decimal places.** `residual_ratio` = 5.5e-05, i.e. the leading
   eigenpair explains 1 − 3×10⁻⁹ of ‖H‖²_F. After λ₁ = 1102 and λ₂ = 0.046 the remaining
   ~14,334 eigenvalues average ≈3×10⁻⁴. This is one direction plus numerical dust.

3. **The direction is BOS.** `cos_align` = 1.000 against a 0.0084 chance floor.

4. **The no-dominance control behaves entirely differently**: `cos_align` 0.063, λ₁/λ₂ of 6.6
   rather than 24,200, and BOS is *below* average norm there (2.12 vs 7.07).

5. **`R1` alone is not the discriminator — use `cos_align` and λ₁/λ₂.** The healthy layer still
   scores `R1` = 0.820, because Frobenius share is generous when the spectrum has a long tail.
   This is §9.10u's conclusion reached from the spectral side, and it should be stated
   explicitly: a reader shown only "1.000 vs 0.820" will badly underrate the gap, whereas
   λ₁/λ₂ separates the two by three and a half orders of magnitude.

6. **The dominance is corpus-independent — the predicted drop on deployment text did not
   happen.** `R1` = 1.0 and `cos_align` = 1.0 on all three L1 arms. `|x_BOS|` is *identical*
   (480.895) across c4, instruct and ultrachat, exactly as causal masking requires, since BOS
   attends to nothing — a strong internal check that the measurement is correct.

   The apparent fall in `|x_BOS|/|x_rest|` (490 → 42.6 for instruct) is an artefact of the
   **double-BOS** protocol deviation (§9.10j): the chat arms carry a second BOS at position 1,
   which the probe averages into "rest". The arithmetic closes to within 1%:

   | Arm | tokens/sample | predicted \|x_rest\| | measured |
   |---|---|---|---|
   | instruct | 47.8 | (480.9 + 45.8×1.03)/46.8 = 11.26 | 11.282 |
   | ultrachat | 1172.8 | (480.9 + 1170.8×0.96)/1171.8 = 1.37 | 1.374 |
   | c4 (single BOS) | 419.5 | — | 0.98 |

   The implied ordinary-token norm is 1.03 (instruct) and 0.96 (ultrachat), both consistent
   with c4's 0.98.

   **Consequence for the project's story:** chat calibration cannot be curing the collapse by
   diluting BOS out of the Hessian — the dominance is unchanged under chat calibration. It
   must act through the other channel RESULTS already argues for, putting curvature into
   template-token directions that c4 never populates. This run rules out one tempting
   alternative explanation rather than contradicting the existing account.

## What this does not establish

Structure, not causation. It cannot show that rank-one-ness *causes* the collapse; that link
rests on the existing intervention arms (token-normalised Hessian, single-module RTN swap,
chat calibration), which this result explains rather than re-proves. It is also one model,
one matrix, one seed — the 14-of-16-model generality claim comes from the §9.10q census, not
from here.

## Reproduce

```
qsub -t 1-4 jobs/w50_hess_rank1.sh       # arms: l1_calib, l1_deploy, l16_calib, l1_chat
awk 'FNR==1 && NR!=1 {next} 1' runs/hess_rank1_*.csv > runs/hess_rank1.csv
```

Compute nodes without outbound network need the streamed corpora prefetched on a login node
first (`--dump-calib`); see the job header. The control layer (16) was chosen from
`runs/stats/llama31-8b-pos/stats.csv`: L1's template/ordinary input-norm ratio is 123×, L16's
is 0.92×, and L0/L31 are both anomalous.

**Provenance note:** `runs/hess_rank1.csv` in this branch was transcribed from the CRC job
output rather than copied off the cluster; the `hess_path` column points at
`/users/xmu2/ifh_store/...` on CRC, where the saved `H` matrices (≈820 MB each) still live.
Re-running the two commands above regenerates it.
