# W50 — the calibration Hessian at the sink-forming matrix is rank one

Llama-3.1-8B-Instruct, run 2026-09-13 on CRC. Data: `runs/hess_rank1.csv`.
Code: `src/hessian_rank1.py`, `jobs/w50_hess_rank1.sh`.

## Question

Is `H_c = Σ_t x_t x_tᵀ` at layer-1 `down_proj` effectively rank one, and is the direction
it spans the BOS direction? If so, that is the structural reason GPTQ's compensation —
which solves against `H_c` — degenerates there.

`R1 = λ₁²/Σλᵢ²`, `residual = √(1−R1)`, `cos_align = |⟨v₁, b⟩|` with `b` the normalised mean
BOS activation. `H` is collected through the pipeline's own accumulator
(`MaskedGPTQ.add_batch`); preceding layers stay fp16, so this measures an fp16-intrinsic
property, not something quantization creates.

## Results

| Arm | λ₁ | λ₂ | λ₁/λ₂ | R1 | residual | cos_align | \|x_BOS\| | \|x_rest\| |
|---|---|---|---|---|---|---|---|---|
| **L1 c4** (target) | 1102.5 | 0.0456 | **24,200×** | **1.000000** | 5.5e-05 | **1.000** | 480.9 | 0.98 |
| L1 instruct | 19350.5 | 0.2054 | 94,200× | 1.000000 | 1.7e-05 | 1.000 | 480.9 | 11.28 |
| L1 ultrachat | 788.8 | 0.0470 | 16,800× | 1.000000 | 8.6e-05 | 1.000 | 480.9 | 1.37 |
| **L16 c4** (control) | 5.20 | 0.785 | 6.6× | 0.820 | 0.424 | **0.063** | 2.12 | 7.07 |

Chance floor for `cos_align` at dim 14336 is 0.0084. All arms have `n_tokens` ≫ `dim`
(24.5k–150k), so no Hessian is rank-deficient by construction.

## Findings

1. **Premise reproduced independently:** `|x_BOS|` 480.9 vs `|x_rest|` 0.98, against the
   481 vs 1–2.5 recorded in §9.10n.

2. **`H_c` is rank one to five decimal places** and **the direction is BOS.** The residual is
   5.5e-05; after λ₁ = 1102 and λ₂ = 0.046 the remaining ~14,334 eigenvalues average ≈3e-04.
   `cos_align` = 1.000 against a 0.0084 floor. The control is nothing like it: `cos_align`
   0.063, λ₁/λ₂ of 6.6, and BOS *below* average norm.

3. **`R1` alone is a weak discriminator — use `cos_align` and λ₁/λ₂.** The healthy layer still
   scores 0.820, because Frobenius share is generous against a long tail. Same conclusion as
   §9.10u, reached from the spectral side.

4. **The dominance is corpus-independent; the predicted drop on deployment text did not
   happen.** `R1` and `cos_align` are 1.0 on all three L1 arms, and `|x_BOS|` is identical
   (480.895) across c4, instruct and ultrachat — as causal masking requires, since BOS attends
   to nothing. The apparent fall in `|x_BOS|/|x_rest|` (490 → 42.6) is the double-BOS deviation
   (§9.10j): the chat arms put a second BOS at position 1, which the probe averages into
   "rest". Predicted `|x_rest|` 11.26 vs measured 11.282 (instruct), 1.37 vs 1.374 (ultrachat).

   **Consequence:** chat calibration cannot be curing collapse by diluting BOS out of the
   Hessian — the dominance is unchanged under it. It must act through the channel RESULTS
   already argues for, adding curvature in template-token directions c4 never populates.

   Both corpora tested here are curing arms (instruct = `v2l_calinst` .611, ultrachat =
   `v2l_calchat` .669, against `v2l_none` .150), so the conclusion stands on curing sets. But
   it is not yet the cleanest test: `c4chat` (.644) holds the c4 content fixed and changes
   only whether this model's template tokens are present, and `c4wrongchat` (.162) is the
   foreign-template arm that does **not** cure. Identical dominance across that pair — same
   content, same token budget, opposite IFEval outcomes — would settle it. Added as arms 5–6;
   **not yet run**.

## Limits

Structure, not causation: the causal link rests on the existing intervention arms
(token-normalised Hessian, single-module RTN, chat calibration), which this explains rather
than re-proves. One model, one matrix, one seed.

## Reproduce

```
qsub -t 1-4 jobs/w50_hess_rank1.sh
awk 'FNR==1 && NR!=1 {next} 1' runs/hess_rank1_*.csv > runs/hess_rank1.csv
```

Offline compute nodes need the streamed corpora prefetched on a login node (`--dump-calib`).
Control layer 16 chosen from `runs/stats/llama31-8b-pos/stats.csv`: L1's template/ordinary
norm ratio is 123×, L16's is 0.92×, L0/L31 anomalous.

`runs/hess_rank1.csv` was transcribed from the job output; the saved `H` matrices (~820 MB
each) stay on CRC under `~/ifh_store/hessians/`.
