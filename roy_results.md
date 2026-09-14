# Results — W50 and W52

Runs on CRC, 2026-09-13/14. Data in `roy_run/`. Code: `src/hessian_rank1.py`,
`src/comp_residual.py`; jobs `jobs/w50_hess_rank1.sh`, `jobs/w52_residual.sh`.
Models: Llama-3.1-8B-Instruct and Qwen2.5-14B-Instruct, fp16, no quantization applied.

Mistral-7B-Instruct-v0.3 was run and then withdrawn: its sink removal is incomplete
(a second massive-activation token class sits outside the 8-position window the code
inspects), so those numbers are not comparable to the other two and are not reported
here. See `jobs/w52_residual.sh` for the arithmetic.

---

## W50 — rank-one structure of the calibration Hessian

Target Llama-3.1-8B-Instruct layer-1 `down_proj` (dim 14336), plus layer 16 as a
no-dominance control. `H` collected through `MaskedGPTQ.add_batch`, preceding layers
left in fp16. `R1 = λ₁²/Σλᵢ²`, `residual = √(1−R1)`, `cos_align = |⟨v₁, b⟩|` with `b`
the normalised mean BOS activation. Chance floor for `cos_align` at dim 14336 = 0.0084.

**All columns are calibration-side.** W50 has no deployment component: `H`, the
eigenpair, and the BOS/rest norms all come from the calibration corpus named in the
Arm column.

| Arm | n_tokens | λ₁ | λ₂ | λ₁/λ₂ | R1 | residual | cos_align | \|x_BOS\| | \|x_rest\| | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| L1, c4 | 53700 | 1102.52 | 0.0455823 | 24,188 | 1.000000 | 5.5e-05 | 1.000 | 480.895 | 0.98 | 490.5 |
| L1, instruct | 24476 | 19350.5 | 0.205362 | 94,226 | 1.000000 | 1.7e-05 | 1.000 | 480.895 | 11.282 | 42.6 |
| L1, ultrachat | 150119 | 788.77 | 0.0470255 | 16,773 | 1.000000 | 8.6e-05 | 1.000 | 480.895 | 1.374 | 350.1 |
| L16, c4 | 53700 | 5.20047 | 0.78483 | 6.63 | 0.820054 | 0.424201 | 0.063 | 2.117 | 7.071 | 0.3 |

`|x_BOS|` = 480.895 in all three L1 arms.

---

## W52 — Hessian geometry and GPTQ compensation residual

Target matrices: Llama L1 `down_proj` (dim 14336), Qwen2.5-14B L4 `down_proj`
(13824). Calibration 128 × 2048; deployment z from 64 IFEval prompts through each
model's chat template, cached per model and shared across calibration conditions.
`percdamp` 0.05, act-order on.

`r_λ(z) = z_S − H_SF(H_FF+λI)⁻¹z_F` evaluated on GPTQ's sequential split
(S={i}, F={i+1..n} at step i) via `r_i = (Uz)_i/U_ii`, U the upper Cholesky of
(H+λI)⁻¹. `amp` = ‖r‖/‖z‖; `pred` = ‖B_λz_F‖/‖z_S‖; `cos` = cos(B_λz_F, z_S).
`L` = mean ‖r‖². `sink_removed` rebuilds the regressor from
H − Σ_{t∈sink} x_t x_tᵀ. Sink positions detected from activation norms separately
on the calibration and deployment sides, within the first 8 positions.
H accumulated in float64.

### Hessian geometry

**All columns are calibration-side**: the spectrum of `H`, and `cos_v1_sink` between
`v₁` and the mean calibration sink activation.

| model | calib | hessian | λ₁ | λ₂ | trace | R1_trace | λ₁/λ₂ | cos_v1_sink |
|---|---|---|---|---|---|---|---|---|
| Llama | c4 | full | 1103 | 0.04553 | 1104 | 0.998211 | 24,213 | 1.000 |
| Llama | c4 | sink_removed | 0.09462 | 0.01621 | 2.027 | 0.046677 | 5.84 | 0.727 |
| Llama | c4chat | full | 2039 | 0.04265 | 2041 | 0.999138 | 47,802 | 1.000 |
| Llama | c4chat | sink_removed | 0.07723 | 0.02435 | 1.797 | 0.042969 | 3.17 | 0.691 |
| Qwen | c4 | full | 2.693e5 | 106.7 | 2.695e5 | 0.999062 | 2523 | 1.000 |
| Qwen | c4 | sink_removed | 1.714 | 0.5627 | 124.8 | 0.013735 | 3.05 | 0.002 |
| Qwen | c4chat | full | 19.44 | 7.005 | 149.0 | 0.130485 | 2.78 | 0.874 |
| Qwen | c4chat | sink_removed | 3.046 | 1.826 | 123.2 | 0.024730 | 1.67 | 0.002 |

### Residuals

**All columns are deployment-side.** Calibration enters only by fixing the regressor:
`B_λ` is built from `H` (or `H − Σ x_t x_tᵀ`), then evaluated on the deployment `z`.
`L_*`, `amp_*`, `pred_template` and `cos_pred_template` are all measured over the 64
IFEval prompts' first 8 positions, split by the deployment-side sink detection. This
off-distribution evaluation is the step in THEORY_BRIEF_v2 §2.

| model | calib | hessian | L_sink | L_template | L_ratio | amp_sink | amp_template | pred_template | cos_pred_template |
|---|---|---|---|---|---|---|---|---|---|
| Llama | c4 | full | 4.917 | 6.142 | 1.249 | 0.0046 | 1.768 | 1.347 | 0.120 |
| Llama | c4 | sink_removed | 286957 | 2.390 | 8.3e-06 | 1.114 | 1.181 | 0.794 | 0.147 |
| Llama | c4chat | full | 5.512 | 1.830 | 0.332 | 0.0049 | 1.101 | 0.823 | 0.621 |
| Llama | c4chat | sink_removed | 288346 | 0.01188 | 4.1e-08 | 1.117 | 0.101 | 0.966 | 0.995 |
| Qwen | c4 | full | 117183 | 507.15 | 0.0043 | 4.234 | 1.977 | 1.590 | 0.010 |
| Qwen | c4 | sink_removed | 8466.6 | 521.68 | 0.0616 | 1.888 | 1.353 | 1.089 | 0.215 |
| Qwen | c4chat | full | 0.3784 | 0.3435 | 0.908 | 0.0126 | 0.071 | 0.973 | 0.997 |
| Qwen | c4chat | sink_removed | 4602.9 | 0.2934 | 6.4e-05 | 1.155 | 0.065 | 0.975 | 0.997 |

### Detected sink positions and norms

Mixed: `sink (calib)`, `|x_sink|` and `|x_template|` are calibration-side;
`sink (deploy)` and `sink tokens` are deployment-side. The two need not agree — for
Qwen under c4 they are different tokens, which is why its calibration `|x_sink|` is
7569 while the same position measures 38 on deployment input. Llama sinks at BOS on
both sides, so its two figures coincide.

| model | calib | sink (calib) | sink (deploy) | sink tokens | \|x_sink\| calib | \|x_template\| calib |
|---|---|---|---|---|---|---|
| Llama | c4 | 0 | 0\|1 | `<\|begin_of_text\|>` | 480.895 | 1.098 |
| Llama | c4chat | 0\|1 | 0\|1 | `<\|begin_of_text\|>` | 480.895 | 1.184 |
| Qwen | c4 | 0 | 0\|2 | `<\|im_start\|>`, `Ċ` | 7569.18 | 9.556 |
| Qwen | c4chat | 0\|2 | 0\|2 | `<\|im_start\|>`, `Ċ` | 52.228 | 12.155 |

Consistency check on the full Hessians, λ₁ = (2/N)·k·κ² from the detected sink tokens:
Llama predicts 1102.5 against 1102.5 measured; Qwen predicts 269,241 against 269,280.

### Per-position, deployment window, full Hessian

**Deployment-side**, per position of the deployment window; `sink` marks the
deployment-side detection. The regressor behind `amp` and `cos` is still the
calibration Hessian named in the column heading.

Llama-3.1-8B-Instruct, L1 `down_proj`:

| pos | token | sink | \|z\| | amp (c4) | cos (c4) | amp (c4chat) | cos (c4chat) |
|---|---|---|---|---|---|---|---|
| 0 | `<\|begin_of_text\|>` | ● | 480.89 | 0.005 | 1.000 | 0.005 | 1.000 |
| 1 | `<\|begin_of_text\|>` | ● | 480.89 | 0.005 | 1.000 | 0.005 | 1.000 |
| 2 | `<\|start_header_id\|>` | | 2.19 | 1.749 | −0.008 | 0.623 | 0.809 |
| 3 | `system` | | 1.07 | 1.121 | 0.121 | 0.797 | 0.630 |
| 4 | `<\|end_header_id\|>` | | 0.93 | 4.508 | 0.112 | 2.761 | 0.183 |
| 5 | `ĊĊ` | | 1.01 | 1.228 | 0.086 | 0.842 | 0.541 |
| 6 | `Cut` | | 0.90 | 0.967 | 0.255 | 0.804 | 0.846 |
| 7 | `ting` | | 1.01 | 1.035 | 0.155 | 0.779 | 0.719 |

Qwen2.5-14B-Instruct, L4 `down_proj`:

| pos | token | sink | \|z\| | amp (c4) | cos (c4) | amp (c4chat) | cos (c4chat) |
|---|---|---|---|---|---|---|---|
| 0 | `<\|im_start\|>` | ● | 38.07 | 1.208 | 0.000 | 0.016 | 1.000 |
| 1 | `system` | | 13.62 | 2.764 | 0.001 | 0.036 | 0.999 |
| 2 | `Ċ` | ● | 66.38 | 7.260 | −0.000 | 0.009 | 1.000 |
| 3 | `You` | | 8.82 | 2.287 | 0.004 | 0.065 | 0.998 |
| 4 | `Ġare` | | 12.61 | 1.149 | 0.017 | 0.053 | 0.999 |
| 5 | `ĠQ` | | 7.74 | 1.356 | 0.017 | 0.070 | 0.998 |
| 6 | `wen` | | 26.22 | 1.032 | 0.003 | 0.018 | 1.000 |
| 7 | `,` | | 3.95 | 3.274 | 0.016 | 0.182 | 0.985 |

---

## Existing IFEval reference values

From `runs/`, frozen protocol, 3-bit g128. End-to-end generation scores, so neither
calibration- nor deployment-side in the sense above — the calibration corpus is what
the column names, the score is IFEval over 541 prompts. These are the existing
measurements; W50/W52 did not re-measure them.

| model | fp16 | RTN3 | GPTQ3 c4 | GPTQ3 c4chat | GPTQ3 drop-BOS |
|---|---|---|---|---|---|
| Llama-3.1-8B | .768 | .565 | .150 | .644 | .560 |
| Qwen2.5-14B | .820 | .697 | .412 | .772 | — |

---

## Verification

- `--self-test` checks four invariants with no model: the sequential residual against
  the direct block formula; `Accum` against `MaskedGPTQ._accum`; the sink outer-product
  removal against a direct sink-free accumulation; and `r_λ`'s invariance to a global
  rescaling of H.
- 8 aggregate rows, all model × calibration × Hessian combinations present, no
  duplicates, no empty or non-finite fields.
- Four per-token files at 1024 rows each (64 prompts × 8 positions × 2 Hessian
  conditions); aggregates recomputed from them match the CSV to four decimals.

## Files

```
roy_run/hess_rank1.csv                             W50, 4 rows
roy_run/comp_residual.csv                          W52, 8 rows
roy_run/comp_residual_{l,q}_{c4,c4chat}.csv        per-arm, 2 rows each
roy_run/comp_residual_{l,q}_{c4,c4chat}.tokens.csv per-token, 1024 rows each
```

Saved Hessians (~820 MB each) remain on CRC under `~/ifh_store/`.

Note when parsing the per-token files: Qwen's position-7 token is a literal comma,
quoted by `csv.writer`, so `awk -F,` shifts every column after it. Use a real CSV
parser.
