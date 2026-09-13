#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=4:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W50
#$ -t 1-4
# W50: is the calibration Hessian of the sink-forming matrix rank one?
#
# Theory under test: at Llama-3.1-8B layer-1 down_proj the BOS activation norm
# is ~481 against 1-2.5 elsewhere (fp16-intrinsic, attention sink; RESULTS
# 9.10n). Since H_c = sum_t x_t x_t^T weights each token by ||x_t||^2, that one
# token should supply essentially all of H_c -- making H_c effectively rank one
# and the direction it spans the BOS direction. That is the structural reason
# GPTQ's compensation, which solves against H_c, degenerates there.
#
# Read-out per arm: R1 (Frobenius-squared share of the leading eigenpair),
# residual_ratio = sqrt(1-R1), cos_align = |<v1, mean BOS activation>|.
# Prediction: arm 1 R1 -> 1 and cos_align high; arms 2-4 markedly lower.
#
#   qsub -t 1 jobs/w50_hess_rank1.sh     # verification arm first
#   qsub    jobs/w50_hess_rank1.sh       # all four
#   awk 'FNR==1 && NR!=1 {next} 1' runs/hess_rank1_*.csv > runs/hess_rank1.csv
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 3h"
HDIR="$STORE/hessians"     # ~820 MB per down_proj Hessian (14336^2 fp32)

rank1 () {  # $1 tag  $2 targets  $3 calib  $4 data-source  $5.. extra flags
  local tag="$1" targets="$2" calib="$3" src="$4"; shift 4
  $T python src/hessian_rank1.py --model "$LLAMA" --targets "$targets" \
      --calib "$calib" --data-source "$src" \
      --hess-dir "$HDIR/llama31-8b-$tag" --out "runs/hess_rank1_$tag.csv" "$@"
}

case "$SGE_TASK_ID" in
  # 1: the suspected collapse matrix on the frozen calibration corpus.
  1) rank1 l1_calib  "1:down_proj"  c4        calib ;;
  # 2: same matrix, deployment-style text (the IF calibration prompts through
  #    the chat template -- the repo's existing deploy prompt set). Short
  #    prompts, so use all 512 to keep the token count off the floor; n_tokens
  #    is recorded in the CSV because a Hessian estimated from fewer tokens
  #    than its dimension (14336) is rank-deficient by construction.
  2) rank1 l1_deploy "1:down_proj"  instruct  deploy --n-calib 512 ;;
  # 3: healthy control -- same matrix type, a layer with no dominance. Chosen
  #    from runs/stats/llama31-8b-pos: L1 template/ordinary norm ratio is 123x,
  #    L16 is 0.92x (and L0/L31 are anomalous, so mid-network it is).
  3) rank1 l16_calib "16:down_proj" c4        calib ;;
  # 4: token-matched deploy control (128 x 2048 multi-turn chat), so that
  #    arm 2's short-prompt token budget cannot be the explanation for a lower
  #    R1. Same corpus the W31 calibration arm used.
  4) rank1 l1_chat   "1:down_proj"  ultrachat deploy ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W50] done task $SGE_TASK_ID"
