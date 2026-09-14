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
#$ -V
#$ -o logs/
#$ -N IFH_W50
#$ -t 1-6
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
# This job does NOT source jobs/_w2x_header.sh: it loads an fp16 HF model, so it
# needs neither gptqmodel nor the pinned 3-bit stack, and it must run for users
# other than jzheng7. Override anything via the environment, e.g.
#   IFH_STORE=$HOME/ifh HF_HOME=$HOME/hf IFH_MODULE_LOAD=pytorch/2.9.1 \
#     qsub -q <your-gpu-queue> -M you@nd.edu -t 1 jobs/w50_hess_rank1.sh
# The `#$ -V` above is what makes those overrides reach the job: without it SGE
# starts the job in a fresh environment and every IFH_* variable is silently
# lost (symptom: "[W50] python: MISSING" because the module was never loaded).
#
#   qsub -t 1 jobs/w50_hess_rank1.sh     # verification arm first
#   qsub    jobs/w50_hess_rank1.sh       # all four
#   awk 'FNR==1 && NR!=1 {next} 1' roy_run/hess_rank1_*.csv > roy_run/hess_rank1.csv
set -e
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/hf}"
# Compute nodes may have no outbound network. Weights must then be pre-fetched
# on a login node, and so must the streamed corpora (see IFH_CALIB_DIR below).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
IFH_CALIB_DIR="${IFH_CALIB_DIR:-data/calib_cache}"
IFH_OUT="${IFH_OUT:-roy_run}"
IFH_STORE="${IFH_STORE:-$HOME/ifh_store}"     # Hessians land here (~820 MB each)
IFH_MODEL="${IFH_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
IFH_CONDA_ENV="${IFH_CONDA_ENV-}"   # empty = module-provided python, no conda
source ~/.bashrc 2>/dev/null || true
# CRC serves python/conda through environment modules, and which ones exist
# differs per account, so take them from the submitting environment:
#   IFH_MODULE_LOAD="conda" qsub ...
source /etc/profile.d/modules.sh 2>/dev/null || true
if [ -n "${IFH_MODULE_LOAD:-}" ]; then
  module load $IFH_MODULE_LOAD || echo "[W50] module load $IFH_MODULE_LOAD failed"
fi
if [ -n "$IFH_CONDA_ENV" ]; then
  conda activate "$IFH_CONDA_ENV" 2>/dev/null \
    || source activate "$IFH_CONDA_ENV" 2>/dev/null \
    || echo "[W50] could not activate $IFH_CONDA_ENV; using the ambient python"
fi
echo "[W50] python: $(command -v python || echo MISSING)"
# HF token for the gated Llama weights: jobs/hf.env if present, else the
# ambient HF_TOKEN / the cached huggingface-cli login.
source jobs/hf.env 2>/dev/null || true
mkdir -p logs "$IFH_OUT" "$IFH_STORE"

# Orphan guard: on qdel, timeout, or normal exit kill every child so no python
# or CUDA process outlives the job holding the card.
ifh_cleanup () {
  pkill -TERM -P $$ 2>/dev/null || true
  sleep 3
  pkill -KILL -P $$ 2>/dev/null || true
}
trap ifh_cleanup TERM INT HUP USR1 USR2 EXIT

T="timeout --signal=TERM --kill-after=120 3h"

rank1 () {  # $1 tag  $2 targets  $3 calib  $4 data-source  $5.. extra flags
  local tag="$1" targets="$2" calib="$3" src="$4"; shift 4
  # Prefer a prefetched corpus when one is there; the texts are identical.
  local cf=""
  [ -f "$IFH_CALIB_DIR/$calib.jsonl" ] && cf="--calib-file $IFH_CALIB_DIR/$calib.jsonl"
  $T python src/hessian_rank1.py --model "$IFH_MODEL" --targets "$targets" \
      --calib "$calib" --data-source "$src" $cf \
      --hess-dir "$IFH_STORE/hessians/$tag" --out "$IFH_OUT/hess_rank1_$tag.csv" "$@"
}

case "${SGE_TASK_ID:-1}" in
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
  # 5-6: the arms that actually decide whether BOS dominance is the variable that
  #      separates a curing calibration set from a non-curing one. Same c4 content
  #      in both; 5 wraps it in THIS model's chat template (Llama IFEval .644 vs
  #      .150 for plain c4 -- cures), 6 wraps it in a FOREIGN template (.162 --
  #      does not cure). Identical dominance across the two would show the cure
  #      does not run through diluting BOS out of the Hessian.
  5) rank1 l1_c4chat      "1:down_proj" c4chat      calib ;;
  6) rank1 l1_c4wrongchat "1:down_proj" c4wrongchat calib ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W50] done task ${SGE_TASK_ID:-1}"
