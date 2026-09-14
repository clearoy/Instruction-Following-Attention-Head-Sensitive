#!/bin/bash
#$ -M xmu2@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -q gpu
#$ -l gpu_card=1
#$ -l h_rt=6:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -V
#$ -o logs/
#$ -N IFH_W52
#$ -t 1-6
#$ -tc 3
# W52: the mechanistic half of the chain -- Hessian geometry and the GPTQ
# compensation residual at each model's sink-forming matrix, under c4 and under
# chat-template calibration. Each task emits TWO csv rows, `full` and
# `sink_removed`, so the three conditions the argument needs (c4, c4 minus the
# BOS outer products, chat) come out of two runs per model.
#
# Read-out (src/comp_residual.py): R1_trace = lam1/tr(H), R1_frob, lam1/lam2 and
# cos(v1, sink activation) for the geometry; then L_sink, L_template and their
# ratio for r_lam(z) = z_S - B_lam z_F, plus ||B z_F||/||z_S|| and cos(B z_F, z_S)
# to separate overshoot from wrong direction.
#
# Prediction being tested: H stays near-rank-1 and sink-aligned under BOTH
# calibrations, but L_template is large under c4 and small under chat -- and
# removing the sink from the c4 Hessian also collapses L_template. That would
# make the cure a change in the residual calibration geometry rather than a
# removal of the rank-1 dominance.
#
# Sink-forming matrices (RESULTS 9.10n/p): Llama L1 down_proj (BOS, norm 481),
# Mistral-v0.3 L1 down_proj (BOS, 807), Qwen2.5-14B L4 down_proj -- whose sink is
# NOT BOS but the first newline at position 2 (70 vs 4-25). comp_residual.py
# detects the sink position from the calibration norms rather than assuming BOS.
#
# Cheap: only layers 0..L are walked, so even Qwen-14B fits on a 24 GB card here
# (device_map spills the unused tail to host RAM). Unlike W51, which needs the
# whole model resident for generation.
#
#   qsub jobs/w52_residual.sh
#   awk 'FNR==1 && NR!=1 {next} 1' runs/comp_residual_*.csv > runs/comp_residual.csv
# SGE copies the job script to a spool dir, so $0 is NOT the original
# path -- locate the header from the submit directory instead.
source "${SGE_O_WORKDIR:-$PWD}/jobs/_w5x_header.sh" || { echo "header not found"; exit 3; }
T="timeout --signal=TERM --kill-after=120 5h"

resid () {  # $1 model  $2 model-key  $3 target  $4 corpus  $5 tag
  local model="$1" key="$2" target="$3" corpus="$4" tag="$5"
  local cf; cf="$(calib_file "$corpus" "$key")"
  [ -f "$cf" ] || { echo "[W52] missing prefetched corpus $cf -- dump it on a login node first"; exit 4; }
  # One z-cache per MODEL, shared by that model's calibration conditions, so the
  # deployment z are identical by inspection and not merely by construction.
  # Sibling tasks may race to create it; comp_residual.py writes it atomically
  # and the contents are identical either way (z comes from the unquantized model).
  $T python src/comp_residual.py --model "$model" --target "$target" \
      --calib "$corpus" --calib-file "$cf" --deploy "$FULL" \
      --z-cache "$IFH_STORE/zcache_$key.pt" \
      --out "runs/comp_residual_$tag.csv"
}

case "${SGE_TASK_ID:-1}" in
  1) resid "$LLAMA" llama "1:down_proj" c4     l_c4 ;;
  2) resid "$LLAMA" llama "1:down_proj" c4chat l_c4chat ;;
  3) resid "$Q14"   q14   "4:down_proj" c4     q_c4 ;;
  4) resid "$Q14"   q14   "4:down_proj" c4chat q_c4chat ;;
  5) resid "$M7"    m7    "1:down_proj" c4     m_c4 ;;
  6) resid "$M7"    m7    "1:down_proj" c4chat m_c4chat ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W52] done task ${SGE_TASK_ID:-1}"
