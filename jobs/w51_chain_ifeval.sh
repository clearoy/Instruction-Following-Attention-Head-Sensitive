#!/bin/bash
#$ -M xmu2@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -q gpu
#$ -l gpu_card=2
#$ -l h_rt=24:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -V
#$ -o logs/
#$ -N IFH_W51
#$ -t 1-6
#$ -tc 1
# W51: the IFEval end of the mechanism chain, two models x three calibration
# Hessians. 3-bit g128, protect none, frozen protocol otherwise.
#
#   c4       the collapse condition (BOS-dominated calibration geometry)
#   c4chat   same documents, this model's chat template -> the cure
#   dropbos  same c4 documents with the BOS position excluded from H
#            (--hess-drop-pos 1: on raw c4 the tokenizer adds exactly one BOS at
#            position 0, so this is H - x_B x_B^T positionally). The causal arm.
#
# Reference values already in the repo under the frozen protocol, for comparison:
#   Llama  fp16 .768  RTN3 .565  c4 .150  c4chat .644  dropbos(pos 2) .560
#   Q14    fp16 .820  RTN3 .697  c4 .412  c4chat .772  dropbos --
# Those used --hess-drop-pos 2 for Llama, which on c4 also drops one ordinary
# token; this job uses 1, which is the exact BOS removal.
#
# WHY gpu_card=2: on a 24 GB card even the Llama arm OOMs inside GPTQ. The
# budget is model weights (16 GB) + the 128 captured calibration activations
# (~2 GB, resident) + the layer-1 down_proj Hessian (822 MB) + the act-order
# permutation's second copy of it -- and that last allocation is the one that
# fails. quantize_protected.py was written for 141 GB H200s. device_map="auto"
# shards across every visible GPU, so two cards give 48 GB and all three models
# fit; Qwen (28 GB weights) needs them most. On a large-memory card, override
# with `qsub -l gpu_card=1`.
#
# WHY -tc 1 AND NOT -tc 3: each arm materialises a fake-quant checkpoint the size
# of the model (Llama 16 GB, Qwen 28 GB) and deletes it after
# scoring. With all three models downloaded (58.5 GB) a 100 GB home has ~34 GB
# spare, so two concurrent arms is already marginal.
# Raise it at submit time (`qsub -tc 3`) only if IFH_STORE points somewhere with
# room for two checkpoints at once. W52 has no such limit and ships with -tc 3.
#
#   qsub -t 1-3 jobs/w51_chain_ifeval.sh          # Llama only
#   qsub -t 4-6 jobs/w51_chain_ifeval.sh          # Qwen only
#   awk 'FNR==1 && NR!=1 {next} 1' roy_run/scores_w51_*.csv > roy_run/scores_w51.csv
# SGE copies the job script to a spool dir, so $0 is NOT the original
# path -- locate the header from the submit directory instead.
source "${SGE_O_WORKDIR:-$PWD}/jobs/_w5x_header.sh" || { echo "header not found"; exit 3; }
T="timeout --signal=TERM --kill-after=120 23h"

arm () {  # $1 model  $2 model-key  $3 tag  $4 corpus  $5.. extra flags
  local model="$1" key="$2" tag="$3" corpus="$4"; shift 4
  local ckpt="$IFH_STORE/models/$key-w51-$tag"
  local cf; cf="$(calib_file "$corpus" "$key")"
  [ -f "$cf" ] || { echo "[W51] missing prefetched corpus $cf -- dump it on a login node first"; exit 4; }
  # An arm that dies between quantization and scoring leaves the whole checkpoint
  # behind, because the rm below only runs on success. Clear any stale copy first
  # or a couple of failures will exhaust the disk quota.
  rm -rf "$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 \
      --protect none --calib "$corpus" --calib-file "$cf" "$@" --out "$ckpt"
  run_ifeval "$ckpt" "w51_$tag"
  cp "$ckpt/PROTECT_PROTOCOL.json" "$IFH_OUT/protocols/$(basename "$ckpt").json" 2>/dev/null || true
  rm -rf "$ckpt"          # 16-28 GB each; home quota is 100 GB
}

case "${SGE_TASK_ID:-1}" in
  1) arm "$LLAMA" llama  l_c4      c4 ;;
  2) arm "$LLAMA" llama  l_c4chat  c4chat ;;
  3) arm "$LLAMA" llama  l_dropbos c4     --hess-drop-pos 1 ;;
  # Qwen-14B is 29.5 GB of weights against 2 x 22.3 GB. Neither device_map "auto"
  # nor "balanced" leaves GPU 0 room for the 2.7 GB of captured activations plus
  # the Hessian, so pin the split: 13 + 17 covers the weights and leaves ~9 GB
  # free on GPU 0 and ~5 GB on GPU 1. Llama (16 GB) needs none of this.
  4|5|6) export IFH_MAX_MEMORY="${IFH_MAX_MEMORY:-0:13GiB,1:17GiB,cpu:40GiB}"
         export IFH_LOG_PLACEMENT=1
         case "$SGE_TASK_ID" in
           4) arm "$Q14" q14 q_c4      c4 ;;
           5) arm "$Q14" q14 q_c4chat  c4chat ;;
           6) arm "$Q14" q14 q_dropbos c4 --hess-drop-pos 1 ;;
         esac ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W51] done task ${SGE_TASK_ID:-1}"
