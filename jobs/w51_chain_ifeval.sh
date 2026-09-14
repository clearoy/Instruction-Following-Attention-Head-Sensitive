#!/bin/bash
#$ -M xmu2@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -q gpu
#$ -l gpu_card=1
#$ -l h_rt=24:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -V
#$ -o logs/
#$ -N IFH_W51
#$ -t 1-9
#$ -tc 1
# W51: the IFEval end of the mechanism chain, three models x three calibration
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
#   M7v0.3 fp16 .552  RTN3 .420  c4 .468  c4chat --     dropbos --
# Those used --hess-drop-pos 2 for Llama, which on c4 also drops one ordinary
# token; this job uses 1, which is the exact BOS removal.
#
# NOTE ON QWEN (tasks 4-6): Qwen2.5-14B in bf16 is ~28 GB, so generation does not
# fit on one 24 GB A10 or RTX 4500. It does not need a bigger card, only more of
# them: common.load_model uses device_map="auto", which shards across every
# visible GPU, so two cards give 48 GB and the model fits natively.
#   qsub -l gpu_card=2 -t 4-6 jobs/w51_chain_ifeval.sh
# Only the Qwen arms need this; Llama (16 GB) and Mistral (14.5 GB) fit on one.
#
# WHY -tc 1 AND NOT -tc 3: each arm materialises a fake-quant checkpoint the size
# of the model (Llama 16 GB, Qwen 28 GB, Mistral 14.5 GB) and deletes it after
# scoring. With all three models downloaded (58.5 GB) a 100 GB home has ~34 GB
# spare, so two concurrent arms is already marginal and three is not possible.
# Raise it at submit time (`qsub -tc 3`) only if IFH_STORE points somewhere with
# room for three checkpoints at once. W52 has no such limit and ships with -tc 3.
#
#   qsub -t 1-3 jobs/w51_chain_ifeval.sh          # Llama only
#   qsub -t 7-9 jobs/w51_chain_ifeval.sh          # Mistral only
#   awk 'FNR==1 && NR!=1 {next} 1' runs/scores_w51_*.csv > runs/scores_w51.csv
# SGE copies the job script to a spool dir, so $0 is NOT the original
# path -- locate the header from the submit directory instead.
source "${SGE_O_WORKDIR:-$PWD}/jobs/_w5x_header.sh" || { echo "header not found"; exit 3; }
T="timeout --signal=TERM --kill-after=120 23h"

arm () {  # $1 model  $2 model-key  $3 tag  $4 corpus  $5.. extra flags
  local model="$1" key="$2" tag="$3" corpus="$4"; shift 4
  local ckpt="$IFH_STORE/models/$key-w51-$tag"
  local cf; cf="$(calib_file "$corpus" "$key")"
  [ -f "$cf" ] || { echo "[W51] missing prefetched corpus $cf -- dump it on a login node first"; exit 4; }
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 \
      --protect none --calib "$corpus" --calib-file "$cf" "$@" --out "$ckpt"
  run_ifeval "$ckpt" "w51_$tag"
  cp "$ckpt/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$ckpt").json" 2>/dev/null || true
  rm -rf "$ckpt"          # 16-28 GB each; home quota is 100 GB
}

case "${SGE_TASK_ID:-1}" in
  1) arm "$LLAMA" llama  l_c4      c4 ;;
  2) arm "$LLAMA" llama  l_c4chat  c4chat ;;
  3) arm "$LLAMA" llama  l_dropbos c4     --hess-drop-pos 1 ;;
  4) arm "$Q14"   q14    q_c4      c4 ;;
  5) arm "$Q14"   q14    q_c4chat  c4chat ;;
  6) arm "$Q14"   q14    q_dropbos c4     --hess-drop-pos 1 ;;
  7) arm "$M7"    m7     m_c4      c4 ;;
  8) arm "$M7"    m7     m_c4chat  c4chat ;;
  9) arm "$M7"    m7     m_dropbos c4     --hess-drop-pos 1 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W51] done task ${SGE_TASK_ID:-1}"
