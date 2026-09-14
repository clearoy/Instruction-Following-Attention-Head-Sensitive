# Shared preamble for the W51/W52 chain jobs. Unlike jobs/_w2x_header.sh this
# makes no assumption about whose account it runs in: python comes from an
# environment module (IFH_MODULE_LOAD) or a conda env (IFH_CONDA_ENV), paths come
# from the environment, and the corpora are read from prefetched jsonl because
# compute nodes here have no outbound network.
#
# The `#$ -V` in each job is what carries these variables in; without it SGE
# starts the job in a fresh environment and everything below silently defaults.
set -e
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
IFH_STORE="${IFH_STORE:-$HOME/ifh_store}"
IFH_CALIB_DIR="${IFH_CALIB_DIR:-data/calib_cache}"
IFH_OUT="${IFH_OUT:-roy_run}"   # this effort's results, kept out of runs/
IFH_CONDA_ENV="${IFH_CONDA_ENV-}"

LLAMA="${IFH_LLAMA:-meta-llama/Llama-3.1-8B-Instruct}"
Q14="${IFH_Q14:-Qwen/Qwen2.5-14B-Instruct}"
FULL="data/ifeval_input_data.jsonl"

source ~/.bashrc 2>/dev/null || true
source /etc/profile.d/modules.sh 2>/dev/null || true
if [ -n "${IFH_MODULE_LOAD:-}" ]; then
  module load $IFH_MODULE_LOAD || echo "[w5x] module load $IFH_MODULE_LOAD failed"
fi
if [ -n "$IFH_CONDA_ENV" ]; then
  conda activate "$IFH_CONDA_ENV" 2>/dev/null || source activate "$IFH_CONDA_ENV" 2>/dev/null \
    || echo "[w5x] could not activate $IFH_CONDA_ENV; using the ambient python"
fi
echo "[w5x] python: $(command -v python || echo MISSING)"
mkdir -p logs "$IFH_OUT" "$IFH_OUT/protocols" "$IFH_STORE"

ifh_cleanup () {
  pkill -TERM -P $$ 2>/dev/null || true
  sleep 3
  pkill -KILL -P $$ 2>/dev/null || true
}
trap ifh_cleanup TERM INT HUP USR1 USR2 EXIT

# short model key -> the prefetched calibration jsonl for that corpus.
# c4 is model-independent (the c4 branch never touches a tokenizer); c4chat is
# NOT -- it wraps the documents in that model's own chat template.
calib_file () {  # $1 corpus  $2 model key
  case "$1" in
    c4) echo "$IFH_CALIB_DIR/c4.jsonl" ;;
    *)  echo "$IFH_CALIB_DIR/$1_$2.jsonl" ;;
  esac
}

run_ifeval () {   # $1 ckpt-or-hf-id  $2 tag
  ${T:-} python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" --tag "$2" --batch 16
  ${T:-} python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "$IFH_OUT/scores_$2.csv"
}
