#!/usr/bin/env bash
set -u

ROOT=${ROOT:-"$HOME/code/nano-egg-snn-es"}
CONDA=${CONDA:-"$HOME/miniforge3/bin/conda"}
ENV_NAME=${ENV_NAME:-hyperscalees}
CUDA_NVCC_DIR=${CUDA_NVCC_DIR:-"$HOME/miniforge3/envs/$ENV_NAME/lib/python3.12/site-packages/nvidia/cuda_nvcc"}

GPU_IDS_STR=${GPU_IDS:-"0 1 2 3 4"}
BATCH_SIZES_STR=${BATCH_SIZES:-"4 16 64 256 1024"}
POP_SIZES_STR=${POP_SIZES:-"2 4 16 64 256 1024 4096 16384 65536 262144 1048576"}

EPOCHS=${EPOCHS:-10}
CONFIG_TIMEOUT=${CONFIG_TIMEOUT:-3h}
UPDATE_FRACTION=${UPDATE_FRACTION:-0.01}
CIFAR_TRAIN_SAMPLES=${CIFAR_TRAIN_SAMPLES:-50000}
CIFAR_EVAL_SAMPLES=${CIFAR_EVAL_SAMPLES:-10000}
RUN_ROOT=${RUN_ROOT:-"runs/remote_nanoegg_grid/$(date +%Y%m%d_%H%M%S)"}

cd "$ROOT" || exit 2
mkdir -p "$RUN_ROOT"/{logs,status,queues}

read -r -a GPU_IDS <<< "$GPU_IDS_STR"
read -r -a BATCH_SIZES <<< "$BATCH_SIZES_STR"
read -r -a POP_SIZES <<< "$POP_SIZES_STR"

manifest="$RUN_ROOT/manifest.tsv"
{
  echo -e "batch_size\tpopulation_size\tassigned_gpu"
  idx=0
  for bs in "${BATCH_SIZES[@]}"; do
    for pop in "${POP_SIZES[@]}"; do
      gpu="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
      echo -e "${bs}\t${pop}\t${gpu}"
      echo -e "${bs}\t${pop}" >> "$RUN_ROOT/queues/gpu${gpu}.tsv"
      idx=$((idx + 1))
    done
  done
} > "$manifest"

cat > "$RUN_ROOT/run_config.env" <<EOF
ROOT=$ROOT
ENV_NAME=$ENV_NAME
CUDA_NVCC_DIR=$CUDA_NVCC_DIR
GPU_IDS=$GPU_IDS_STR
BATCH_SIZES=$BATCH_SIZES_STR
POP_SIZES=$POP_SIZES_STR
EPOCHS=$EPOCHS
CONFIG_TIMEOUT=$CONFIG_TIMEOUT
UPDATE_FRACTION=$UPDATE_FRACTION
CIFAR_TRAIN_SAMPLES=$CIFAR_TRAIN_SAMPLES
CIFAR_EVAL_SAMPLES=$CIFAR_EVAL_SAMPLES
RUN_ROOT=$RUN_ROOT
EOF

run_one() {
  local gpu="$1"
  local bs="$2"
  local pop="$3"
  local tag="b${bs}_p${pop}"
  local out_dir="$RUN_ROOT/$tag"
  local log="$RUN_ROOT/logs/${tag}.log"
  local status="$RUN_ROOT/status/${tag}.status"
  local start_ts end_ts rc outcome

  start_ts="$(date -Is)"
  {
    echo "status=running"
    echo "gpu=$gpu"
    echo "batch_size=$bs"
    echo "population_size=$pop"
    echo "start=$start_ts"
  } > "$status"

  timeout "$CONFIG_TIMEOUT" bash -lc "
    cd '$ROOT' &&
    export CUDA_HOME='$CUDA_NVCC_DIR' &&
    export XLA_FLAGS=--xla_gpu_cuda_data_dir='$CUDA_NVCC_DIR' &&
    export CUDA_VISIBLE_DEVICES='$gpu' &&
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 &&
    '$CONDA' run -n '$ENV_NAME' python experiments/spikformer_es_smoke.py \
      --preset spikformer_4_256 \
      --data-source cifar10 \
      --cifar-train-samples '$CIFAR_TRAIN_SAMPLES' \
      --cifar-eval-samples '$CIFAR_EVAL_SAMPLES' \
      --epochs '$EPOCHS' \
      --population-size '$pop' \
      --batch-size '$bs' \
      --update-fraction '$UPDATE_FRACTION' \
      --save-every 0 \
      --output-dir '$out_dir'
  " > "$log" 2>&1
  rc=$?
  end_ts="$(date -Is)"

  if [ "$rc" -eq 0 ]; then
    outcome=ok
  elif [ "$rc" -eq 124 ]; then
    outcome=timeout
  elif grep -Eiq 'out.of.memory|RESOURCE_EXHAUSTED|CUDA_ERROR_OUT_OF_MEMORY|OOM' "$log"; then
    outcome=oom
  else
    outcome=fail
  fi

  {
    echo "status=$outcome"
    echo "gpu=$gpu"
    echo "batch_size=$bs"
    echo "population_size=$pop"
    echo "start=$start_ts"
    echo "end=$end_ts"
    echo "exit_code=$rc"
    echo "log=$log"
    echo "out_dir=$out_dir"
  } > "$status"
}

worker() {
  local gpu="$1"
  local queue="$RUN_ROOT/queues/gpu${gpu}.tsv"
  local worker_log="$RUN_ROOT/logs/worker_gpu${gpu}.log"
  echo "worker gpu=$gpu start=$(date -Is)" > "$worker_log"
  while IFS=$'\t' read -r bs pop; do
    [ -n "$bs" ] || continue
    echo "worker gpu=$gpu run batch=$bs pop=$pop at $(date -Is)" >> "$worker_log"
    run_one "$gpu" "$bs" "$pop"
    echo "worker gpu=$gpu done batch=$bs pop=$pop at $(date -Is)" >> "$worker_log"
  done < "$queue"
  echo "worker gpu=$gpu end=$(date -Is)" >> "$worker_log"
}

for gpu in "${GPU_IDS[@]}"; do
  worker "$gpu" &
done

wait
echo "all_done=$(date -Is)" > "$RUN_ROOT/DONE"
