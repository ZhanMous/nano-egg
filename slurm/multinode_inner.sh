#!/bin/bash

nvidia-smi

export NCCL_P2P_DISABLE=1
export NCCL_P2P_DIRECT_DISABLE=1
export NCCL_SHM_DISABLE=1
export XLA_FLAGS="--xla_gpu_enable_analytical_sol_latency_estimator=false"
export HF_HOME="~/data/.cache/huggingface"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}

python run.py --coord_addr ${MASTER_ADDR}:${MASTER_PORT} --num_procs $SLURM_NTASKS --track --batch_size $1 --population_size $2
