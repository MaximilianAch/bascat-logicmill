#!/bin/bash -l
#SBATCH -J oss-1
#SBATCH -D ./
#SBATCH -o ./logs/job.out.%j
#SBATCH -e ./logs/job.err.%j
#SBATCH --time=0-03:59:00
#
# Request all CPUs and memory on node:
#SBATCH --nodes=1
#SBATCH --partition=gpu1 # for partial node
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem=250000


nvidia-smi
module purge
module load cuda/12.8

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$HOME/Research/llms/local_llms/data/OSS-120G"
PORT=9731

source ~/.bashrc
eval "$(pyenv init -)"
pyenv shell vllm

HOSTNAME=$(hostname)
CONFIG_FILE="$SCRIPT_DIR/../vllm_server_info_1gpu.json"

trap "rm -f '$CONFIG_FILE'; echo 'Cleaned up $CONFIG_FILE'" EXIT

echo "========================================="
echo "Starting MAX PERFORMANCE vllm server on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Python environment: $(pyenv version)"
echo "Python path: $(which python)"
echo "vllm version: $(python -c 'import vllm; print(vllm.__version__)')"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel Size: -"
echo "Quantization: Model native (mxfp4)"
echo "Max Batched Tokens: 16384"
echo "Max Num Seqs: 256"
echo "Chunked Prefill: ENABLED"
echo "Enforce Eager: DISABLED (CUDA graphs enabled)"
echo "Writing server info to: $CONFIG_FILE"
echo "========================================="

cat > "$CONFIG_FILE" << EOF
{
  "hostname": "$HOSTNAME",
  "port": $PORT,
  "model": "$MODEL_PATH",
  "job_id": "$SLURM_JOB_ID",
  "started_at": "$(date -Iseconds)"
}
EOF

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --disable-log-stats \
    --port $PORT \
    --host 0.0.0.0 \
    --trust-remote-code

    # --host 127.0.0.1 # bind to localhost only
    # --dtype auto # auto-select 