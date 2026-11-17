#!/usr/bin/env bash
# Download, benchmark, and release each model on the same GPU node.
set -euo pipefail
umask 077
module load python/3.12.13
project_dir="$(pwd)"
runtime="$(mktemp -d "${TMPDIR:?Slurm node-local storage required}/wolfreasoner-models.XXXXXX")"
export HF_HOME="$runtime/hf"
export HF_HUB_DISABLE_XET=1
df -h "$runtime"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -m venv "$runtime/venv"
python_bin="$runtime/venv/bin/python"
"$python_bin" -m pip install --quiet --no-cache-dir 'torch==2.7.1' \
    --index-url https://download.pytorch.org/whl/cu126
"$python_bin" -m pip install --quiet --no-cache-dir 'transformers==4.51.3' accelerate \
    -e "$HOME/altruagent-starter" -e "$project_dir"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
export WEREWOLF_MODEL_DEVICE=cuda
for model in Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3; do
    model_dir="$runtime/${model##*/}"
    "$runtime/venv/bin/hf" download "$model" --local-dir "$model_dir" \
      --include 'model-*.safetensors' '*.json' '*.model' '*.jinja'
    for proposer in ids judge; do
        WEREWOLF_MODEL_PATH="$model_dir" WEREWOLF_PROPOSER="$proposer" \
            "$python_bin" scripts/bench_hybrid_proposer.py \
            --out "outputs/hybrid-$SLURM_JOB_ID/${model##*/}-$proposer.json"
    done
    # This directory was created by this job. Reclaim only its model copy.
    rm -r -- "$model_dir"
done
