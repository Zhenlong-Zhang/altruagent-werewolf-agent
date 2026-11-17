#!/usr/bin/env bash
set -euo pipefail
umask 077
module load python/3.12.13
project_dir="$(pwd)"
runtime="${SCRATCH:?Use a Hoffman2 compute job}/wolfreasoner-models"
mkdir -p "$runtime"
python -m venv "$runtime/venv"
python_bin="$runtime/venv/bin/python"
"$python_bin" -m pip install --quiet 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu126
"$python_bin" -m pip install --quiet 'transformers==4.51.3' accelerate \
  -e "$HOME/altruagent-starter" -e "$project_dir"
export HF_HUB_DISABLE_XET=1
for model in Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3; do
    "$runtime/venv/bin/hf" download "$model" --local-dir "$runtime/${model##*/}" \
      --include 'model-*.safetensors' '*.json' '*.model' '*.jinja'
done
