#!/usr/bin/env bash
# Download, benchmark, and release each model on the same GPU node.
set -euo pipefail
umask 077
module load python/3.12.13
project_dir="$(pwd)"
if [[ "${WEREWOLF_COLLECT_AFTER_BENCH:-0}" == 1 ]]; then
    (
        set -a
        source "$project_dir/.env"
        set +a
        : "${ALTRUAGENT_CONTROL_URL:?Set the platform control URL in the project .env}"
        : "${ALTRUAGENT_OFFICIAL_AGENT_KEY:?Set the official agent key in the project .env}"
    )
fi
runtime="$(mktemp -d "${TMPDIR:?Slurm node-local storage required}/wolfreasoner-models.XXXXXX")"
export HF_HOME="$runtime/hf"
export HF_HUB_DISABLE_XET=1
df -h "$runtime"
export WEREWOLF_MODEL_DEVICE="${WEREWOLF_MODEL_DEVICE:-cuda}"
torch_index=https://download.pytorch.org/whl/cu128
if [[ "$WEREWOLF_MODEL_DEVICE" == cuda ]]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
elif [[ "$WEREWOLF_MODEL_DEVICE" == cpu ]]; then
    torch_index=https://download.pytorch.org/whl/cpu
else
    echo "WEREWOLF_MODEL_DEVICE must be cpu or cuda" >&2
    exit 1
fi
python -m venv "$runtime/venv"
python_bin="$runtime/venv/bin/python"
export PIP_RETRIES=10 PIP_RESUME_RETRIES=10 PIP_TIMEOUT=90
"$python_bin" -m pip install --quiet --no-cache-dir 'pip>=25.2'
"$python_bin" -m pip install --quiet --no-cache-dir 'torch==2.7.1' \
    --index-url "$torch_index"
"$python_bin" -m pip install --quiet --no-cache-dir 'transformers==4.51.3' accelerate \
    -e "$HOME/altruagent-starter" -e "$project_dir"
if [[ "$WEREWOLF_MODEL_DEVICE" == cuda ]]; then
    "$python_bin" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
fi
models=("$@")
if (( ${#models[@]} == 0 )); then
    models=(Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3)
fi
for model in "${models[@]}"; do
    model_dir="$runtime/${model##*/}"
    "$runtime/venv/bin/hf" download "$model" --local-dir "$model_dir" \
      --revision "${WEREWOLF_MODEL_REVISION:-main}" \
      --include 'model-*.safetensors' '*.json' '*.model' '*.jinja'
    if [[ "${WEREWOLF_SKIP_BENCH:-0}" != 1 ]]; then
        for proposer in ids judge; do
            WEREWOLF_MODEL_PATH="$model_dir" WEREWOLF_PROPOSER="$proposer" \
                "$python_bin" scripts/bench_hybrid_proposer.py \
                --out "outputs/hybrid-$SLURM_JOB_ID/${model##*/}-$proposer.json"
        done
    fi
    if [[ "${WEREWOLF_COLLECT_AFTER_BENCH:-0}" == 1 ]]; then
        export WEREWOLF_MODEL_PATH="$model_dir"
        export WEREWOLF_PROPOSER=ids
        export WEREWOLF_HYBRID_MODE="${WEREWOLF_HYBRID_MODE:-hybrid_collect}"
        export WEREWOLF_LOG_DIR="$project_dir/outputs/hybrid-$SLURM_JOB_ID/decisions"
        mkdir -p "$WEREWOLF_LOG_DIR"
        (
            cd "$HOME/altruagent-starter"
            set -a
            source "$project_dir/.env"
            set +a
            exec "$python_bin" -u -m agent --match --agent werewolf_agent.hybrid_entry
        ) &
        collector_pid=$!
        trap 'kill "$collector_pid" 2>/dev/null || true' EXIT
        echo "Running $model in $WEREWOLF_HYBRID_MODE mode; logs in $WEREWOLF_LOG_DIR"
        while kill -0 "$collector_pid" 2>/dev/null; do
            sleep 30
            if [[ "$WEREWOLF_HYBRID_MODE" == hybrid_collect ]]; then
                if ! "$python_bin" scripts/train_hybrid.py "$WEREWOLF_LOG_DIR" \
                    --generator-id "hybrid-slate-v1:${model##*/}" \
                    --output "outputs/hybrid-$SLURM_JOB_ID/selector.json"; then
                    continue
                fi
            else
                "$python_bin" scripts/report_hybrid.py "$WEREWOLF_LOG_DIR" \
                    > "outputs/hybrid-$SLURM_JOB_ID/evaluation.json"
                complete_games=$("$python_bin" -c \
                    'import json,sys; print(sum(r["termination_reason"] == "completed" for r in json.load(open(sys.argv[1]))))' \
                    "outputs/hybrid-$SLURM_JOB_ID/evaluation.json")
                if (( complete_games < ${WEREWOLF_EVAL_GAMES:-1} )); then
                    continue
                fi
            fi
            kill "$collector_pid" 2>/dev/null || true
            wait "$collector_pid" || true
            trap - EXIT
            break
        done
        if [[ "$WEREWOLF_HYBRID_MODE" == hybrid_collect ]]; then
            test -s "outputs/hybrid-$SLURM_JOB_ID/selector.json"
        else
            test -s "outputs/hybrid-$SLURM_JOB_ID/evaluation.json"
            (( ${complete_games:-0} >= ${WEREWOLF_EVAL_GAMES:-1} ))
        fi
    fi
    # This directory was created by this job. Reclaim only its model copy.
    rm -r -- "$model_dir"
done
