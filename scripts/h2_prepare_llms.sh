#!/usr/bin/env bash
# Stage models and run diagnostics, collection or validation in one allocation.
set -euo pipefail
umask 077
module load python/3.12.13
project_dir="$(pwd)"
env_file="${WEREWOLF_ENV_FILE:-$project_dir/.env}"
if [[ "${WEREWOLF_COLLECT_AFTER_BENCH:-0}" == 1 ]]; then
    (
        set -a
        source "$env_file"
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
if [[ "${WEREWOLF_ROUND3_CHECKS:-0}" == 1 ]]; then
    "$python_bin" -m scripts.check_strength_regressions
    "$python_bin" -m scripts.check_strength_comparison
    "$python_bin" -m scripts.check_strength_accounting
fi
if [[ "$WEREWOLF_MODEL_DEVICE" == cuda ]]; then
    "$python_bin" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
fi
if [[ "${WEREWOLF_SEMANTIC_STAGE:-0}" == 1 ]]; then
    export WEREWOLF_SEMANTIC_ENCODER="$runtime/all-MiniLM-L6-v2"
    "$runtime/venv/bin/hf" download sentence-transformers/all-MiniLM-L6-v2 \
      --local-dir "$WEREWOLF_SEMANTIC_ENCODER" \
      --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
      --include 'model.safetensors' '*.json' 'vocab.txt'
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
    if [[ "${WEREWOLF_BALANCE_BENCH:-0}" == 1 ]]; then
        for balance in none cyclic; do
            WEREWOLF_MODEL_PATH="$model_dir" WEREWOLF_PROPOSER=grounded \
                WEREWOLF_GROUNDED_ORDER_BALANCE="$balance" \
                "$python_bin" -m scripts.bench_strength \
                --scenarios wolf_remove_seer,verified_check_over_claim,investigate_unknown_set \
                --out "outputs/hybrid-$SLURM_JOB_ID/balance-$balance-strength.json"
            WEREWOLF_MODEL_PATH="$model_dir" WEREWOLF_PROPOSER=grounded \
                WEREWOLF_GROUNDED_ORDER_BALANCE="$balance" \
                "$python_bin" -m scripts.bench_strength --suite mapped \
                --scenarios verified_check_over_claim,investigate_unknown_set \
                --out "outputs/hybrid-$SLURM_JOB_ID/balance-$balance-mapped.json"
        done
    elif [[ "${WEREWOLF_STRENGTH_BENCH:-0}" == 1 ]]; then
        if [[ "${WEREWOLF_STRENGTH_F32_CONTROL:-0}" == 1 ]]; then
            WEREWOLF_MODEL_PATH="$model_dir" WEREWOLF_PROPOSER=ids WEREWOLF_MODEL_DTYPE=float32 \
                "$python_bin" -m scripts.bench_strength --suite historical \
                --out "outputs/hybrid-$SLURM_JOB_ID/ids-float32-control.json"
        fi
        read -r -a benchmark_proposers <<< "${WEREWOLF_BENCH_PROPOSERS:-ids reasoned}"
        for proposer in "${benchmark_proposers[@]}"; do
            WEREWOLF_MODEL_PATH="$model_dir" WEREWOLF_PROPOSER="$proposer" \
                "$python_bin" -m scripts.bench_strength \
                --out "outputs/hybrid-$SLURM_JOB_ID/$proposer-strength.json"
            if [[ "${WEREWOLF_BENCH_MAPPED:-0}" == 1 ]]; then
                WEREWOLF_MODEL_PATH="$model_dir" WEREWOLF_PROPOSER="$proposer" \
                    "$python_bin" -m scripts.bench_strength --suite mapped \
                    --out "outputs/hybrid-$SLURM_JOB_ID/$proposer-mapped.json"
            fi
        done
    elif [[ "${WEREWOLF_SKIP_BENCH:-0}" != 1 ]]; then
        for proposer in ids judge; do
            WEREWOLF_MODEL_PATH="$model_dir" WEREWOLF_PROPOSER="$proposer" \
                "$python_bin" scripts/bench_hybrid_proposer.py \
                --out "outputs/hybrid-$SLURM_JOB_ID/${model##*/}-$proposer.json"
        done
    fi
    if [[ "${WEREWOLF_COLLECT_AFTER_BENCH:-0}" == 1 ]]; then
        export WEREWOLF_MODEL_PATH="$model_dir"
        if [[ "${WEREWOLF_SEMANTIC_STAGE:-0}" == 1 ]]; then
            : "${WEREWOLF_SEMANTIC_CHECKPOINT:?Set the frozen semantic policy artifact}"
            "$python_bin" -m scripts.bench_semantic_runtime \
                --checkpoint "$WEREWOLF_SEMANTIC_CHECKPOINT" \
                --out "outputs/hybrid-$SLURM_JOB_ID/semantic-runtime.json"
        fi
        export WEREWOLF_PROPOSER="${WEREWOLF_PROPOSER:-ids}"
        export WEREWOLF_HYBRID_MODE="${WEREWOLF_HYBRID_MODE:-hybrid_collect}"
        export WEREWOLF_LOG_DIR="$project_dir/outputs/hybrid-$SLURM_JOB_ID/decisions"
        mkdir -p "$WEREWOLF_LOG_DIR"
        (
            cd "$HOME/altruagent-starter"
            set -a
            source "$env_file"
            set +a
            exec "$python_bin" -u -m agent --match \
                --agent "${WEREWOLF_AGENT_ENTRY:-werewolf_agent.hybrid_entry}"
        ) &
        collector_pid=$!
        trap 'kill "$collector_pid" 2>/dev/null || true' EXIT
        echo "Running $model in $WEREWOLF_HYBRID_MODE mode; logs in $WEREWOLF_LOG_DIR"
        while kill -0 "$collector_pid" 2>/dev/null; do
            sleep 30
            if [[ "$WEREWOLF_HYBRID_MODE" == hybrid_collect && "${WEREWOLF_EVALUATE_ONLY:-0}" != 1 ]]; then
                start_args=()
                if [[ -n "${WEREWOLF_HYBRID_WEIGHTS:-}" ]]; then
                    start_args=(--start "$WEREWOLF_HYBRID_WEIGHTS")
                fi
                generator_id=$("$python_bin" scripts/report_hybrid.py "$WEREWOLF_LOG_DIR" --generator-id)
                if ! "$python_bin" scripts/train_hybrid.py "$WEREWOLF_LOG_DIR" \
                    --generator-id "$generator_id" "${start_args[@]}" \
                    --min-games "${WEREWOLF_TRAIN_MIN_GAMES:-30}" \
                    --step "${WEREWOLF_TRAIN_STEP:-0.05}" \
                    --output "outputs/hybrid-$SLURM_JOB_ID/selector.json"; then
                    continue
                fi
            else
                if ! "$python_bin" scripts/report_hybrid.py "$WEREWOLF_LOG_DIR" \
                    > "outputs/hybrid-$SLURM_JOB_ID/evaluation.json"; then
                    continue
                fi
                complete_games=$("$python_bin" -c \
                    'import json,sys; print(sum(r["termination_reason"] == "completed" for r in json.load(open(sys.argv[1]))))' \
                    "outputs/hybrid-$SLURM_JOB_ID/evaluation.json")
                if (( complete_games < ${WEREWOLF_EVAL_GAMES:-1} )); then
                    continue
                fi
                evaluation_arms="${WEREWOLF_EVAL_ARMS:-${WEREWOLF_HYBRID_COMPARE_PROPOSERS:-}}"
                if [[ -n "$evaluation_arms" ]]; then
                    if ! "$python_bin" scripts/report_hybrid.py "$WEREWOLF_LOG_DIR" \
                        --check-arms "$evaluation_arms" \
                        --minimum-per-arm "${WEREWOLF_COMPARE_GAMES_PER_ARM:-12}"; then
                        continue
                    fi
                fi
            fi
            kill "$collector_pid" 2>/dev/null || true
            wait "$collector_pid" || true
            trap - EXIT
            break
        done
        if [[ "$WEREWOLF_HYBRID_MODE" == hybrid_collect && "${WEREWOLF_EVALUATE_ONLY:-0}" != 1 ]]; then
            test -s "outputs/hybrid-$SLURM_JOB_ID/selector.json"
        else
            test -s "outputs/hybrid-$SLURM_JOB_ID/evaluation.json"
            (( ${complete_games:-0} >= ${WEREWOLF_EVAL_GAMES:-1} ))
            evaluation_arms="${WEREWOLF_EVAL_ARMS:-${WEREWOLF_HYBRID_COMPARE_PROPOSERS:-}}"
            if [[ -n "$evaluation_arms" ]]; then
                "$python_bin" scripts/report_hybrid.py "$WEREWOLF_LOG_DIR" \
                    --check-arms "$evaluation_arms" \
                    --minimum-per-arm "${WEREWOLF_COMPARE_GAMES_PER_ARM:-12}"
            fi
        fi
    fi
    # This directory was created by this job. Reclaim only its model copy.
    rm -r -- "$model_dir"
done
