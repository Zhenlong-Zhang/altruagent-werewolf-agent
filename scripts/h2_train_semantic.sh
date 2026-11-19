#!/usr/bin/env bash
# Run only in a Slurm compute allocation, after the frozen collection completes.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${WEREWOLF_TRAIN_LOG_DIR:?Set frozen Phase E decision directory}"
: "${WEREWOLF_BEHAVIOR_CHECKPOINT:?Set the frozen behavior selector}"
: "${WEREWOLF_LINEAR_CHILD:?Set the separately audited REINFORCE reference}"
module load python/3.12.13
project_dir="$(pwd)"
runtime="$(mktemp -d "${TMPDIR:?Node-local storage required}/wolf-semantic.XXXXXX")"
export HF_HOME="$runtime/hf" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS" WEREWOLF_CPU_THREADS="$OMP_NUM_THREADS"
python -m venv "$runtime/venv"
python_bin="$runtime/venv/bin/python"
export PIP_RETRIES=10 PIP_RESUME_RETRIES=10 PIP_TIMEOUT=90
"$python_bin" -m pip install --quiet --no-cache-dir 'pip>=25.2'
"$python_bin" -m pip install --quiet --no-cache-dir 'torch==2.7.1' \
  --index-url https://download.pytorch.org/whl/cpu
"$python_bin" -m pip install --quiet --no-cache-dir 'transformers==4.51.3' accelerate \
  -e "$HOME/altruagent-starter" -e "$project_dir"
export WEREWOLF_SEMANTIC_ENCODER="$runtime/all-MiniLM-L6-v2"
"$runtime/venv/bin/hf" download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir "$WEREWOLF_SEMANTIC_ENCODER" \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --include 'model.safetensors' '*.json' 'vocab.txt'
out="outputs/semantic-$SLURM_JOB_ID"
mkdir -p "$out"
"$python_bin" -m scripts.check_strength_regressions
"$python_bin" -m scripts.check_strength_comparison
"$python_bin" -m scripts.check_strength_accounting
"$python_bin" -m scripts.analyze_selector_update "$WEREWOLF_TRAIN_LOG_DIR" \
  --start "$WEREWOLF_BEHAVIOR_CHECKPOINT" --trained "$WEREWOLF_LINEAR_CHILD" \
  --out "$out/linear-reference-audit.json"
"$python_bin" -m scripts.check_semantic_selector \
  --encoder-path "$WEREWOLF_SEMANTIC_ENCODER" --out "$out/pretrain-checks.json"
generator_id=$("$python_bin" -m scripts.report_hybrid "$WEREWOLF_TRAIN_LOG_DIR" --generator-id)
"$python_bin" -m scripts.train_semantic_ppo "$WEREWOLF_TRAIN_LOG_DIR" \
  --start "$WEREWOLF_BEHAVIOR_CHECKPOINT" --encoder-path "$WEREWOLF_SEMANTIC_ENCODER" \
  --generator-id "$generator_id" --min-games 30 --epochs 32 --seed 7391 \
  --output "$out/selector.json" --report "$out/training-report.json"
"$python_bin" -m scripts.check_semantic_selector \
  --encoder-path "$WEREWOLF_SEMANTIC_ENCODER" --artifact "$out/selector.json" \
  --out "$out/posttrain-checks.json"
test -s "$out/selector.json"
test -s "$out/selector.pt"
