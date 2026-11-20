#!/usr/bin/env bash
#SBATCH -A pi_ra_lab
#SBATCH --job-name=wolf-r04-bc-check
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/u/home/z/zzl/werewolf-round04/phase-bc-attempt1
#SBATCH --output=/u/home/z/zzl/werewolf-round04/logs/bc-check-%j.log
set -euo pipefail
umask 077
module load python/3.12.13
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_bin=/u/home/z/zzl/werewolf-round04/venv/bin/python
sdk_deps="$(mktemp -d "${TMPDIR:?}/wolf-r04-sdk-deps.XXXXXX")"
"$python_bin" -m pip install --quiet --no-cache-dir --target "$sdk_deps" 'mcp==1.28.1' 'python-dotenv==1.1.1' 'httpx==0.28.1'
export PYTHONPATH="$PWD:$PWD/starter:$sdk_deps${PYTHONPATH:+:$PYTHONPATH}"
out="outputs/checks-$SLURM_JOB_ID"
mkdir -p "$out"
# Preserve every failing suite; a failure must not hide independent checks.
failed=0
for suite in check_strategy_controller check_strategy_transport check_contextual_primitives check_contextual_selector check_contextual_ppo; do
  if "$python_bin" -m "scripts.$suite" > "$out/$suite.log" 2>&1; then
    echo "$suite PASS"
  else
    echo "$suite FAIL"
    failed=1
  fi
done
"$python_bin" -m scripts.summarize_strategy_probe --input /u/home/z/zzl/werewolf-round04/cpu-probe/outputs/probe-2186591 --out "$out/cpu-probe-summary.json"
exit "$failed"
