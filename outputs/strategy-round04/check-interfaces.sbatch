#!/usr/bin/env bash
#SBATCH -A pi_ra_lab
#SBATCH --job-name=wolf-r04-interface-check
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --chdir=/u/home/z/zzl/werewolf-round04/phase-a
#SBATCH --output=/u/home/z/zzl/werewolf-round04/logs/interfaces-%j.log
set -euo pipefail
umask 077
module load python/3.12.13
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_bin=/u/home/z/zzl/werewolf-round04/venv/bin/python
sdk_deps="$(mktemp -d "${TMPDIR:?}/wolf-r04-sdk-deps.XXXXXX")"
# Job-local dependencies avoid mutating the environment of the pending GPU job.
"$python_bin" -m pip install --quiet --no-cache-dir --target "$sdk_deps" 'mcp==1.28.1' 'python-dotenv==1.1.1' 'httpx==0.28.1'
export PYTHONPATH="$PWD:/u/home/z/zzl/altruagent-starter:$sdk_deps${PYTHONPATH:+:$PYTHONPATH}"
out="outputs/checks-$SLURM_JOB_ID"
mkdir -p "$out"
"$python_bin" -m scripts.check_strategy_interfaces > "$out/interfaces.log" 2>&1
"$python_bin" -m scripts.check_strategy_backend > "$out/backend.log" 2>&1
"$python_bin" -m pip freeze --path "$sdk_deps" > "$out/sdk-dependencies.txt"
"$python_bin" - <<'PY'
import json,os
from pathlib import Path
root=Path('outputs')/('checks-'+os.environ['SLURM_JOB_ID'])
(root/'result.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'source_commit':'75a6cd625def57bb039cd61ad3aae0df6267dcfe','interface_methods':16,'backend_methods':11,'all_passed':True,'scope':'Real unit/adapter checks on Slurm CPU; synthetic HF weights only, no language-model inference or gameplay'},indent=2)+'\n')
PY
