#!/usr/bin/env bash
#SBATCH -A pi_ra_lab
#SBATCH --job-name=wolf-r04-env
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/u/home/z/zzl/werewolf-round04/logs/environment-%j.log
set -euo pipefail
umask 077
module load python/3.12.13
env_root=/u/home/z/zzl/werewolf-round04/venv
if [[ ! -x "$env_root/bin/python" ]]; then
    python -m venv "$env_root"
fi
python_bin="$env_root/bin/python"
export PIP_RETRIES=10 PIP_RESUME_RETRIES=10 PIP_TIMEOUT=90
export TMPDIR="${TMPDIR:?Node-local temporary storage is required}"
"$python_bin" -m pip install --quiet --no-cache-dir 'pip>=25.2'
"$python_bin" -m pip install --quiet --no-cache-dir 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu126
"$python_bin" -m pip install --quiet --no-cache-dir 'transformers==4.51.3' 'huggingface_hub==0.36.0' 'accelerate==1.6.0' 'httpx==0.28.1'
"$python_bin" - <<'PY'
import importlib.metadata,json,os
from pathlib import Path
p=Path('/u/home/z/zzl/werewolf-round04/environment.json')
p.write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','huggingface_hub','accelerate','httpx')},'model_storage':'allocated node TMPDIR only; no global scratch writes','purpose':'Environment setup only; no model load, inference or training'},indent=2)+'\n')
PY
