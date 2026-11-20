#!/usr/bin/env bash
#SBATCH -A pi_ra_lab
#SBATCH --job-name=wolf-r04-cu128
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/u/home/z/zzl/werewolf-round04/logs/cu128-%j.log
set -euo pipefail
umask 077
module load python/3.12.13
python_bin=/u/home/z/zzl/werewolf-round04/venv/bin/python
export PIP_RETRIES=10 PIP_RESUME_RETRIES=10 PIP_TIMEOUT=90
"$python_bin" -m pip install --quiet --no-cache-dir 'torch==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
"$python_bin" - <<'PY'
import importlib.metadata,json,os
from pathlib import Path
assert importlib.metadata.version('torch')=='2.7.1+cu128'
Path('/u/home/z/zzl/werewolf-round04/environment-cu128.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','huggingface_hub','accelerate','httpx')},'reason':'Official PyTorch2.7 CUDA12.8 build adds Blackwell kernels; enables campus97GB GPUs','references':['https://pytorch.org/blog/pytorch-2-7/','https://pytorch.org/get-started/previous-versions/'],'environment_scope':'Only the new round04 private venv; previous GPU job cancelled unstarted before upgrade'},indent=2)+'\n')
PY
