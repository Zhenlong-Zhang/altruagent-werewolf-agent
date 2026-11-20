#!/usr/bin/env bash
#SBATCH -A pi_ra_lab
#SBATCH --job-name=wolf-r04-assets
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/u/home/z/zzl/werewolf-round04/logs/assets-%j.log
set -euo pipefail
umask 077
module load python/3.12.13
stage_root=/u/scratch/z/zzl/werewolf-round04
mkdir -p "$stage_root"
exec 9>"$stage_root/assets.lock"
flock -n 9 || { echo 'Another assets writer owns this stage'; exit 1; }
if [[ ! -x "$stage_root/venv/bin/python" ]]; then
    python -m venv "$stage_root/venv"
fi
python_bin="$stage_root/venv/bin/python"
export PIP_RETRIES=10 PIP_RESUME_RETRIES=10 PIP_TIMEOUT=90
"$python_bin" -m pip install --quiet --no-cache-dir 'pip>=25.2'
"$python_bin" -m pip install --quiet --no-cache-dir 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu126
"$python_bin" -m pip install --quiet --no-cache-dir 'transformers==4.51.3' 'huggingface_hub==0.36.0' 'accelerate==1.6.0' 'httpx==0.28.1'
export HF_HUB_DISABLE_XET=1 HF_HOME="$stage_root/hf"
"$python_bin" - <<'PY'
import hashlib,json
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path('/u/scratch/z/zzl/werewolf-round04')
models=[('Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218','vanilla8b'),('nics-efc/MARSHAL-Generalist-Qwen3-8B','369d9decea6e8b347f81706fa169dbbddddce360','marshal8b'),('sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41','encoder')]
records=[]
for name,revision,label in models:
 path=Path(snapshot_download(name,revision=revision,local_dir=root/'models'/label,
      allow_patterns=['*.safetensors','*.json','*.model','*.jinja','vocab.txt','merges.txt'],max_workers=4))
 config=json.loads((path/'config.json').read_text())
 records.append({'model_id':name,'revision':revision,'path':str(path),
     'architecture':config.get('architectures'),'config_sha256':hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
     'staged_files':sorted(str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and '.cache' not in p.parts)})
 print(json.dumps({'staged':name,'revision':revision}),flush=True)
output=root/'assets.json'
output.write_text(json.dumps({'models':records,'purpose':'Read-only pinned model staging; no inference or training'},indent=2)+'\n')
PY
