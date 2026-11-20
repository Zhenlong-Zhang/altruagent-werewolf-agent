#!/usr/bin/env bash
#SBATCH -A pi_ra_lab
#SBATCH --job-name=wolf-r04-cpu-probe
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --constraint=intel-gold-6442Y|intel-xeon6-6736P
#SBATCH --time=00:45:00
#SBATCH --chdir=/u/home/z/zzl/werewolf-round04/cpu-probe
#SBATCH --output=/u/home/z/zzl/werewolf-round04/logs/cpu-probe-%j.log
set -euo pipefail
umask 077
module load python/3.12.13
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
python_bin=/u/home/z/zzl/werewolf-round04/venv/bin/python
runtime="$(mktemp -d "${TMPDIR:?}/wolf-r04-cpu.XXXXXX")"
export HF_HOME="$runtime/hf" HF_HUB_DISABLE_XET=1
sdk_deps="$runtime/sdk-deps"
"$python_bin" -m pip install --quiet --no-cache-dir --target "$sdk_deps" 'mcp==1.28.1' 'python-dotenv==1.1.1' 'httpx==0.28.1'
export PYTHONPATH="$PWD:/u/home/z/zzl/altruagent-starter:$sdk_deps${PYTHONPATH:+:$PYTHONPATH}"
out="outputs/probe-$SLURM_JOB_ID"
mkdir -p "$out"
lscpu > "$out/cpu.txt"
"$python_bin" -m scripts.check_contextual_primitives > "$out/primitives.log" 2>&1
for model in vanilla marshal; do
  if [[ "$model" == vanilla ]]; then
    model_id=Qwen/Qwen3-8B
    revision=b968826d9c46dd6066d109eabc6255188de91218
  else
    model_id=nics-efc/MARSHAL-Generalist-Qwen3-8B
    revision=369d9decea6e8b347f81706fa169dbbddddce360
  fi
  model_path="$runtime/$model"
  "$python_bin" - "$model_id" "$revision" "$model_path" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],allow_patterns=['*.safetensors','*.json','*.model','*.jinja','vocab.txt','merges.txt'],max_workers=4)
PY
  "$python_bin" -m scripts.probe_strategy_cpu --model-path "$model_path" --model-id "$model_id" --revision "$revision" --out "$out/$model.jsonl"
done
