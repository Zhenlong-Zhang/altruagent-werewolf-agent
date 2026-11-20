#!/usr/bin/env bash
#SBATCH -A pi_ra_lab
#SBATCH --job-name=wolf-r04-model-screen
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vmem40G|vmem48G|vmem80G|vmem94G|vmem141G
#SBATCH --time=02:00:00
#SBATCH --chdir=/u/home/z/zzl/werewolf-round04/phase-a
#SBATCH --output=/u/home/z/zzl/werewolf-round04/logs/model-screen-%j.log
set -euo pipefail
umask 077
module load python/3.12.13
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python_bin=/u/home/z/zzl/werewolf-round04/venv/bin/python
runtime="$(mktemp -d "${TMPDIR:?Compute-node temporary disk is required}/wolf-r04-screen.XXXXXX")"
export HF_HOME="$runtime/hf" HF_HUB_DISABLE_XET=1
export PYTHONPATH="$PWD:/u/home/z/zzl/altruagent-starter${PYTHONPATH:+:$PYTHONPATH}"
out="outputs/screen-$SLURM_JOB_ID"
mkdir -p "$out"
"$python_bin" -m pip install --quiet --no-cache-dir -e /u/home/z/zzl/altruagent-starter
"$python_bin" -m pip freeze > "$out/environment.txt"
"$python_bin" - <<'PY'
import json,os,torch
assert torch.cuda.is_available(), 'Allocated CUDA device unavailable'
assert torch.cuda.is_bf16_supported(), 'Both checkpoints require matched BF16'
assert torch.cuda.device_count()==1, 'Expected one visible allocated GPU'
print(json.dumps({'job':os.environ['SLURM_JOB_ID'],'device':torch.cuda.get_device_name(0),
                 'torch':str(torch.__version__),'cuda':torch.version.cuda,
                 'node_local_runtime':os.environ['HF_HOME']}),flush=True)
PY
"$python_bin" -m scripts.check_strategy_interfaces > "$out/interfaces.log" 2>&1
"$python_bin" -m scripts.check_strategy_backend > "$out/backend.log" 2>&1
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
snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],
                  allow_patterns=['*.safetensors','*.json','*.model','*.jinja','vocab.txt','merges.txt'],max_workers=4)
PY
    "$python_bin" -m scripts.bench_strategy_models \
        --model-path "$model_path" --model-id "$model_id" --revision "$revision" \
        --max-new-tokens 2048 --seconds 30 --temperature .6 --out "$out/$model.json"
done
# TMPDIR is owned/reclaimed by this allocation; no global scratch files removed.
