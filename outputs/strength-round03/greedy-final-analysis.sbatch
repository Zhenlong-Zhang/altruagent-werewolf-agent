#!/usr/bin/env bash
#SBATCH -A pi_ra_lab
#SBATCH --job-name=wolf-r03-greedy-audit
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --chdir=/u/home/z/zzl/werewolf-strength-r03h
#SBATCH --output=/u/home/z/zzl/werewolf-strength-r03h/logs/final-audit-%j.log
set -euo pipefail
umask 077
module load python/3.12.13
python -m scripts.summarize_strength outputs/hybrid-2133983/decisions \
  --census controller-g.json --seed r03-semantic-greedy-heldout-v1 \
  --arms incumbent,guarded,nn_greedy --assignment-version wolf-policy-v1 \
  --out outputs/hybrid-2133983/final-summary.json \
  > outputs/hybrid-2133983/final-console.json
completion_status=0
python -m scripts.analyze_strength_operations outputs/hybrid-2133983/decisions \
  --summary outputs/hybrid-2133983/final-summary.json \
  --training-artifact /u/home/z/zzl/werewolf-strength-r03g/outputs/semantic-2123930/selector.json \
  --profile-set greedy3 --out outputs/hybrid-2133983/final-operations.json \
  || completion_status=$?
python - <<'PY'
import hashlib,json
from pathlib import Path
base=Path('/u/home/z/zzl/werewolf-strength-r03g')
fpath=base/'outputs/hybrid-2123942/final-summary.json'
gpath=Path('outputs/hybrid-2133983/final-summary.json')
tpath=base/'outputs/semantic-2123930/selector.json'
f=json.loads(fpath.read_text());g=json.loads(gpath.read_text());t=json.loads(tpath.read_text())
fs={row['match_hash'] for row in f['games']};gs={row['match_hash'] for row in g['games']};ts=set(t['train_match_hashes'])
checks={'E_F_disjoint':not(ts&fs),'E_G_disjoint':not(ts&gs),'F_G_disjoint':not(fs&gs),
        'G_seed':g['seed']=='r03-semantic-greedy-heldout-v1',
        'F_seed':f['seed']=='r03-ids-guard-heldout-v1'}
r={'checks':checks,'passed':all(checks.values()),'counts':{'E_training':len(ts),'F_assigned':len(fs),'G_assigned':len(gs)},
   'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (fpath,gpath,tpath)}}
out=Path('outputs/hybrid-2133983/split-audit.json')
with out.open('x') as stream:json.dump(r,stream,indent=2)
print(json.dumps(r),flush=True)
assert r['passed'],'Dataset isolation failed'
PY
python -m scripts.compare_greedy_actions outputs/hybrid-2133983/decisions \
  --summary outputs/hybrid-2133983/final-summary.json \
  --training-artifact /u/home/z/zzl/werewolf-strength-r03g/outputs/semantic-2123930/selector.json \
  --old /u/home/z/zzl/altruagent-werewolf-agent/outputs/hybrid/qwen3-8b-round1/selector.json \
  --out outputs/hybrid-2133983/greedy-action-differences.json
exit "$completion_status"
