#!/usr/bin/env bash
#SBATCH -A pi_ra_lab
#SBATCH --job-name=wolf-r03h-greedy
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=intel-xeon6-6736P
#SBATCH --chdir=/u/home/z/zzl/werewolf-strength-r03h
#SBATCH --output=/u/home/z/zzl/werewolf-strength-r03h/logs/evaluation-%j.log
# Controller enforces MAX75 created games and stops new games with <5 minutes remaining.
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export WEREWOLF_CPU_THREADS=4
export WEREWOLF_ENV_FILE=/u/home/z/zzl/altruagent-werewolf-agent/.env
export WEREWOLF_MODEL_DEVICE=cpu WEREWOLF_MODEL_DTYPE=bfloat16
export WEREWOLF_MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
export WEREWOLF_SKIP_BENCH=1 WEREWOLF_COLLECT_AFTER_BENCH=1 WEREWOLF_ROUND3_CHECKS=1
export WEREWOLF_PROPOSER=ids WEREWOLF_EVIDENCE_GUARD=none
export WEREWOLF_GROUNDED_ORDER_BALANCE=none
export WEREWOLF_AGENT_ENTRY=werewolf_agent.policy_compare_entry
export WEREWOLF_HYBRID_MODE=hybrid_greedy WEREWOLF_EVALUATE_ONLY=1
export WEREWOLF_HYBRID_OPPONENTS=direct,rules,rules,rules,rules,rules
export WEREWOLF_EVAL_ARMS=incumbent,guarded,nn_greedy
export WEREWOLF_EVAL_GAMES=36 WEREWOLF_COMPARE_GAMES_PER_ARM=12
export WEREWOLF_POLICY_COMPARE_SEED=r03-semantic-greedy-heldout-v1
export WEREWOLF_SEMANTIC_STAGE=1
export WEREWOLF_SEMANTIC_BENCH_MODE=hybrid_greedy
export WEREWOLF_SEMANTIC_CHECKPOINT=/u/home/z/zzl/werewolf-strength-r03g/outputs/semantic-2123930/selector.json
export WEREWOLF_POLICY_PROFILES='[{"name":"incumbent","mode":"hybrid_greedy","guard":"none","weights":"/u/home/z/zzl/altruagent-werewolf-agent/outputs/hybrid/qwen3-8b-round1/selector.json"},{"name":"guarded","mode":"hybrid_greedy","guard":"verified-innocent-v1","weights":"/u/home/z/zzl/altruagent-werewolf-agent/outputs/hybrid/qwen3-8b-round1/selector.json"},{"name":"nn_greedy","mode":"hybrid_greedy","guard":"verified-innocent-v1","weights":"/u/home/z/zzl/werewolf-strength-r03g/outputs/semantic-2123930/selector.json"}]'
python3 - <<'PY'
import json
import hashlib
from pathlib import Path
artifact = Path('/u/home/z/zzl/werewolf-strength-r03g/outputs/semantic-2123930/selector.json')
raw = artifact.read_bytes()
metadata = json.loads(raw)
tensor_hash = hashlib.sha256((artifact.parent / metadata['weights_file']).read_bytes()).hexdigest()
assert tensor_hash == metadata['weights_sha256']
assert hashlib.sha256(raw + tensor_hash.encode()).hexdigest()[:16] == '6ca4df8fd98821e6'
parent_path = Path('/u/home/z/zzl/altruagent-werewolf-agent/outputs/hybrid/qwen3-8b-round1/selector.json')
assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == metadata['parent_sha256']
report = json.loads((artifact.parent / 'training-report.json').read_text())
assert metadata['train_games'] == 30 and metadata['parent'] == 'd7ebf7670444067a'
assert metadata['accepted_epochs'] > 0, 'No accepted PPO update: stop before heldout matches'
assert report['artifact'] == metadata
assert report['initial_logits_exact_legacy'] and report['fixed_advantages_unchanged'] and report['encoder_frozen']
assert 0 <= report['final_mean_kl'] <= .02 and report['final_max_kl'] <= .1
print({'semantic_candidate_preflight': 'passed', 'accepted_epochs': metadata['accepted_epochs']}, flush=True)
PY
bash scripts/h2_prepare_llms.sh Qwen/Qwen3-8B
