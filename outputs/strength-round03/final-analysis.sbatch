#!/usr/bin/env bash
#SBATCH -A pi_ra_lab
#SBATCH --job-name=wolf-r03-final-audit
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --chdir=/u/home/z/zzl/werewolf-strength-r03g
#SBATCH --output=/u/home/z/zzl/werewolf-strength-r03g/logs/final-audit-%j.log
set -euo pipefail
umask 077
module load python/3.12.13
python -m scripts.summarize_strength outputs/hybrid-2123942/decisions \
  --census controller-f.json --seed r03-ids-guard-heldout-v1 \
  --arms incumbent,guarded,sampled,adapted --assignment-version wolf-policy-v1 \
  --out outputs/hybrid-2123942/final-summary.json \
  > outputs/hybrid-2123942/final-console.json
python -m scripts.analyze_strength_operations outputs/hybrid-2123942/decisions \
  --summary outputs/hybrid-2123942/final-summary.json \
  --training-artifact outputs/semantic-2123930/selector.json \
  --out outputs/hybrid-2123942/final-operations.json
