# Hoffman2 workspace

This project uses the production Slurm cluster (`hoffman2.oarc.ucla.edu`). The project checkout is separate from existing class/research folders and jobs.

## Current strength round (19 November 2026)

The frozen E collection is job2116767 in `/u/home/z/zzl/werewolf-strength-r03e`:30 completed eligible games,136 sampled choices and142 acknowledged decisions. The isolated semantic-PPO retry is job2123930 in `/u/home/z/zzl/werewolf-strength-r03g`; its32 accepted updates and verified checkpoint are archived at [job-2123930](outputs/strength-round03/job-2123930). Both Qwen3-8B and the MiniLM encoder remain frozen. The actor/critic is trained; this does not fine-tune the language model.

The separate evaluation job2123942 uses8 CPUs,64GB, one live game at a time and4 CPU threads per HF process. Its exact submission is [evaluate-semantic-v2.sbatch](outputs/strength-round03/evaluate-semantic-v2.sbatch). No training occurs in that job. Source hashes, prior failed attempts and the fixed comparison contract are preserved under [round03](rounds/03_strength/15_semantic_execution.md). These instructions supersede the earlier prototype configurations below.

Tournament serving uses `werewolf_agent.competition_entry`, never a practice comparison table. Stage the pinned model/tokenizer, explicit selector checkpoint and, for schema2, the pinned MiniLM encoder with `WEREWOLF_SEMANTIC_ENCODER`. Run the official starter inside an already active allocation covering the competition window. Only one runtime may use an Official Agent Key. Keep the private `.env` outside Git and freeze the chosen guard, mode, precision and thread settings with the release. The final serving profile must follow the heldout comparison; a successful training job alone is insufficient. Actual tournament-context connection, schedule and simultaneous-assignment capacity still require verification for the competition window.

## Current cluster snapshot

- Account: `pi_ra_lab`, routed to campus; no highp access.
- Python module available for this project: `python/3.12.13` (the default `python3` is 3.9 and is too old for the starter).
- At initial setup, the only existing job was `1780018` (`260e-part2`, pending priority) in `/u/home/z/zzl/260e-part2`. This project does not edit or cancel it.

## Environment

Load the current Python module, create a project-local virtual environment, install the starter and this policy package. Keep credentials in a private ignored `.env` inside the starter checkout only; never in this repo or job scripts.

```bash
module load python/3.12.13
python --version
python -m venv ~/altruagent-starter/.venv
source ~/altruagent-starter/.venv/bin/activate
cd ~/altruagent-starter
pip install -e ".[dev]"
pip install -e "$HOME/altruagent-werewolf-agent"
```

Submit offline training and platform practice through Slurm using the current
account and an explicit CPU, memory and walltime request. The first offline
training run used 1 CPU and completed in 8 seconds. Reproduce it from the
repository checkout:

```bash
sbatch -A pi_ra_lab -J wolf-offline -c 1 --mem=512M -t 00:08:00 \
  --wrap='module load python/3.12.13 && python scripts/offline_train.py --out-dir "outputs/h2-$SLURM_JOB_ID"'
```

The LLM practice runtime needs the CPU PyTorch and Transformers environment
and a downloaded Qwen model. Set `WEREWOLF_CPU_MODEL` to the model directory
and `OPENAI_MODEL` to its recorded name. The rules, search and REINFORCE
variants use the starter environment alone. Source the project's private
`.env` with shell export enabled inside the Slurm job. The active runtime
is `python -m agent --match --agent werewolf_agent.entry`.

For a fixed official comparison, set `WEREWOLF_CANDIDATE=manifest` and
`WEREWOLF_MATCH_MANIFEST` to JSON mapping each match ID to
`[zero_based_seat, candidate]`. Run
`python scripts/summarize_comparison.py <manifest> <decision_log_dir>`
after platform settlement. Replays and model evaluation limits are in the
[training report](rounds/01_training/05_closeout.md).

## Joint LLM and RL

Apply both starter patches before collecting:
`patches/starter-game-result.patch` records settlement and
`patches/starter-message-result.patch` acknowledges speech or pass delivery.
For this experiment, launch the starter with
`--match --agent werewolf_agent.hybrid_entry` inside the Slurm job.

- `WEREWOLF_HYBRID_MODE=hybrid_collect` samples the agenda before play.
  Freeze the local model and starting checkpoint across the collection.
- `WEREWOLF_HYBRID_MODE=llm_select` selects the first model proposal.
- `WEREWOLF_HYBRID_MODE=hybrid_greedy` selects using the weights specified in
  `WEREWOLF_HYBRID_WEIGHTS`. Use fresh Testing games for policy comparison.
- Optional `WEREWOLF_PUBLIC_BELIEF=1` adds role probabilities with weaker
  updates from less credible accusers. It changes the proposer ID and
  requires a separate frozen collection batch.

Use a dedicated `WEREWOLF_LOG_DIR` for training games. In a CPU batch job,
run `python scripts/train_hybrid.py LOG_DIR --generator-id GENERATOR_ID
--output NEW_WEIGHTS`. Use `--start OLD_WEIGHTS` when collecting from a
previous checkpoint. The generator ID is in each `hybrid_choice` log row.
See the [joint policy contract](rounds/02_hybrid/02_contract.md).
