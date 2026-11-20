# AltruAgent Werewolf agent

A role-aware, inspectable Werewolf policy and match-analysis toolkit for the AltruAgent tournament. The aim is measured playing strength; this repository does not assume that RL is automatically the best method.

## Current strength experiment (19 November 2026)

The frozen Qwen3-8B proposer now supports a frozen MiniLM semantic encoder and a trained attention/PPO actor–critic (30,421 actor parameters, 1,345 critic parameters). The language-model weights remain frozen. Training used30 completed games and136 sampled decisions; the trained artifact is [checkpoint6ca4df8fd98821e6](outputs/strength-round03/job-2123930/selector.json).

The fresh sampled-policy comparison completed53 games: semantic sampled won5/13 versus6/12 for the old sampled control; incumbent and guarded greedy controls won7/14 and8/14. A separately predeclared greedy comparison completed54 games: incumbent16/21, guarded7/12 and semantic11/21. All490 focal decisions across these two pilots were acknowledged with no focal fallback or delivery error. Small samples and unequal roles leave wide uncertainty; neither pilot demonstrates improved semantic-policy strength. On identical observed candidate slates, the NN changed only1/98 eligible greedy choices relative to the old guarded head. **The incumbent remains the selected release.**

Read the concise [English report](outputs/strength-round03/strength_report.html), [G results](outputs/strength-round03/job-2133983/final-summary.json), [operations audit](outputs/strength-round03/job-2133983/final-operations.json), and [selected release manifest](outputs/strength-round03/selected-release.json). Training, evaluation, failure accounting and artifacts are complete for this round; tournament-wide superiority and actual tournament deployment remain unverified.

Use the dedicated `werewolf_agent.competition_entry` for a selected competition profile; see [release instructions](rounds/03_strength/16_deployment_release.md) and [the frozen validation contract](rounds/03_strength/17_greedy_validation_contract.md). The prototype entry examples below remain historical controls and do not select the experimental neural release automatically.

## Baseline toolkit

- A modular baseline now includes private-role/seer evidence parsing, soft public-vote and message evidence, role-aware actions, a small discussion policy, and legal-action selection.
- Decision logs are append-only JSONL and intentionally exclude platform tokens and arbitrary raw API payloads.
- The official starter remains a separate checkout under `starter/` and is excluded from this repository. Install this package into the starter environment and pass `--agent werewolf_agent.entry`.
- No raw match logs or account credentials are included in this public repository. Aggregate results and synthetic diagnostic records are versioned for reproducibility.

## Run through the official starter

Requires Python 3.11+ and the official AltruAgent starter, whose repository is linked from the platform's Agent Guide.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

In the starter checkout, with its environment active and its `.env` configured privately:

```bash
pip install -e /path/to/werewolf-agent
git apply /path/to/werewolf-agent/patches/starter-game-result.patch
python -m agent --match --agent werewolf_agent.entry
```

The starter patch calls the policy's terminal hook for every seat, including
eliminated seats. Run `python scripts/summarize_logs.py data/decision_logs`
from this repository to summarize logs stored there.

Use `--tournament` only for registered tournament play. Do not commit `.env` or credentials. `WEREWOLF_LOG_DIR` can point to a protected local directory for decision logs; default is `data/decision_logs/`.

## Project plan

The [training report](rounds/01_training/05_closeout.md) links the offline
benchmark, frozen weights and official pilot replays. The experimental
policies are enabled only for practice. Their online comparison uses one
focal seat with the rule policy in the other six seats.

Use `WEREWOLF_CANDIDATE` to select `direct`, `belief`, `search` or
`reinforce`. Set `WEREWOLF_CANDIDATE_SEAT` to a zero-based seat. The
`compare` mode assigns one candidate and seat from the match ID before
the roles are observed. The LLM modes can use a predownloaded model via
`WEREWOLF_CPU_MODEL` in a Hoffman2 compute job with PyTorch and
Transformers installed. Set `OPENAI_MODEL` to the model name for the
experiment log.

See [WORKPLAN.md](WORKPLAN.md). The next steps are benchmark logging, a balanced Testing batch, and strategy iteration against fixed policy versions. Only adopt imitation learning, RL, or a local simulator if controlled evaluation supports it.

## Platform rules

The platform has 7 seats: 2 wolves, 1 seer, 4 villagers. Each registered contestant has one seat policy. Internal analysis modules may propose beliefs/actions, but one coordinator sends the final legal move. Each seat may use only its own role information, public game state, and received messages; logs must not leak one seat's private role into another seat's policy input.

## Grounded policy experiment

The prospective 25-game pilot favored legacy IDs (9/12 wins) over grounded
(5/13). Grounded is not promoted. Legacy IDs remains the competition incumbent.
The subsequent experiment kept its prompt/ranking/speech and enabled only
`WEREWOLF_EVIDENCE_GUARD=verified-innocent-v1` to exclude privately checked
nonwolves from a seer vote when unknown targets remain. The
[incumbent pivot](rounds/03_strength/13_winning_incumbent_pivot.md) records the
revised training and complete-policy comparison. Those evaluations are now complete above; the guard and neural candidate were not promoted.

The [strength round](rounds/03_strength/00_intake.md) adds an experimental
`WEREWOLF_PROPOSER=grounded` backend. It separates confirmed observations from
player claims, binds alternatives to explicit target seats and applies versioned
evidence constraints. Raw proposal diagnostics and policy guard checks are
reported separately. The reasoned-v1 trial performed worse than its control;
neither a larger model nor generated reasoning is presumed to improve play.

`werewolf_agent.competition_entry:create_agent` constructs the selected hybrid
policy for any assigned contestant seat, including tournament contexts. Set a
staged `WEREWOLF_MODEL_PATH`, model revision/device/dtype, explicit
`WEREWOLF_HYBRID_MODE` and `WEREWOLF_HYBRID_WEIGHTS` for selector modes. This
entry point is separate from the focal-seat practice table. Its existence does
not establish a tournament win-rate advantage.

For seven-seat self-hosted Testing, `werewolf_agent.hybrid_entry` with
`WEREWOLF_FOCAL_COMPETITION_FACTORY=1` invokes that exact factory only at the
focal seat and uses the configured six opponents elsewhere. This permits a
production-policy canary with one model instance per game. A controlled
comparison uses `WEREWOLF_HYBRID_COMPARE_PROPOSERS=ids,grounded` and a frozen
comparison seed/checkpoint. Keep controller creation receipts for every game;
`scripts/summarize_strength.py` includes absent workers and unresolved results.

New sampled cohorts use `werewolf_agent.collection_entry` with an explicit
starting checkpoint and `WEREWOLF_COLLECTION_SEED`. It assigns one focal seat
before observations and versions the cohort separately from proposer comparisons.
`werewolf_agent.selector_compare_entry` compares ordered `incumbent` and
`adapted` paths supplied by `WEREWOLF_SELECTOR_CHECKPOINTS`. Keep the proposer,
roster and mode fixed; set `WEREWOLF_EVALUATE_ONLY=1` for sampled evaluation.
The [on-policy contract](rounds/03_strength/11_onpolicy_contract.md) records
the current training and held-out evaluation design.

`WEREWOLF_CPU_THREADS` optionally caps each HF process's Torch intra-op threads
and is included in its provenance. Budget concurrent processes together within
the Slurm allocation. Cyclic display-order aggregation is an opt-in diagnostic,
`WEREWOLF_GROUNDED_ORDER_BALANCE=cyclic`; the full-context seven-option test
exceeded the initial 30-second soft budget, so it is not the default policy.

## Full-state strategy and contextual learning (development)

The [method audit](rounds/04_strategy/03_method_audit.md) found trusted-fact
scoring and late-transcript bugs, narrow action/speech candidates, state
aliasing and insufficient fresh-batch learning in the earlier experiment.
The new opt-in path adds owned persistent memory, all engine-legal actions,
typed contextual speech plans and ACK-controlled responses. It does not
replace the incumbent or establish a win-rate improvement.

The structured actor and separate critic use per-slot frozen text embeddings,
typed player/reference relations and attributed MIT attention/PPO primitives
from SLA/MAPPO; see [sources and adaptations](werewolf_agent/vendor/SOURCES.md).
The public MARSHAL-Generalist-Qwen3-8B checkpoint is a screened transfer
candidate, not a Werewolf-trained champion. Model IDs/revisions and the
[literature fit](rounds/04_strategy/07_literature_fit.md) are recorded explicitly.

The [runtime contract](rounds/04_strategy/12_runtime_contract.md) defines the
source-pinned teacher/sample/greedy factory. The
[data and learning contract](rounds/04_strategy/13_data_learning_integration.md)
defines fresh-journal verification, game-disjoint teacher imitation (C0), and
multiple fresh PPO batches with the previous trained critic. JSONL requests
retain actual probabilities even when the fixed executor rejects or overrides
them. BC confirmation filters do not censor PPO requests. Death is distinct
from eventual faction terminal; incomplete games are never fabricated losses.

The scripts `prepare_contextual_data.py`, `train_contextual_bc.py` and
`train_contextual_ppo.py` require Slurm and explicit input/config/output paths.
Runtime and optimizer checks use synthetic fixtures; actual new-generator
collection, service throughput and prospective win-rate comparisons remain
required before promotion. Follow the [execution receipts](rounds/04_strategy/08_execution.md)
for current verified status rather than treating source implementation as
completed competition training.
