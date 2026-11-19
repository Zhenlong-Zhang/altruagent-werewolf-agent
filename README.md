# AltruAgent Werewolf agent

A role-aware, inspectable Werewolf policy and match-analysis toolkit for the AltruAgent tournament. The aim is measured playing strength; this repository does not assume that RL is automatically the best method.

## Current status

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
The next experiment keeps its prompt/ranking/speech and enables only
`WEREWOLF_EVIDENCE_GUARD=verified-innocent-v1` to exclude privately checked
nonwolves from a seer vote when unknown targets remain. The
[incumbent pivot](rounds/03_strength/13_winning_incumbent_pivot.md) records the
revised training and complete-policy comparison.

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
