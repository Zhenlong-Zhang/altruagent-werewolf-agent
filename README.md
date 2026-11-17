# AltruAgent Werewolf agent

A role-aware, inspectable Werewolf policy and match-analysis toolkit for the AltruAgent tournament. The aim is measured playing strength; this repository does not assume that RL is automatically the best method.

## Current status

- A modular baseline now includes private-role/seer evidence parsing, soft public-vote and message evidence, role-aware actions, a small discussion policy, and legal-action selection.
- Decision logs are append-only JSONL and intentionally exclude platform tokens and arbitrary raw API payloads.
- The official starter remains a separate checkout under `starter/` and is excluded from this repository. Install this package into the starter environment and pass `--agent werewolf_agent.entry`.
- No match logs, account credentials, or result data are included in this public repository.

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
