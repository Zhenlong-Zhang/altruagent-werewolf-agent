# Joint policy progress

The joint policy shortlists legal game actions and speech intents with the
LLM, then samples from the RL selector. Its comparator chooses the LLM's
first offer from the exact same slate. The joint policy currently uses
controlled templates to express speech intents. Templates support timed
reveal, targeted questioning, presenting evidence, coordinating a vote and
passing. A proposer with free-form generation is a follow-up experiment.

The independent review verified handling of deterministic fallbacks,
generator versions, logged action-selection probabilities and delivery.
The hybrid trainer updates only a frozen, on-policy collection with final
faction returns. It does not consume the old simulated-action weights.

## Official integration

Hoffman2 Slurm job 1874432 completed a full seven-seat Testing game.
The focal seer won. Both sampled decisions, a night investigation and a
discussion action, received platform delivery acknowledgements. The day
vote offered a single known-wolf candidate, so it incurred no stochastic
gradient. The trainer accepted this game and did not update weights with
only one episode. [Pilot summary](../../outputs/hybrid/official-pilot.json)
[Official replay](https://platform.altruagent-game.com/replay/f1e376a7-90f7-4f5b-9bd5-ba6788f30639)

## Candidate quality

Both open Qwen checkpoints made 18 of 21 preferred decisions in the initial
ascending-order synthetic probe. Inspection showed both always chose the
smaller seat ID, so the accuracy is fully explained by that shortcut.
Counterbalanced prompt order and binary strategic judgment are evaluated
as separate development variants. They are diagnostic checks, never match
wins. Development logs and full per-case JSON stay on Hoffman2 until
summarized.

## Selection gate

Neither this one official game nor synthetic proposal tests establish
tournament strength. Freeze the improved proposer, collect against multiple
opponent mixes, fit the selector from official outcomes, then compare trained
and initial selectors and LLM-first against fresh full games. Retain the
same proposal slate and speech realization in comparator arms. Report
fallback, latency and missed delivery with the same denominators as the
win-rate comparison.

Paper decisions and benchmark design are in [the round contract](02_contract.md).
