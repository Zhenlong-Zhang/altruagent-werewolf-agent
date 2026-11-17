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
Hoffman2 job 1874895 repeated every position in both candidate orders.
With action-ID scoring, Qwen 0.5B preferred the reference option in 30/42
presentations, versus 22/42 for Qwen 1.5B. Reversing the offered order
changed the model's leader in 6/21 and 16/21 paired cases respectively.
The first-option control scored 21/42. The two models' initial 18/21
scores therefore do not justify choosing either model by size.

The judgment proposer evaluates each described action or speech
intent with the same prompt and ranks the Yes/No log odds. It removes
numeric action IDs from the output vocabulary. Its counterbalanced
development comparison finished under job 1874954. Neither model changed
its top choice when option order was reversed. The reference first-choice
counts were 8/42 for Qwen 0.5B and 14/42 for Qwen 1.5B, both lower than the
action-ID variants. The judgment proposer remains experimental. These small synthetic
probes measure behavior and latency under controlled presentations.
Platform win-rate selection uses independent operating data.

## Larger model comparison

Hoffman2 allows two GPUs per user under the current account. The general
GPU request in job 1875227 received a Tesla P4 with 7,680 MiB. Slurm also
advertised nodes with 32 GiB and larger devices. Preparation job 1875317
sets up CUDA PyTorch and downloads Qwen3-8B and Mistral-7B-Instruct-v0.3
on scratch. Job 1875323 requests a single 32 GiB or larger GPU after
preparation to compare both families with the same proposal probes.
The 20B candidate remains a separate hardware and runtime evaluation.

[Hoffman2 Slurm](https://www.hoffman2.idre.ucla.edu/Using-H2/Computing/Computing-Slurm.html)
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

## Selection gate

Neither this one official game nor synthetic proposal tests establish
tournament strength. Freeze the improved proposer, collect against multiple
opponent mixes, fit the selector from official outcomes, then compare trained
and initial selectors and LLM-first against fresh full games. Retain the
same proposal slate and speech realization in comparator arms. Report
fallback, latency and missed delivery with the same denominators as the
win-rate comparison.

Paper decisions and benchmark design are in [the round contract](02_contract.md).
