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
failed before downloading because the account's shared scratch quota was
full. Its blocked dependent job 1875323 was canceled. Replacement job
1875375 requests one GPU with at least 32 GiB and stages Qwen3-8B and
Mistral-7B-Instruct-v0.3 sequentially on scheduler-managed node-local
storage. The 20B candidate needs a separate hardware and runtime check.

Job 1875375 received an H200 but failed while pip downloaded PyTorch.
Jobs 1979164 and 1979255 were canceled while pending. Job 1979285
uses CUDA 12.8 wheels and also permits the L40S, H100 and RTX PRO 6000.
It stages Qwen3-8B, checks its proposer, and starts official Testing
collection on the same allocation. The job checks collected logs every
30 seconds and writes the first selector checkpoint after 30 usable
matches. The platform's Testing page creates the matches separately.

Job 1979285 completed both probes on an RTX PRO 6000 Blackwell.
Qwen3-8B ranked the reference first in 20/42 ID presentations and 4/42
Yes/No presentations. The first-option control scored 21/42. The job
stopped before collection because it looked for the private environment
file in the starter checkout. Replacement job 1982203 reads the file from
the project checkout. At that stage no Qwen3-8B selector checkpoint had been produced.

Job 1982203 connected to Testing but received no games and timed out after
six hours with zero usable episodes. Collection requires creating matches
on the Testing page. On November 19, match
`edd71920-6f66-4ff3-9304-f5ba912d8290` was created with seven self-hosted
seats. CPU latency job 2104644 started on n6134 after reducing its request
to four CPUs and 48 GiB. CPU collection job 2105587 started on n1184.
GPU job 2104491 was canceled while pending to keep one collection runtime.
The CPU collection then demonstrated delivered actions before scaling games.

## First Qwen3-8B parameter update

Job 2105587 completed on n1184 using four CPUs and 48 GiB. Qwen3-8B
ran in float32 from revision `b968826d9c46dd6066d109eabc6255188de91218`.
Its development probe returned valid proposals in all 42 presentations,
with a 5.662-second median and 12.899-second maximum including cold start.
The older-node timing probe, job 2104644 on n6134, reached its 15-minute
limit before producing a complete probe result.

The official collection completed 30 eligible focal-seat games against six
rules opponents, covering 17 villager, eight wolf and five seer roles.
All 65 sampled choices had accepted delivery acknowledgements, with zero
model fallbacks. Maximum measured model time in collection was 15.157
seconds. The collecting policy won 16 games. These are training outcomes.

One on-policy REINFORCE update produced checkpoint `d7ebf7670444067a`
from parent `c2cdbc1abfba2673`. Seventeen of the 20 selector parameters
changed, with L2 change 0.005207830814243517. A second training invocation
inside job 2107750 reproduced the checkpoint byte for byte from the
archived trajectories.

The checkpoint, training summary, exact runtime dependencies and source
checksums are in [the round artifacts](../../outputs/hybrid/qwen3-8b-round1/).
The 210 seat logs are retained locally under
`data/qwen3-8b-round1/decisions` and on Hoffman2 under
`outputs/hybrid-2105587/decisions`. The public artifacts include the starter
commit and runtime patch needed to record outcomes and delivery.

Job 2107750 completed on n1182 with the trained checkpoint in
`hybrid_greedy` mode. All three fresh games completed, all seven selected
decisions received accepted acknowledgements, and no model fallback
occurred. Maximum model time was 20.502 seconds. All three games were
wins, across two villager roles and one seer role. Their match hashes are
disjoint from training. These three games check operation and checkpoint
loading. A comparative win-rate assessment against LLM-first and the
initial selector remains open.

The checkpoint and summaries have been downloaded to the local project.
Training and validation jobs both ended with exit code zero. Full
validation logs are retained under `data/qwen3-8b-validation/decisions`.
The webpage login token expired during the attempted 29th creation;
refreshing the page restored creation, and all 30 training games completed.

[Hoffman2 Slurm](https://www.hoffman2.idre.ucla.edu/Using-H2/Computing/Computing-Slurm.html)
[Hoffman2 temporary storage](https://www.hoffman2.idre.ucla.edu/Using-H2/Storage/Storage.html)
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
