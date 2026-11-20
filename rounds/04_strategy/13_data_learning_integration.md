# Fresh collection, imitation and PPO integration

Parent contract amendment before live collection, 2026-11-20. Core source tests
have passed; this extension still needs independent review and Slurm checks.
It does not claim a new trained platform policy or playing-strength result.

## Collector evidence

`contextual_data.py` and `prepare_contextual_data.py` bind each registered focal
game to the precollection protocol, session hash/seat, archived runtime config,
verified source/SDK manifests, actual generator metadata and checkpoint.
Registration is performed before launching that game's worker. The registry
accounts for every created game, including absent/unresolved journals. Archive
the small runtime JSON beside the collection; node-local model paths inside it
may expire, but the configuration and its hash must survive. No credentials are
part of these files. Private packets/journals remain outside the public repo.

The protocol contains exact `schema,collection_id,mode,checkpoint_id,lineage,
source_manifest_sha256,sdk_manifest_sha256,plan`. Schema is
`contextual-collection-protocol-v1`; protocol_hash is its canonical fingerprint.
Plan declares finite `maximum_created_games` and `target_completed_games`, with
population/assignment/resource choices recorded before outcomes. These are
revisable engineering quantities, not a user-imposed cumulative resource cap.
Do not silently omit over-cap games: record the violation and preserve all
registrations. A new justified protocol may follow; no new batch budget or
population weights have yet been selected from platform outcomes.

Registry JSONL rows include `game_id,session_hash,focal_seat,collection_id,
protocol_hash,journal_path,runtime_config_path`. One focal learner/game. Session
hash must equal SHA256(game_id)[:16]. Complete runtime provenance precedes all
requests; decision IDs increase across reconnects. Recompile each exact packet
and candidate list and compare the serialized behavior snapshot. Source/config
file hashes and the actual portable fitting implementation files are rechecked by the trainer immediately before fitting. Archive the small source and SDK manifests too; expired model-weight paths are not required during fitting.

The official SDK `get_result` own return is authoritative, with a verified
terminal/session audit and score ±1. Initial Werewolf labels additionally require
`termination_reason=completed`, as documented in the project's pinned
05-gameplay.md. Scheduler failure, missing receipt or eliminated-before-team-end
is not a negative return. Unknown delivery is not itself an exclusion: sampled
requests and their original probabilities remain eligible when the authorized
state chain and eventual faction outcome are verified.

## BC and C0

BC uses only the actual native teacher preference, resolved to the exact offered
candidate, with a confirmed same-request action receipt or canonical chat ACK /
authorized transcript reconciliation. Gated, rejected and unresolved teacher
requests remain in raw accounting; they are not relabeled as alternative
teacher choices. This confirmation filter is **BC only**, never PPO filtering.

`train_contextual_bc.py` initializes the structured network and fits game-mean
teacher cross entropy, plus a separately parameterized training-only critic
warm start from eventual faction scores. Games split deterministically by
seeded ID hash; the development set contains whole games. Fit for an explicit
finite configured number of epochs and retain the earliest minimum development
cross-entropy checkpoint, with matching actor/critic optimizer states. Call it
pre-PPO **C0**, not an untrained model or a winning policy. Development teacher
accuracy is imitation accuracy, not game win rate.

The first implementation uses completed teacher games for a common actor/value
warm-start cohort. This is a simplifying execution choice: supervised teacher
targets do not inherently need final outcomes. If missing terminal coverage
becomes material, valid confirmed teacher prefixes can be retained under an
explicit revised actor-only BC protocol; do not pretend they have value labels.

## On-policy continuation and reconnects

`train_contextual_ppo.py` loads hash-bound safetensors and safe optimizer state,
requires the new verified sample-mode batch, and replays probabilities/values
before updating. Each new rollout batch uses the previous trained critic.
Changing focal generator/controller/encoder/source requires a new lineage;
population versions may change only between declared batches. A training report
with zero accepted steps is not described as a successful RL improvement.

The actor objective divides its summed nonforced request terms by **all games
with requests**, including singleton-only games with zero actor contribution.
Critic-only/singleton-only batches cannot move the actor via stale Adam momentum.
Checkpoint IO retains weight/report/optimizer hashes and refuses overwriting.

Sampling now uses `owner-decision-sha256-uniform53-v1`: owner/seed plus persisted
decision ID determine each pseudorandom uniform independently. This avoids
replaying the start of a stream after reconnect. Native model generation separately derives a uint32 seed from the base seed, owner and persisted decision ID; different games/seats no longer share the same base-plus-counter generation seed. The adapter verifies the
recorded uniform from the archived config, and PPO verifies the selected draw
against the full probabilities. Counters/polls never fabricate policy steps.

Mixed population routing gates transcript hooks on the **returned seat policy**,
so a router may return an unchanged D7 incumbent without invoking new callbacks.
The actual patched-SDK fixture now includes that route in addition to new-policy
flush, failed refresh and direct legacy behavior.

## Remaining empirical work

Measure the selected real-model service with synchronized authorized seat
requests before fixing population size/concurrency and finite BC/PPO budgets.
Then collect new teacher games, fit C0, run multiple fresh rollout/update batches,
and compare incumbent/C0/adapted checkpoints on a separate prospective cohort.
The GPU screen remains queued; the completed CPU probe's four90-second native
calls all timed out. This is a deployment timing result, not a comparison of
strategic model quality. No final tournament deployment is claimed.
