# Strength round implementation contract

The user prioritizes tournament playing strength. The parent owns experiment sizing, scheduler resources, platform games, policy promotion and final interpretation. Planner guidance is in `01_plan.md`.

## Capability intervention

- Preserve the historical `ids` and `judge` local proposal algorithms and default float32 CPU loading behavior.
- Add opt-in `WEREWOLF_PROPOSER=reasoned`. Generate a short preliminary rationale using the acting seat's offered alternatives and authorized observations, then condition an offered-ID token reranking pass on that rationale. This is a reasoning-conditioned version of ID ranking. A semantic-embedding selector and free-form delivered speech are separate future interventions.
- Use `WEREWOLF_REASONING_MAX_TOKENS=48` and `WEREWOLF_REASONING_SECONDS=30` as revisable initial engineering defaults. Validate configuration. One monotonic soft compute budget starts before cold loading and covers both stages. Check before and after each stage, pass the remaining bound to generation, and reject late output. A running model load/prefill cannot be synchronously preempted by these checks, so measured end-to-end elapsed remains essential.
- Empty or invalid rationale, nonfinite ranks, elapsed budget, or an inference exception must reach the existing deterministic legal fallback with explicit diagnostics. A token-capped preliminary rationale may be used as tentative analysis when nonempty, with truncation recorded. The second-stage rank list must contain distinct offered IDs only. Existing speech templates remain the delivered messages.
- Add `WEREWOLF_MODEL_DTYPE=auto|float32|bfloat16|float16`. `auto` retains current CPU/CUDA behavior. Explicit dtype changes receive distinct provenance. BF16 execution and latency must be measured on Hoffman2.
- Version reasoned prompting, token/time budgets, requested model revision, actual loaded dtype/device, and preliminary-generation diagnostics. Preserve the exact historical generator identity when no new precision/revision configuration is supplied for the legacy proposer. Changed precision or proposer identities never silently form the same on-policy training batch.

## Competition interface

- Add a dedicated `werewolf_agent.competition_entry:create_agent` factory that constructs the configured HybridPolicy for any assigned seat.
- Add an explicit constructor opt-in for tournament action and message processing. Historical practice routing and default tournament fallback remain available through their existing entry points.
- The competition factory accepts an explicitly configured selector mode and checkpoint, validates the mode/backend, and enables hybrid action/message paths even when tournament_id is present. Sampling may use the existing collection selector mode when chosen intentionally; mode and checkpoint remain logged.
- Tests must verify arbitrary seats, tournament-marked action and speech, legal fallback and unchanged default practice guards.

## Ownership and verification

- Worker owns `werewolf_agent/variants.py`, `werewolf_agent/hybrid.py`, `werewolf_agent/competition_entry.py`, focused regression checks and `03_worker_log.md`.
- Parent owns `hybrid_entry.py`, comparison assignment, benchmark expansion, scheduler scripts, real-model timing and official games.
- Reviewer independently checks the implementation and targeted checks before the comparative result is accepted.
- All execution of inference, gameplay, training and regression checks is performed through Hoffman2 Slurm. Local work is source editing and inspection.
- Focused stubs will exercise reasoning conditioning, late-output rejection, provenance changes, malformed/unknown IDs, legal fallback and tournament routing. Existing archived data/checkpoint compatibility is retained.
- Real-model diagnostics compare IDs and reasoned at the same dtype, model revision and offered alternatives. Official A/B retains the frozen selector, mode and opponent roster and assigns arm/seat before observing role or outcome. Any selector adaptation uses a separately versioned on-policy batch.

## Claim boundary

A passing integration test establishes execution. Development probes establish behavior on those cases. Prospective match results and uncertainty determine whether playing strength improved. A bounded proposer, larger model or short winning streak alone does not establish tournament superiority.

## Evidence-grounded v2 phase

The parent approved a distinct `grounded` intervention after the frozen
reasoned-v1 probe produced false private facts and repeated-investigation
recommendations. Remote v1 worktrees remain controls. Local next-phase
implementation may add a new pure evidence module and opt-in dispatch.

Grounded-v2 compiles the acting seat's private facts and public death
reveals into source-labeled PlayerN sentences, labels chat as unverified,
and states the exact platform roster and rules. It annotates each offered
target's confirmed alignment and investigation status. One offered-slot
token ranking pass maps single-token display slots back to exact offered
IDs, including arbitrary nonnumeric IDs. It generates no rationale.

An opt-in common policy guard removes verified innocents from seer day
votes when unknown candidates exist, retains confirmed-wolf priority and
fresh-investigation filtering, and preserves legal fallback for all-known
or empty target sets. Chat claims provide soft evidence only. The same
helpers apply to practice and competition policy paths. Raw proposal
probes measure ranking with compiled inputs; separate full-entry checks
measure candidate guard behavior. Forced decisions must remain visibly
separate from model-ranking successes.

New prompt/guard versions, actual precision/revision and phase diagnostics
receive distinct provenance. The local-HF model name in call/pool metadata
will be corrected at this phase boundary. Focused fixtures will cover
int/string knowledge keys, PlayerN/action-ID separation, unverified claims,
verified-innocent avoidance and repeated checks. Promotion remains subject
to new model and game evidence.

## Cyclic display-slot balancing phase

The parent approved a distinct opt-in `WEREWOLF_GROUNDED_ORDER_BALANCE=cyclic`
after wolf-target diagnostics showed systematic second-slot selection.
Absent or explicit `none` retains the phase-B one-pass algorithm, prompt
and generator identity. Social dialogue changes are deferred.

For K distinct offered candidates, K<=7, score every cyclic rotation with
fresh A.. display slots. Normalize log probabilities across the offered
slot scores for each pass, map them back to exact original IDs and average.
Rank by descending mean, resolving exact ties by original offered order.
Complete rotations cancel additive slot bias; this does not guarantee
invariance to general contextual or presentation effects. Unknown seer
claims remain unverified evidence and receive no new hard target mask.

All passes share the original cold-load-inclusive soft deadline. Reject
nonfinite scores, invalid IDs, more than seven candidates, or an unfinished
or late pass without returning partial ranks. Record completed/required
passes, per-pass timings, inverse-mapped log probabilities and aggregate
scores. The cyclic-only aggregation version participates in configuration
and generator identity. Tests will cover additive bias cancellation,
content preference, opaque IDs, stable ties, later-pass NaN/lateness,
configuration validation and absent/none preservation. All execution stays
on Hoffman2; adoption requires fresh raw model and prospective game evidence.

## Explicit CPU thread allocation boundary

The parent approved `WEREWOLF_CPU_THREADS` for concurrent HF processes inside
one Slurm allocation. If supplied, it must be a positive integer and cannot
exceed valid `SLURM_CPUS_PER_TASK` when that limit is present. Validate this
at inexpensive factory startup without importing/loading a model. Explicit
settings apply before every HF call, including cached models. Log requested
and actual intra-op thread counts only when explicitly configured, and
include them in versioned generator provenance. Preserve the absent-option
algorithm and metadata exactly. Legacy malformed settings must retain legal
fallback and the raw requested value in diagnostics. Focused H2 checks cover
invalid/over-allocation, actual cached-call application, changed identity,
default preservation and malformed legacy fallback.

## Incumbent IDs plus narrow verified-innocent guard

The parent, Planner and Reviewer approved retaining the stronger measured
IDs incumbent and isolating one evidence guard from the broader grounded
intervention. `WEREWOLF_EVIDENCE_GUARD=verified-innocent-v1` changes only
seer/day_vote action candidates: retain existing confirmed-wolf priority;
otherwise remove privately verified nonwolves only if an unknown eligible
target remains. Normalize private seat keys inside this guarded branch.
Retain all-known/legal fallback and the original action-target resolver.

Absent or explicit `none` keeps the incumbent configuration, generator,
prompt, ranker, speech and action behavior. Enabling the guard adds its own
version to configuration and generator provenance without selecting a
grounded prompt. Other role/phase paths and speech remain unchanged. Reject
invalid enum values at inexpensive factory startup; legacy inference failure
retains legal fallback and raw requested guard diagnostics. Tests distinguish
guarded candidate filtering from LLM reasoning and cover arbitrary IDs,
known-wolf priority, all-known fallback, string keys, unaffected paths and
absent/none identity. All checks and collection run on Hoffman2. This
boundary does not activate cyclic ranking or responsive dialogue.
