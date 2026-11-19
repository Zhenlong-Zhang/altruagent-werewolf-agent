# Worker implementation log

## Contract and ownership

Planner wrote `01_plan.md`. Reviewer accepted `02_contract.md` before implementation. The main agent owns scheduler work, real model calls, comparison assignment and match orchestration. Worker owns `variants.py`, `hybrid.py`, `competition_entry.py` and the focused regression script.

## Implemented intervention

- Added opt-in `WEREWOLF_PROPOSER=reasoned`. It generates a short tentative rationale, then uses that rationale in a second offered-ID ranking pass. This retains a legal ranked-ID interface and tests reasoning conditioning while preserving the existing ID and Yes/No controls.
- Added `WEREWOLF_MODEL_DTYPE` with the existing `auto` behavior retained. BF16 is an explicit measured-runtime candidate. Actual loaded dtype and staged revision are logged. A mismatched explicitly pinned SHA is rejected.
- Added a monotonic soft budget beginning before cold loading, checked before/after preparation, generation and final ranking synchronization. The generation stage receives the remaining time. Model operations can finish after a check, so the implementation rejects late results and records the overrun rather than claiming preemptive interruption.
- Diagnostics contain generated rationale tokens, truncation, bounded rationale text, stage times and budget exhaustion. The rationale remains internal. Delivered speech uses the existing offered safe templates.
- Added versioned configuration digests for reasoned or explicitly configured precision/revision. Actual resolved dtype/revision participate on sampled model calls. Forced pre-load choices can retain an unloaded identity. The main agent was notified to select a sampled-call identity for learning and report all relevant identities. Historical identities remain unchanged when the new options are absent.
- Added an explicit tournament opt-in to HybridPolicy and a separate competition factory for arbitrary assigned seats. The existing practice policy retains its default tournament fallback. Selector competition modes require an explicit checkpoint, whose existing schema validation remains active.

## Focused verification prepared

`python -m scripts.check_strength_regressions` contains 16 focused cases. It uses tiny Torch tensors and fake models/tokenizers, without downloading weights or running a neural language model. Main will execute it in a Hoffman2 Slurm runtime.

Coverage includes reasoning-conditioned ranking, legacy direct ranking, empty rationale, NaN scores, cold-load budget, final post-synchronization lateness, unsupported reasoned backends, malformed/unknown proposal IDs, deterministic legal fallback, stale singleton diagnostics, arbitrary-seat production action/message routing with real seat-visible facts and checkpoint logging, the existing practice guard, malformed production configuration, and precision provenance.

Reviewer prompted two corrections before execution. The late-output test now advances its clock after `Tensor.tolist`, which detects a missing post-sync check. A normal competition factory-to-action test now checks fallback selection, the logged cause and accepted delivery.

Reviewer also identified configuration parsing after a caught inference failure. The competition factory now validates the inexpensive model configuration at startup, before gameplay and model loading. The legacy path records raw requested budgets for its identity and retains legal fallback for malformed configuration. Two focused cases cover malformed budgets/dtype/proposer/revision and legacy fallback.

Local source inspection completed with `git diff --check`. Local inference, gameplay and regression execution were not performed. H2 regression and real-model evidence are pending at this log entry.

The parent froze the implementation in an isolated Hoffman2 checkout,
`~/werewolf-strength-r03`, based on commit `50a4fb0` plus the round's
source snapshot. Slurm job 2111205 requests four CPUs and 48 GiB on the
measured Xeon6 family. Its planned order is the 16 focused checks and five
orchestration checks, then controlled proposal probes and prospective
same-precision comparison. The job was pending when recorded here.

The first H2 execution reached the regression checks and found five
fixture errors before any model inference. The actual SDK requires the
`LegalAction.input` argument. Both shared fixture actions now specify
`input={}`. Their state/context fields were inspected against every policy
and logger access used by these checks. The fix changes the check fixture,
with model implementation and benchmark conditions retained. A rerun is
pending.

The repaired snapshot passed all 16 focused regression cases and all five
parent-owned assignment/reporting cases in Slurm job 2111362. The parent
then froze the implementation for real-model diagnostics and prospective
comparison. Initial BF16 ID-call measurements were approximately 1.3–2.6
seconds with the intended dtype and model revision confirmed. Detailed
probe and match results are owned by the parent and remain to be collected.

## Limits to interpret

Reasoned ranking still ends in ID-token scores and may retain numeric or presentation bias. Its quality, validity and latency require the frozen same-precision comparisons. The work does not implement a semantic embedding selector, MAPPO or learned language realization. Tournament-context stubs establish dispatch behavior; actual tournament assignment verification remains separate.

## Phase B: evidence-grounded alternative

The parent reported real reasoned-v1 failures in which unverified accusations displaced private seer facts and repeated investigations were proposed. Phase A remains a separate control; those observations do not establish effects on actual match outcomes.

Added the opt-in `grounded` proposer with `grounded-v2` prompt identity and `hard-evidence-v1` candidate guard. `grounding.py` compiles explicit PlayerN sentences with separate own-role, investigation, pack and public-reveal sources. Chat remains quoted, unverified evidence. Contradictory trusted facts raise a clear error. The rules match checked-in `werewolf.md`, including the absence of a Doctor, peaceful first night and the three **consecutive** days without elimination rule.

The new proposer performs one conditional letter-slot scoring pass; slots map back to exact offered action IDs, including opaque IDs. Targets come from explicit target fields or PlayerN labels; a numeric action ID alone never establishes a target. It generates no rationale. Presentation bias remains a risk and requires the parent's reordered/mapped probes.

The opt-in normal policy guard avoids confirmed innocents when unknown candidates remain, prefers confirmed wolves, avoids wolf partners, and avoids repeated checks. It preserves offered legal alternatives when all targets are known or no target can be resolved. Guard logs record before/after IDs, reasons and legal fallback. Raw scoring diagnostics remain distinct from forced guard choices. Historical IDs and reasoned-v1 target paths remain intact.

Local HF/HTTP model metadata now names the configured backend rather than inheriting an unrelated API model name. New provenance includes the prompt, guard, precision, revision and soft proposal budget.

The focused script now contains 23 methods. Added source/claim separation, int/string seat keys, exact opaque-ID mapping, speech null targets, conflicting facts, backend metadata and a normal arbitrary-seat competition guard test with an unchanged IDs control. NaN and final post-`tolist` late rejection checks cover both reasoned and grounded paths. Local `git diff --check` passes; execution is reserved for the parent's isolated Hoffman2 phase-B snapshot.

## Cyclic balancing boundary

The parent reported that the raw wolf-target probe always selected the second slot across both seven-rotation wolf scenarios. The accepted next intervention isolates display-slot balancing; responsive dialogue remains a proposal without code changes.

Added opt-in `WEREWOLF_GROUNDED_ORDER_BALANCE=cyclic`. For up to seven distinct candidates, each original candidate occupies every display slot once. Each pass uses the same phase-B authoritative prompt, normalizes over offered slot scores, maps back to exact original IDs, then contributes to a mean log probability. Exact ties preserve original offered order. Complete rotations cancel an additive slot-score effect; general context effects can remain.

The original one-pass branch is retained. Absent and explicit `none` produce the same configuration and generator identity. Only cyclic adds `cyclic-mean-offered-logp-v1` provenance. Diagnostics report required/completed passes, each pass's timing and mapped log probabilities, and final means. Duplicate IDs, too many alternatives, nonfinite logits/log probabilities, or a late/incomplete pass reject the entire model result. Every synchronized host conversion is followed by the same cold-load-inclusive soft-budget check.

Prepared 28 focused test methods, adding synthetic position-bias cancellation/content survival, opaque mapping and deterministic ties, second-pass NaN/late rejection, limit/enum validation, identical default/none behavior and cyclic-only identity. Local diff whitespace validation passes. No local model or test computation was run; H2 execution and measured throughput remain pending.

## Explicit CPU thread boundary

The parent requested independent thread control for multiple HF processes sharing a larger Slurm allocation. Added optional `WEREWOLF_CPU_THREADS`, validated as a positive integer no larger than `SLURM_CPUS_PER_TASK` when available. Validation happens before client/model loading. Every explicit local HF call sets and records the requested intra-op thread count, including calls with an already cached model. No model or Torch import is introduced into factory configuration validation.

Only explicitly configured runs add `requested_cpu_threads` and `actual_cpu_threads` fields; generator provenance includes the raw request and resolved actual count. A malformed legacy setting is caught by the existing proposal fallback and its raw value remains available in the log. Absent settings retain the historical cold-load thread setup and configuration identity.

The focused script now has 31 methods. Three new cases cover startup rejection of zero, negative, fractional, malformed and over-allocation settings; two cached calls applying the explicit setting and versioning absent/4/2 threads; and deterministic legal fallback with raw malformed-setting diagnostics. Static whitespace checks pass. Execution remains delegated to the parent's next isolated Hoffman2 job.

## Incumbent-preserving evidence guard pivot

The parent reported prospective phase-B results favoring IDs over the broader grounded policy and requested retaining the incumbent while isolating the known-innocent voting repair. Added `WEREWOLF_EVIDENCE_GUARD=verified-innocent-v1` independently of proposer selection. Its only new action behavior occurs for seer/day_vote: normalize private seat keys, retain existing confirmed-wolf priority, and otherwise exclude verified nonwolves if an unknown eligible target exists. All-known candidates remain legal alternatives. The historical target resolver, rank prompt, ranker and speech templates remain intact.

Only enabled settings add `evidence_guard_version` to configuration and generator identity. Absent and explicit `none` remain identical to the incumbent. Invalid guard enum values fail inexpensive production startup; the legacy proposal path catches the failure, returns a deterministic legal action and records the raw requested guard in provenance.

Prepared 35 focused methods. Four additions check absent/none prompt+configuration+generator identity; normal competition dispatch with opaque IDs, string private keys, all-known fallback and confirmed-wolf priority; unaffected role/phase and speech paths; and malformed production/legacy configuration. The intervention remains explicitly mechanical, with forced choices distinct from model-ranked successes. Local diff checks pass; all runtime execution and new collection remain the parent's Hoffman2 responsibility.

## Final execution evidence

The parent executed the later frozen semantic/PPO contract and readiness checks on Hoffman2, then collected E30, F53 and G54 games. Training2123930 completed; evaluation2123942 and2133983 completed; final read-only audits2136086 and2141810 passed. Allocation-accounting.psv retains the initial failed training2122369 and duplicate recovery audit2146450; the latter rejected existing immutable audit files without model or gameplay changes. Final JSON, input hashes, checkpoints and the concise English report are in outputs/strength-round03. Independent Reviewer verified all378 G raw-log hashes and the finite stopping rule. Exact results, the retained-incumbent decision and unresolved tournament limitations are recorded in05_closeout.md.
