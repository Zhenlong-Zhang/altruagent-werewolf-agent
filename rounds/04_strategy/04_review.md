# Independent architecture review — Round04

Status: preimplementation review, 2026-11-20. The Reviewer independently
read the intake, R03 evidence/closeout, current source, Planner01/07 and
primary author-linked papers/repositories. No inference, training, gameplay
or tests were run locally. D7/IDs/greedy/no-guard remains the incumbent.

## Decision

Accept Planner01's route **in principle**: repair evidence/message handling,
expose meaningful legal actions and contextual speech tactics, screen a real
released strategic model on the same interface, then train a full-context
strategy policy on successive fresh batches against a population. This is a
substantial capacity change, not another coarse-state head experiment.
Implementation acceptance requires the exact02 input/action/message/rollout
contract and focused tests below. Public asset staging may proceed on H2;
no assumption of allocated GPUs or stronger Werewolf transfer is justified.

Do not substitute another thirty-game, zero-old-value, bounded-residual
update or report-only iteration for this objective. R03 established a
reproducible pipeline, but G changed only one of98 eligible recorded choices
and did not demonstrate a stronger agent. R03 F/G now serve as development
evidence, not a fresh final test for this round.

## What is wrong, and what is merely unproven

| Finding from source/evidence | Classification | Required response |
| --- | --- | --- |
| `build_suspicion` sets known nonwolf to -8, then adds repeated accusations before clamping to [-4,4]. Twenty-four +0.35 claims can make it positive. The comment that the clamp prevents this is false. | Actual correctness bug in trusted-fact preservation. | Keep confirmed facts immutable and separate from soft suspicion; repeated untrusted text must never overwrite their identity/alignment. Check final beliefs and rendered speech, not just one finite accusation count. |
| `_actions` removes Abstain whenever any target action exists; old wolf masks also prohibit partner voting. `_propose` retains a short LLM/rules slate. | Confirmed action-coverage restrictions, not demonstrated optimal strategy. Abstain omission is a coverage defect relative to the full legal-action interface. | Present all currently legal discrete actions, including Abstain when offered. Treat risky tactics such as bussing as strategy choices rather than silently asserting universal illegality. Separate legal masks from heuristic preference masks. |
| `_speech` offers generic asks/one heuristic target, positive-wolf disclosure and Pass. It lacks negative-check defense, contextual contradictions, explicit commitment updates and deliberate wolf cover stories. | Confirmed expressive-capacity limitation. | Distinct intent/target/channel/evidence candidates with actual contextual differences; multiple paraphrases of one target do not solve this. |
| `choose_message` terminates after one attempt/day; visibility is cropped to ten envelopes/240 characters and recipients are dropped from the proposer view. | Confirmed memory/control restriction. | Bounded reactive communication, durable source/channel-aware records and truthful delivery accounting. |
| Messages arriving after the early termination/update can be absent at voting because the platform clears `new_messages`; the runner does not refresh the transcript before action. | Source/guide-supported message-loss path; a precise transition fixture is needed before claiming its empirical frequency. | Fetch only the acting seat's authorized transcript before voting; merge/deduplicate sequence IDs and preserve private visibility in an isolated SDK snapshot. |
| Semantic actor sees coarse role/day/phase/alive fields plus candidate labels, not the actual claim/check/history state. | Confirmed state aliasing, not a demonstrated failure of attention or PPO generally. | Encode persistent evidence/player/commitment state; maintain separate fact slots so truncation cannot erase them. |
| R03 uses thirty terminal rewards, old critic zero, one frozen mixed roster, a strong old prior and small-KL fitting. | Valid but limited learning/data design. | Fresh multibatch policy collection, learned old values, declared credit assignment and varied opponents; more epochs are not new games. |

Hypotheses still needing experiments: more responsive speech improves wins;
MARSHAL transfers better than vanilla Qwen; richer context improves learned
selection; population training improves generalization. None follows from a
paper title, parameter count or legal delivery. Hard factual consistency does
not mean wolves must always tell the public truth: deliberate deception must
be tracked separately from the actor's true private knowledge.

## Verified reuse and adaptation boundaries

**SLA2024.** The [paper](https://arxiv.org/html/2310.18940) combines deduction,
diverse language candidates, observation/action attention and population
training; it reports BC followed by10M MAPPO steps. It is not the R03
coarse-state/one-batch setup. The [author homepage](https://zelaix.github.io/)
links [SLA](https://github.com/zelaix/SLA), verified commit
`c81f4500b0be74e4a270370ba35bb94caceb915b`. Its [MIT license](https://github.com/zelaix/SLA/blob/main/LICENSE)
permits attributed reuse; actual files include Werewolf environment/agents,
actor–critic, runner and training entry points. The source uses Doctor rules
and OpenAI generation/embedding APIs; its README's Python3.6/Torch1.5 example
is dated. Reuse verified return/mask/rollout/attention machinery with modern
adapter tests, not its environment or hyperparameters blindly. No ready
AltruAgent checkpoint was verified. The inspected HTML shaping-reward table appears inconsistent for a
Werewolf elimination; verify the PDF and actual code/objective rather than
copying that entry. Terminal faction wins remain the competition objective.

**LSPO2025.** [Primary paper](https://arxiv.org/html/2502.04686): abstract
language strategies, solve the abstract game with CFR and use preference
training to expand generation. This is relevant to action diversity but
needs a validated transition/information-set model and counterfactual
traversal. Live logged outcomes alone are not CFR. No verified official
code/checkpoint release was located through the inspected author/publisher
links; this is not proof none exists. Do not invent a ready model or call
plain PPO/clustering a faithful LSPO reproduction.

**ONUW2024.** The [official project](https://one-night-ultimate-werewolf.github.io/)
links [KylJin/Werewolf](https://github.com/KylJin/Werewolf), commit
`4daf873e83fe1e9b029cb3d39c7e57f3baa5da10`. Its tree contains discussion
policy weights and a d3rlpy training script; the framework separates belief,
tactic and realization. ONUW's role swapping/single-night rules differ.
The supplied checkpoint's representation cannot be replaced with unrelated
MiniLM dimensions and assumed compatible. No license was found in the
inspected repository tree: use the conceptual tactic decomposition or
independent implementation until code/weight permission is established.
Paper access is not a software/model license.

**A real released strategic model: MARSHAL.** The author's links lead to
[Apache-2.0 code](https://github.com/thu-nics/MARSHAL), inspected commit
`b53f38af12fe5ea1529c50c18ea3af97536c5a71`, and released
[generalist8B weights](https://huggingface.co/nics-efc/MARSHAL-Generalist-Qwen3-8B).
The model card describes strategic self-play on Tic-Tac-Toe, Kuhn Poker and
Mini Hanabi, not Werewolf. This is a concrete zero-shot contender, not a
Werewolf champion. Public model metadata/config verifies:

| Model | Revision | Actual BF16 parameters |
| --- | --- | --- |
| MARSHAL-Generalist-Qwen3-8B | `369d9decea6e8b347f81706fa169dbbddddce360` | 8,190,735,360 |
| MARSHAL-Generalist-Qwen3-4B | `267b80a4c6b6ba09a81fa7bdeacf11ebcdc3d78c` | 4,022,468,096 |

Both are ungated and model-card Apache2. The8B card's frontmatter mistakenly
names4B; actual config/parameter metadata establishes8B. The official eval
example uses a Qwen3/vLLM setup, `<answer>…</answer>` outputs and up to4096
generated tokens. Its8-GPU reference configuration is **not** a proven
minimum. It is also not an available Werewolf simulator. Adapt generation/
parsing/budget to our game and compare the same interface against vanilla
Qwen; using only the old one-token ID scorer would discard much of the
released model's intended reasoning interface. Screen reasoning/answer
formatting honestly before selecting thinking settings or training it.

The8B weights alone occupy about16.38GB decimal in BF16 before KV cache and
runtime overhead. Measure the actual allocated device, memory and throughput;
two-GPU eligibility is not an allocation guarantee. Smaller-scale LoRA/DPO
may be feasible on fewer GPUs after measurement, but reproducing the full
published distributed training is a different undertaking. A shared local
inference service may avoid model copies, provided each request, KV state,
ledger and cache are seat-isolated and queue deadlines are measured.

## Proposed architecture that can change behavior

1. **Authoritative seat adapter.** Input is the current acting-seat snapshot,
legal actions, visible message transcript with sequence/channel/recipient,
own private facts and public engine facts. Never accept terminal labels,
other-seat views or hidden evaluation roles as actor inputs. Refresh phase-
transition messages; retain an append-only ledger and bounded learned summary
with source references. Summary errors cannot mutate authoritative facts.
2. **Complete discrete actions and explicit tactics.** Full legal votes/night
moves remain available. Speech candidates carry intent, target, audience,
evidence refs, commitment change and realized text; include Pass and a
canonical legal fallback. Generate meaningfully different role-appropriate
strategies, including truthful defense/coordination and marked wolf bluffing.
Candidate set coverage is measured separately from selector accuracy.
3. **Responsive controller.** A finite opener/reply/commitment policy counts
acknowledged sends, time, quota and pending/ambiguous delivery. It can reply
to a new claim or update a vote intention without consuming the window in
repeated generic speech. Private wolf coordination reaches only an authorized
living partner. Public realization cannot leak a private channel by accident.
4. **Contextual learner.** Persistent player/evidence/history representations
and full contextual candidate records inform the policy/critic, rather than
labels plus eight coarse numbers. Use an attributed, tested MIT learning
core where useful. Fresh on-policy batches evaluate the old value function
before collection; freeze old logits/values/advantages within each update.
Multiple batches provide genuinely new information, not repeated epochs.
5. **Population/evaluation harness.** Freeze model/policy identities and
opponent-mixture sampling per training phase; cover both factions, seats,
LLM teammates and adversarial styles. Measure against incumbent, new
untrained interface and trained interface separately. Test on a held-out
mixture/fresh namespace after development selection. Old R03 games may aid
debugging or approved warm-start supervision, never masquerade as this test.

## Learning/reward and independence checks required in02

- Keep terminal faction outcome the primary objective. Speech counts,
accusation agreement, superficial persuasion and private-check disclosure
are not win rewards. Any shaping term needs provenance, motivation and an
ablation; potential-based or auxiliary objectives must not silently change
release selection criteria. Never treat false beliefs of opponents as true
facts in the actor state.
- Specify action-level times, rewards, discount/GAE/terminal/death masks and
whether asynchronous turns are accumulated per decision or elapsed turn.
An eliminated actor may receive a later faction result; do not invent an
immediate loss or bootstrap through a terminal. Mark ACK/ambiguity rather
than training a requested but undelivered message as executed behavior.
- A decentralized critic may condition only on actor-visible state. A future
privileged critic is an explicitly reviewed CTDE choice; privileged fields
cannot enter actor embeddings or inference. Faction rewards differ: sharing
one scalar among both opposing factions is incorrect. Call decentralized
adapted MAPPO code IPPO/PPO unless joint training truly implements MAPPO.
- Games, not decisions/seats/epochs, are independence units. Report cluster-
level uncertainty when collecting several learning seats in one game; do
not multiply sample size by seven. Do not duplicate one transcript with
seat permutations and claim independent outcomes.
- Fixed vs sampled/greedy evaluation, model/prompt changes and fallback
behavior define different policies. Collect new data under the actual
chosen generator/controller. If that generator changes, old propensities
alone do not make prior trajectories on-policy for the new system.

## Gate before the first code chunk

02 must freeze: observation/ledger schema and visibility rules; full-action
and speech-candidate schema; source-safe factual versus deliberate-bluff
validation; transcript refresh and message ACK/retry/quota state machine;
model/revision/template/decoding/deadline contract; learning state/targets/
masking/credit rules; reused file/commit/license attribution; actual stage
resource/throughput measurement; fresh training batches and checkpoint
selection/final-evaluation boundaries. Counts may be set after timing,
before outcome-dependent tuning. No arbitrary mandatory match count is
imposed here, but another single30-game coarse update is not an accepted
substitute for the intended capacity change.

Initial acceptance checks should demonstrate: adversarial repeated claims
cannot overwrite a verified fact; an important late message reaches vote
state; all legal discrete actions survive candidate construction; negative-
check defense/claim challenge/commitment update and marked wolf bluff are
representable; private messages cannot leak between seats; rejected/unknown
sends do not create false executed transitions; every updated policy uses
new properly attributed trajectory data; model serving meets measured queue/
response budgets. These checks establish capability/correctness. Only fresh
matched-condition AltruAgent outcomes against a diverse frozen population
can establish stronger play. Winning the competition remains the target,
not an achieved premise.

## Phase A contract review

The Reviewer read Worker03 and02 and accepts the proposed first chunk
conditionally, with two concrete schema clarifications before implementation.
The diagnosis correctly distinguishes immutable-fact/message/ACK bugs from
candidate/state/one-batch limitations; its method audit does not claim that
changing the learner alone solves missing strategic capacity.

1. **Bind statements, not just references.** Candidate assertions need typed
   subject/predicate-or-alignment/source/truth-mode fields. A valid `K1`
   reference beside unchecked text saying Player2 is a wolf does not validate
   that text. Verified own-check/public-engine statements must be rendered
   canonically or constrained to an explicitly checked form. Broader prose
   remains uncertain/attributed, with intentional wolf bluffing in a separate
   assertion mode. Never promote that mode into private facts.
2. **Own the ledger explicitly.** History/window wrappers must identify their
   session hash and acting seat. Reject a different owner and unauthorized
   private-message envelopes instead of relying on callers to label arbitrary
   records 'own history'. Preserve sender/channel/recipient/source sequence.

The complete-answer parser must consume only generated completion tokens,
not prompt text or the first JSON fragment in reasoning. Accept a complete
final answer outside reasoning, reject malformed/duplicate/truncated answer
blocks, and validate action IDs, assertion bindings and audience against the
packet. Native reasoning may be private; it is never delivered as chat.

With those additions in02, approve Worker's new state/candidate/interface
checks and parent's local backend/shared-service/GPU-screen implementation.
All current incumbent paths remain read-only. Source acceptance does not yet
approve live dialogue, learning, a rollout count or final evaluation. Later
Phase B/C contracts still require the concrete ACK/phase-refresh state
machine, fresh value-aware batches and population/evaluation boundaries.

The reported failed asset-staging attempt2183063 stopped before downloads
because shared scratch quota was full. Moving stage files into an allocated
node's temporary directory is an operational revision, not a model change;
retain the failed attempt and do not delete unrelated projects. Retry only
with the declared GPU harness/resource checks. No extra user permission is
needed for the already authorized isolated staging work.

### Phase A implementation gate accepted

The Reviewer verified the02 additions: typed/constrained assertions,
canonical verified surfaces, session/seat-owned history/window wrappers,
private-envelope visibility checks and single complete final-answer parsing.
The two requested schema conditions are met. Phase A's first new interface,
backend/shared-service and focused-check chunk is accepted for implementation
and subsequent H2 verification. This is not source/runtime acceptance of
code not yet reviewed, and does not preapprove later gameplay/training.
Stale draft-header wording was flagged for cleanup; the explicit accepted
scope is authoritative and need not delay the authorized first chunk.

## Phase A initial source inspection

Parent backend/service/screen and Worker compiler/candidate sources were
inspected before GPU submission. No local test/model execution occurred.
The backend visibly shares weights/tokenizer only, passes complete request
messages, uses owner-only UDS directory/socket permissions and revalidates
absolute deadlines after queueing. Inputs beyond the explicit token budget
are rejected, not silently truncated. Generation is bounded with per-token
soft stopping, final CUDA synchronization and late rejection. Both pinned
Qwen/MARSHAL tokenizer configurations were inspected: thinking mode prefills
only the assistant header and thinking tags are nonspecial tokens, so the
strict completion parser's intended tag handling is compatible with these
actual templates.

The following source issues were raised before freeze:

- A free rhetorical suffix guarded by regular expressions is not a verified
  factual boundary. `Player2 is definitely a wolf` and `My teammate is
  Player3` evade the first implementation's checks. This invalidates the
  claimed constraint despite correct typed assertion rendering. The parent
  accepted a principled repair: deliverable rhetoric is rendered from
  intent/target/audience; arbitrary model prose is not delivered. Strategy
  diversity remains in model-selected typed plans/assertions/targets, not
  unvalidated surface clauses. Do not label that controlled realization
  an exact reproduction of unrestricted SLA generation.
- Pending-window fields initially had only a top-level key whitelist.
  Nested untyped values could carry evaluation metadata into actor packets.
  Typed projection or omission until Phase B is required, with nested-field
  counterexamples rather than only a top-level `winner` test.
- The server initially wrote readiness before binding its socket. Fix that
  race or require an actual bound-service readiness check. An EOS on the
  last allowed output token is not necessarily length truncation; finish
  reason must distinguish it from an unfinished capped output.
- The initial backend suite exercised a stateless echo service, not HF
  adapter boundaries. Add focused H2 adapter checks for cold-load expiry,
  input rejection, token-limit/EOS handling, synchronization and native
  reasoning/new-token extraction before the real model screen.

The original42-case comparison primarily measured seer factual alignment/
new-information actions and speech-candidate coverage, not strategic wins.
The parent expanded it to70 cases per model with explicitly unscored wolf
private/public conflicts and small-table history contexts. These broaden
operation/context evidence but do not manufacture gold optimal targets.
Matched seeds, target/ID rotations, ordering, resource and decode settings
remain essential. Model selection must respect these narrow metric limits.
Source acceptance is pending repair inspection and focused checks; asset
staging's scratch-quota failure is preserved and node-local staging remains
the authorized operational fallback.

### Phase A source freeze accepted for H2 diagnostics

The Reviewer re-inspected the repairs and focused checks. Arbitrary generated
prose must now be empty; all delivered clauses come from typed assertions
and controlled intent/target/audience rhetoric, marked
`typed-canonical-v1`. Verified subject/alignment/source bindings, negative-
check semantics and deliberate wolf-bluff separation remain explicit. The
attribution edge was also repaired: lexical source mentions render only a
neutral message/speaker/subject clarification, not a claim of semantic
entailment. Negated/quoted-source counterexamples are in the suite.

Pending-window objects are rejected entirely in Phase A, deferring the typed
pending controller to Phase B and closing nested metadata injection. Private
check parsing is line-bounded; soft-score computation uses reconstructed
trusted observation content, followed by immutable alignment projection.
Owner/private-envelope checks, full legal opaque-ID actions and abstention
remain intact. Sixteen interface methods are prepared for H2 execution.

Backend readiness is now atomically published after socket binding and
permissions. Actual EOS is distinguished from a length cap, including EOS
at the final permitted token. Eleven backend methods cover UDS privacy/queue
TTL and fake-HF native reasoning/new-token, oversize-input, cold-load expiry,
EOS/length and late-generation paths. No local execution was performed.

The70-case-per-model development screen and Slurm batch were reviewed:
matched pinned vanilla/MARSHAL native generation, temperature0.6,2048 new
tokens,30-second soft budget,8192 input tokens; one BF16-capable allocated
GPU, four CPUs,48GB and two hours. Models are staged sequentially in owned
node-local temporary storage. No official key, live game or learning update
is used. Unscored wolf/endgame cases remain unscored; seer reference/coverage
metrics do not establish strategic wins.

No material source blocker remains for this isolated freeze and actual H2
GPU diagnostic. The27 prepared checks and real model outputs remain pending.
This accepts Phase A capability/operation screening only. Persistent bounded
summaries, phase-transition transcript refresh, responsive ACK controller,
fresh contextual learning and population evaluation still require their
later concrete implementation/data/budget reviews.
