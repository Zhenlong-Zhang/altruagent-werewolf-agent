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

### Phase C conceptual learning review — proposal10

The Reviewer read `10_learning_contract_proposal.md` in full. Conceptually
accepted for the later concrete source contract, **not yet C implementation,
collection, or model acceptance**. The route adds real capacity relative to
R03: protected source-bound player/fact/claim tokens, contextual candidate
cross-attention, all engine-legal actions, a separately parameterized value
network, fresh successive on-policy batches and a matched pre-PPO C0 control.
The proposed reuse identifies actual MIT attention/optimizer/buffer
components while removing joint-private-state assumptions; it does not
mislabel this decentralized learner as a faithful MAPPO reproduction.
The exact Phase B transport/controller must be settled first.

The request-action formulation is sound and preferable to ACK-conditioned
sample deletion. If the policy samples request a with recorded probability
pi_old(a|authorized observation, frozen slate), a frozen executor may reject,
delay, acknowledge or deterministically fall back. Those are transition/
execution outcomes of that request, not grounds to replace its probability
with a fictitious probability of the fallback. A rejected draw can therefore
remain on-policy. The frozen generator's stochastic slate is observed before
the actor draw; its generation/validation identity and the whole candidate
slate must be retained. Changing generator/controller/realizer/population
inside a collected batch breaks the claimed fixed transition conditions.

Source-contract conditions and concrete counterexamples:

- **Unknown delivery is not automatically off-policy.** Keep exact pending
  uncertainty as the actor observed it. A completed game with intact sampled
  request, behavior probabilities, actor observations and faction result may
  remain usable even if the receipt was ambiguous. No ACK-dependent filtering
  is justified merely by saying on-policy. Missing sampled-action provenance,
  missing later observations or unknown faction result is different: archive
  it with a declared eligibility/censoring reason. Do not promote a future ACK
  into an earlier packet or issue blind retries that can duplicate delivery.
- **Use separate return masks.** At true faction terminal both bootstrap and
  GAE recursion are0. At an authorized clean fragment boundary bootstrap
  uses the stored old next-value, but recursion stops: bootstrap mask1,
  recursion mask0. A single continuation flag in both recurrence positions
  cannot express this. Death is neither a technical loss nor a faction
  terminal. Attach eventual faction reward to the dead seat's final own
  decision without placing it in any earlier actor input. Unresolved transport
  is not a clean bootstrap boundary just because a time budget expired.
- **Expose controller state and preserve observation causality.** Pending
  request, attempt/ACK quota, stale-window status and available execution
  time affect decisions and transition behavior. Serialize their authorized
  pre-action values. This remains a policy over bounded partial observations;
  do not claim the summarized packet proves a fully observable Markov state.
  Preserve source/channel/subject bindings and explicit overflow handling.
- **Freeze the estimator and reduction.** Gamma1 matches eventual faction
  return without wall-time discount. Lambda0.95 remains an estimator choice:
  inserting additional own speech/request events changes GAE propagation,
  and bootstrapped finite-sample bias can remain. Specify per-game/per-seat/
  per-role weighting before collection so a long discussion or games with
  more learning seats do not silently dominate. Correlated seat trajectories
  remain one game cluster for uncertainty, not independent games.
- **Pin exact learner mechanics before code.** Define token/numeric schema,
  pointer and padding masks, actor/critic parameter partition, categorical
  entropy, old-value/value-clipping conventions, advantage normalization,
  minibatch/reduction, optimizer settings and full-state rollback semantics.
  Candidate permutation tests must include cross-referenced evidence and
  targets; padding and unsupported overflow must not create sampleable
  actions. Replay stored old full-slate probabilities and values before an
  update, including rejected requests and multiple alive/dead seats.
- **Separate learning and generator evidence.** C0's recorded demonstration
  teacher decisions need identity/source provenance and game-disjoint
  validation. BC loss alone does not establish strength. More than one fresh
  rollout/update batch is needed to exercise the learned old critic; final
  comparison retains C0 and D7 with declared deployment modes and a withheld
  population. Exact finite counts are chosen from measured A/B throughput,
  not invented here. A screen win or lower PPO loss is not promotion evidence.

Required fixtures should include identical policy draws with ACK/rejection/
unknown-delivery outcomes (same stored behavior mass), rejected request then
executor fallback without a second fake actor draw, clean truncation versus
terminal masks, death followed by faction win, immutable earlier packets
after later receipts, padding/permutation, and actor/value isolation. A real
completed-game rollout must establish schema/controller compatibility before
scaling. No source was implemented and no local tests/model computation were
run for this review. The GPU compatibility/resource change is operational;
its new allocations are not evidence for this learner's feasibility or wins.

### Phase B controller contract09 — accepted for implementation

The Reviewer read `09_controller_contract.md` in full and accepts the new
isolated Phase B implementation against its stated interfaces. This is source
scope acceptance, not yet H2 readiness or live collection approval. No Phase A
model winner is presumed, and the frozen A source remains unchanged.

The contract now addresses the actual earlier capacity/transport defects:
exact session/seat/day discussion ownership; recursively typed pending state
and immutable before-packets; three accepted/four attempted shared chat bounds
and one accepted buddy message; idempotent ACKs; unknown-delivery locks and
exact new own-message reconciliation without blind retries; monotonic aging
of server time; an isolated SDK overlay that ingests result payloads and the
same acting transport's authorized final transcript before voting; bounded
source-preserving per-player memory; and an actual generated teacher preference
rather than treating slate position0 as a recommendation. REQUEST-MDP logs
remain distinct from demonstrations and executed effects.

Two implementation interpretations must stay explicit:

- A local phase check cannot atomically prevent a server phase change between
  checking and the RPC. Revalidate latest-known owner/phase/deadline before
  sending, rely on authoritative transport acceptance/rejection, and record a
  raced attempted request accurately. Do not promise a hard phase/send atomic
  guarantee or a preemptive model deadline where only soft checks exist.
- Preserve the actual teacher preference when a controller gate rejects,
  overrides or cannot deliver it. Record preferred request, gate and executed
  effect separately. BC eligibility may select valid teacher demonstrations,
  with that selection documented; it must not silently rewrite teacher labels
  or become ACK-conditioned deletion of later sampled PPO requests.

The specified H2 checks are appropriate: malicious nested/owner/private data,
ACK/unknown/quota paths, clock/queue/phase races, the actual hooked runner's
final-message transition and unchanged legacy path, over-budget long history,
protected facts/channel separation and preference-order mapping. Before live
collection the manifest still needs exact slot/token caps, clock settings,
SDK base/overlay hashes and real shared-service burst measurements. A memory
projection retaining fewer raw claims is a bounded observation, not complete
history. No local execution was performed for this contract review.

### Frozen A CPU evidence and initial attributed C primitives

Actual Slurm evidence in `outputs/strategy-round04/checks-2186290/` was
inspected: result metadata pins75a6cd625def57bb039cd61ad3aae0df6267dcfe;
interface16 and backend11 methods both report OK. This establishes those
unit/adapter boundaries on H2, with synthetic HF weights only. It does not
establish real checkpoint generation, gameplay or strength. GPU output is
still pending at this review.

The initial `vendor/slot_attention.py`, `event_ppo.py`, source manifest,
retained MIT license and seven focused checks were statically reviewed.
One-seat masked attention, PPO clipping, clipped-value objective and separate
bootstrap/trace GAE equations match the conceptual10 route. The license text
matches the inspected SLA license and the reuse manifest identifies source
commits/ranges. These small attributed primitives are not the complete
contextual learner or a source-acceptance shortcut for the eventual11 contract.

One concrete repair was requested before their later source gate:
`exact_categorical_kl` checks off-mask zeros but does not validate finite,
nonnegative normalized probability masses or a boolean nonempty mask. It can
return a finite number for invalid categorical inputs, undermining a trust-
bound decision. Validate full simplexes with explicit floating-point sum
accuracy and add malformed/negative/nonfinite/empty-mask fixtures. A genuinely
valid old-positive mass mapping to new zero should still yield infinity.
The attention padding fixtures currently cover finite padded values; reject
nonfinite valid slot values at serialization and zero masked slots explicitly
if asserting immunity to arbitrary padded values. No local tests or model
computations were run. B implementation is incomplete and not live-accepted;
concrete C11 was not yet present at this inspection.

### Concrete C11 source architecture and primitive repair review

The KL repair was re-inspected: boolean/nonempty legal masks, finite
nonnegative masses, off-mask zeros and explicitly tolerance-normalized sums
are now checked. Valid support loss produces infinity for its affected row.
Attention explicitly zeroes masked padded entries, including NaNs, and rejects
nonfinite real slots. Added adversarial checks cover these boundaries. The
static primitive blocker is cleared; all eight focused methods still require
actual H2 execution.

`11_learning_contract.md` was read fully. Its numeric dimensions were counted
independently: global27, player17, fact13, source12, own-ACK29 and candidate/
pending32 match their enumerated fields. Typed relation pointers, separate
actor/value parameters, owner-visible inputs, replayed old values, distinct
bootstrap/trace masks, delayed death-to-faction credit, whole-state rollback
and the game-sum actor versus per-game decision-mean value reductions are
explicit. Natural role weighting and role-standardized evaluation are kept
separate. This accepts the C source architecture, subject to the following
interface/binding clarification before its relevant implementation boundary.

Reconcile its draft API with accepted09: constructor owns mode/owner RNG;
`choose(packet,candidates,deadline=...,decision_id=...)` supports the B seam
with optional decision ID, returns09's `log_probability` and other required
keys, and may add the proposed metadata/full-vector aliases. Full logits are
requested in11 prose but need an explicit output field. The exact B typed
protected-ACK/ballot/pending schema mapping must be agreed before collector
code consumes it; a guessed nested field is not an accepted binding. No live
collection is accepted by this review: source identity, memory projection,
measured populations, minibatches, finite budgets and operation checks remain
to be fixed before gameplay. Parent/Planner were informed of these conditions.

A separately labeled two-native-prompt-per-checkpoint CPU performance probe
may diagnose queue-independent feasibility. A different90-second/16-thread
CPU configuration is not a substitute for the matched GPU30-second screen
or evidence of strategic superiority. No local tests/inference were run.

### Initial B implementation and contextual C runtime inspection

Statically inspected the new memory/dialogue/candidate/policy/transport/entry
modules, SDK builder and controller checks, plus contextual slots/network/
selector. No local execution. The pipeline preserves sampled requests across
late executor fallback, keeps raw teacher preference distinct from BC
eligibility, binds canonical delivery to its before-packet, and uses the
acting transport for transcript refresh. C's API is now09-compatible. Relative
text entities are bound back to player records through typed relations rather
than absolute seat embeddings. No other-seat private-input path was found in
these inspected modules. Source remains under repair, not live-accepted.

Concrete issues sent to Worker/parent before freeze:

- `apply_delivery` rejects accepted-to-rejected receipt conflicts but silently
  accepts the reverse rejected-to-accepted conflict. Final contradictory
  receipts need symmetric explicit handling rather than counter mutation.
- B caps64 raw accepted-message records before C's semantic deduplication;
  updated11 protects64 semantic entries, and one message may contain multiple
  assertions. Align the projection/capacity contract while preserving the
  complete raw journal and explicit overflow behavior.
- Relevance checks only self, verified-fact subjects and prior commitments.
  An ordinary villager's new message about a different legal but unverified
  vote target does not trigger the promised response. Match declared legal-
  target relevance and normalized Player aliases, or explicitly amend the
  heuristic scope before collection.
- Nine B controller checks do not yet exercise the actual admissibility path
  through opening/response/changed commitment and shared private cap; several
  directly call memory.prepare, bypassing those gates. Add focused actual-
  controller checks alongside the three isolated real-runner transport cases.
- C currently classifies singleton by strictly positive floating-point masses.
  Extreme finite logits can underflow FP32 softmax to one positive mass while
  several candidates remain available. Determine forced-singleton status from
  the legal candidate mask; preserve the actual representable sampling
  distribution, and test that underflow is not mislabeled as an engine-forced
  decision. This is accounting/eligibility correctness, not a claim that normal
  initial weights already exhibit that numerical extreme.

These are source repairs/verification conditions, not reasons to block an
initial isolated H2 CPU-check run or the separately frozen A diagnostics.

### Contextual PPO core and B repair reinspection

The new `contextual_ppo.py` and five synthetic optimizer/replay checks were
read. The core retains rejected/unknown sampled requests, replays the recorded
uniform draw, validates old full categorical masses and behavior values before
optimization, freezes GAE through epochs and accumulates the complete declared
game-weighted gradient across packing chunks. Independent optimizers and
transactional rollback cover actor and critic; the fixtures include populated
Adam-state corruption recovery and singleton-only batches not moving an actor
through old momentum. No static blocker was found to the isolated H2 suites.
This is core acceptance for verification, not a live trajectory adapter.

Raw platform terminal/source validation, chronological request linkage, batch
freshness and consumption remain adapter requirements. A truthy core
`terminal_provenance` field is not platform evidence by itself, and assigning
a new caller checkpoint label is not proof that games were freshly sampled.
Those validations must precede actual collection/training. One denominator
clarification was sent: code averages actor terms over games containing a
nonforced actor request;11's wording excludes only no-request games. Declare
the intended actor-eligible-game denominator explicitly before collection,
including the effect of singleton-only games, instead of changing it silently.

B repairs were inspected: contradictory receipt handling is now symmetric;
relevance includes legal targets and normalized Player aliases; protected
memory compacts semantic first/latest records in original ACK chronology,
retaining the complete raw journal. The70-duplicate and A→B→A commitment
fixture targets the chronology pitfall. C singleton classification now uses
the available mask rather than positive floating-point support. Actual H2
results remain pending; no local checks/inference were executed. Additional
actual-controller changed-commitment/private-cap paths remain a useful live-
readiness condition alongside the prepared real-runner transport fixtures.

### Actual B/C verification attempt2 — source/core accepted

Inspected `checks-2187511`: controller12 methods OK, isolated actual SDK
runner3 cases passed, primitive8 methods OK, selector6 methods OK and PPO7
methods OK. These are36 checks/cases on H2, not36 games. The added controller
paths cover response/changed commitment and private/shared quota behavior.
The actor denominator now includes all games with requests, leaving a
singleton-only game's actor contribution zero, matching11. Checkpoint
roundtrip/tamper and weighting fixtures are present. No local tests were run.

The SDK builder now copies an explicit17-module source allowlist plus optional
fixed root metadata, rejects selected symlinks, excludes extraneous runtime/
configuration directories, and records original/modified per-file hashes.
The fresh-interpreter transport check inherits required job dependencies
after overlay/project paths, repairing the first attempt's missing dependency.
The failed initial attempt remains distinct from the successful repair.

`contextual_checkpoint.py` and the Slurm-only training CLI were inspected:
strict architecture/schema/fixed filenames, safetensors actor/value tensors,
manifest identity and tensor/report/optional-optimizer checksums, weights-only
optimizer loading and immutable new output directories are coherent. The CLI
requires declared lineage and a verified-episodes schema rather than deriving
labels itself. The actual fresh-journal verification adapter remains required;
a schema label alone does not prove real terminal or source provenance.

Read-only integrity comparison against `source-phase-bc-attempt2.json` found
73/74 local files match; current `scripts/bench_strategy_models.py` differs.
Parent was informed to record its new source boundary. This does not rewrite
the frozen remote verification evidence or invalidate tested modules, which
match. No additional model/check rerun was requested solely for that drift.

The source/core and its H2 verification are accepted, **not live collection or
competition-strength completion**. Runtime factory/verified fresh-journal
adapter, actual BC/collection and real-model synchronized burst readiness are
still missing. The separately recorded16-thread CPU probe had four of four
native calls rejected after the90-second soft limit; that configuration is
unusable for this controller, not evidence against GPU feasibility or either
model's strategic quality. The GPU screen remains pending.

### Runtime, data preparation and BC integration prefreeze review

Read contracts12/13 and runtime entry, fresh data adapter and BC core/CLI.
The strict eight-field lineage binds generator settings and source identity;
backend provenance is checked before answer parsing; owner reopening requires
identical runtime metadata. Terminal audit remains outside actor inputs.
Sampled request retention is independent of ACK, while teacher BC confirmation
requires its exact preferred request and receipt. BC uses game-disjoint
teacher development cross entropy, separate training-only value fitting and
matching selected-epoch optimizer states. These are appropriate integration
choices, not yet proof of platform learning or live readiness.

Concrete issues sent before the next freeze:

1. Runtime's required source-file set omits actively imported
   `vendor/slot_attention.py`. Enforce that dependency in the manifest
   validator, rather than relying on the current full manifest incidentally
   including it; add an omitted-active-file fixture.
2. Population version rules disagree: trainer permits population-only changes
   between batches, while deployment requires full checkpoint/runtime lineage
   equality. Either explicitly freeze the population for this whole lineage,
   or bind a population-only runtime transition to the new protocol without
   weakening the other seven checkpoint lineage fields.
3. Greedy evaluation preparation currently counts requests without validating
   its recorded checkpoint/mode/index/serialized snapshot/generator as the
   sampled path does. Add equivalent identity validation before actual final
   evaluation; evaluation logs need integrity even without gradient fitting.
4. The trainer's `verify_sources(prepared)` checks protocol/registry/journal/
   runtime-config receipts, not current source-manifest bytes and enumerated
   implementation files. Contract13's immediate prefit source-reverification
   claim is therefore stronger than the current code. Bind an actual training
   source manifest and verify portable current project files before fitting;
   do not demand expired node-local model paths merely because they appeared
   in an archived runtime config. Collection-time factory checks alone do not
   pin subsequently imported trainer implementation.

These are concrete source/contract repairs, not a request for arbitrary live
counts or a reason to skip an isolated check run. No local model computation,
training or test execution occurred. Actual fresh collection/BC/PPO/evaluation
budgets and shared-model burst readiness remain pending.

### Integration attempt3 source gate — accepted for H2 verification

All four prefreeze findings were re-inspected after repair. Runtime now
requires the complete small policy package in its source manifest, including
lazy/vendor dependencies; an omitted attention module is a rejection fixture.
Deployment allows only population-version changes between declared batches,
while preserving the exact checkpoint and the other seven lineage fields;
new-population acceptance and controller-change rejection are explicit tests.
Learned SDK workers set one Torch thread to avoid uncontrolled CPU fan-out.

Data preparation now checks greedy checkpoint/mode/decision/argmax/slate and
generator identity. The fitter verifies archived manifest receipts plus actual
portable current implementation bytes before importing a batch into fitting;
expired node-local model paths are not treated as required source files.
Owner/session/seat/decision-derived generation seeds match runtime and replay.
PPO rollback reports preserve rejection but sanitize infinite metric fields
for JSON serialization. BC's ACK confirmation remains limited to imitation.

Source3 is statically accepted for the complete isolated H2 verification
suites. No further source blocker was identified or theoretical gate added.
Actual suite results and real-model shared-service readiness are still pending;
this is not acceptance of live collection, completed learning, improved play
or tournament deployment. No local tests/inference/training were performed.

### Attempt3 JSON replay failure and narrow repair

Parent reported actual H2 job2189527: six suites passed (controller12,
actual runner4, primitive8, selector6, PPO8, entry7); data passed five of six
including the synthetic two-epoch BC fit, but its file-backed registry fixture
produced no eligible episode. These logs were not yet local at the expected
path during this review, so that execution status is parent-attested here.

Static inspection confirms the identified integration defect and repair:
serializer graph edges were Python tuples, but JSON journals reload them as
lists, causing exact full-snapshot replay equality to reject otherwise
identical data. Edges now originate as JSON lists. The selector fixture
explicitly asserts full snapshot equality across JSON roundtrip; the
file-backed data fixture now prints exclusion reasons on failure. This changes
representation, not graph values or actor tensors. The narrow repair and full
H2 attempt4 verification are accepted without repeating the conceptual gate.
The failed attempt remains evidence of a real serialization defect, not a
successful end-to-end registry test. No local computation was performed.

### Final implemented-pipeline acceptance — actual attempt4 evidence

Independent read-only comparison found **all81 files match**
`source-phase-bc-attempt4.json`, with zero local mismatches. The verification
summary pins archive SHA256
`7fd2ee5b449692738af9db244346a158275b3cb7df15ed79b7acfeb1ab7bc41b`
and records actual H2 job2189749, COMPLETED, exit0:0, elapsed47seconds.
Every actual suite log was inspected: controller12; actual isolated SDK
runner4; attention/PPO primitives8; contextual selector6; PPO/checkpoint8;
pinned runtime entry7; data/BC6. **All51 cases pass.** The file-backed registry
and synthetic two-epoch BC fit now pass, rather than only in-memory fixtures.

The prior actual2189527 logs were also inspected after transfer: its data
fixture failed0-versus1 eligible completed games, while five other data methods
and the six other suites passed. This preserves the failed JSON replay attempt
and distinguishes it from the successful representation repair. No additional
rerun is required merely to repeat already-passing evidence.

The **implemented and tested B/C pipeline is accepted at this correctness
scope**: owned reactive transport/memory, source-pinned runtime, relational
actor/critic, safe checkpoint/replay, request-preserving PPO, confirmed-teacher
BC, and file-backed collection accounting. These are synthetic CPU correctness
and optimizer checks on H2, not real language-model playing experiments.

The overall competition-winning objective remains **unproven and unfinished**.
No actual new platform teacher collection, live PPO training or prospective
strength comparison has completed in this evidence. Real GPU generation and
synchronized service readiness remain pending; the four90-second CPU native
calls failed their deadline. No incumbent promotion/replacement or tournament
readiness claim is accepted. D7 remains the retained incumbent. The next
empirical steps are real-model readiness, finite prospective collection and
multiple fresh learning batches, followed by isolated strength evaluation.
Only file-integrity verification and source/log inspection ran locally; no
local model inference, training, gameplay or test execution occurred.
