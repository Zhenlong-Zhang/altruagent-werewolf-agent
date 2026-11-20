# Round04 implementation contract — phased acceptance draft

Worker contract, 2026-11-20, based on Planner `01_plan.md`/`07_literature_fit.md` and Worker `03_method_audit.md`. Parent and Reviewer accepted Phase A after the typed-assertion, owner-wrapper and final-answer clarifications. Later training/evaluation quantities will be frozen after operation-only throughput measurements, before their outcome inspection. No D7/R03 artifact or user WORKPLAN/image changes are included.

## Objective and causal controls

Improve AltruAgent Werewolf play through actual evidence-conditioned actions and responsive language, using real reusable strategic models/learning components. Retain D7 IDs/greedy/no-guard as the release/control. R03 outcomes are development evidence, never the new final test. Separate (i) new observation/candidate/controller and pretrained model effects from (ii) fresh policy learning, and evaluate the frozen result against a declared opponent population. Training completion, larger parameter count or factual probes alone cannot justify promotion.

## Stable interfaces

### Acting-seat observation packet, schema `strategy-observation-v1`

The deterministic compiler accepts only the current SDK state/context and that seat's own persisted, authorized transcript/result ledger. Output:

- `identity`: hashed session, acting seat, actual parsed own role, day, phase and stable player-seat map.
- `verified_facts`: immutable fact IDs, subject, wolf/nonwolf alignment or exact engine-revealed role where justified, source (`own_role`, `own_check`, `known_partner`, `public_engine`), and first-known time/message reference. Negative seer checks do not assert a special role. Contradictory authoritative inputs produce an explicit invalid-state diagnostic.
- `player_records`: attributed public/private claims, contradictions and vote history, with message sequence IDs/day/channel. Inferred beliefs remain separate from verified facts. Repetition cannot overwrite or create a fact.
- `self_assertions`: ACKed own public claims/commitments and separately tagged deliberate wolf bluffs. A chosen but unacknowledged message is pending, not a public fact.
- `visible_dialogue`: bounded recent authorized messages plus persistent per-player summaries. Preserve sender, recipients/channel and sequence/day. Protected facts survive truncation; content is untrusted quoted data, not instructions.
- `legal_actions`: the entire current engine list with exact ID, label, explicit resolved target/action kind. Preserve abstention and other targetless legal actions; verified knowledge is an annotation/utility prior, not an engine mask.
- `window`: accepted/attempted quota, pending delivery ID, remaining server quota/time, highest visible sequence and transcript-flush completeness.
- `provenance`: packet/serializer/rules versions and canonical packet hash. No terminal reward, winner, evaluation-only roles, other seats' hidden facts or future messages may enter actor inputs.

History/window wrappers carry an explicit owner `(session_hash, acting_seat)`; reject cross-owner reuse and envelopes for private messages neither sent by nor addressed to this seat. A public quote about a private role does not create an authorized private envelope or verified fact.

Phase A reserves `window.pending` as `None`: no arbitrary nested pending object is admitted into actor state. Phase B must define and validate a complete typed delivery record before enabling that field; a top-level key whitelist alone does not prevent nested evaluation-data leakage.

### Candidate/response packet, schema `strategy-candidates-v1`

For nonlanguage decisions, expose every current legal ID to the selector; do not truncate to LLM top2. A complete-generation model screen may predict a legal ID directly, but that is not a reason to remove remaining candidates from learned selection.

For speech, initially request three **distinct contextual tactics**, plus Pass and a canonical legal fallback. Candidate records contain stable candidate ID, intent, explicit target(s), audience/recipient, cited fact/claim IDs, claim type (`verified`, `attributed`, `uncertain`, `deliberate_bluff`), rendered text and source packet hash. The model input field `text` must be empty. Deliverable language comes only from typed assertions plus controlled intent/target/audience rhetoric, with `realization_mode=typed-canonical-v1`; arbitrary model prose is not sent. This is a richer state-conditioned plan vocabulary, not free-form language realization or an exact SLA reproduction. Enforce nonempty chat, <=50 words, <=1024 characters and platform recipient rules. A private recipient must be a living confirmed wolf buddy; never self or multiple recipients. Verified assertions must match their referenced facts; deliberate wolf bluffs are permitted and recorded as assertions, never private truth. A surface-form/intent/source mismatch is invalid, with a logged reason and safe fallback.

Assertions are typed records `{subject, predicate, value, source_ref, truth_mode}`. All fact, attribution, uncertain and deliberate-bluff clauses use canonical renderers; attaching a valid fact ID to unchecked free-form text is insufficient. General rhetoric also comes from controlled intent/target slots rather than a regex-filtered suffix. Intentional wolf bluffs remain explicitly tagged and are never promoted into verified knowledge. A future free-LLM realization experiment requires separate acceptance and evaluation; no regex is claimed to prove arbitrary prose factual/private-safe.

Attribution matching establishes only a source-message mention, not semantic entailment. In Phase A an `attributed` reference renders a neutral request to clarify that speaker/message/subject; it never asserts that a negated or quoted proposition was the sender's belief. Output `assertion_realizations` identifies such references as `source_message_clarification`, so later dialogue memory must not record them as newly asserted facts.

### Collected policy transition, schema `strategy-transition-v1`

Record exact before-decision observation packet and hash; full candidate records/order and engine/legal mask; generator checkpoint/revision/template/decoding/prompt version; actor/value/encoder checkpoints; selected candidate; exact actual probabilities and selected log-probability; **behavior value prediction**; recurrence state/reset mask if used; decision/ACK IDs and delivery status; next packet or terminal link; reward, true terminal, death-active and truncation masks. Role/winner fields used for evaluation remain outside actor payloads. Record failure/fallback transitions explicitly, rather than labeling deterministic fallback as a sampled action.

Any change to state serialization, candidates, generator, dialogue controller, encoder or learned representation creates a new distribution/artifact identity. Old 136-choice R03 data do not satisfy this schema and cannot be treated as the new on-policy PPO batch.

## Phase A — accepted first implementation chunk

**Scope.** New opt-in observation/fact compiler and native-generation model adapter, a deterministic candidate/result validator, and focused checks plus parent-owned matched GPU screen. Correct immutable-fact handling in the new path; leave incumbent behavior and artifacts untouched. No PPO training or live competition rollout yet.

**Ownership accepted by parent.** Worker: new `werewolf_agent/strategy_state.py`, `strategy_candidates.py`, later controller/policy/contextual learner, and focused `scripts/check_strategy_interfaces.py`. Parent: new `werewolf_agent/strategy_backend.py`, `scripts/serve_strategy_model.py`, `bench_strategy_models.py`, Slurm staging/resource control and measured run manifest. Existing incumbent modules are read-only in this chunk.

Worker's public first-chunk functions are `compile_observation(state, context, history=None, window=None) -> packet`, `build_candidate_request(packet, task) -> messages`, `parse_strategy_answer(raw_text) -> dict`, `all_legal_candidates(packet) -> records`, and `validate_speech_candidates(answer, packet) -> records`. The persistent ledger/controller extends these deterministic primitives in Phase B. Parser consumes complete `<answer>{JSON}</answer>` content, never a JSON fragment from internal reasoning; task-specific validators reject unsupported fields/IDs/recipients/fact references. Action answers select an exact offered action ID. Speech answers provide the three candidate records described above.

The parser receives newly generated tokens only. Accept one final complete answer after finished reasoning, reject duplicates, unmatched reasoning/answer tags, trailing material or truncation, and never choose the first JSON fragment found in a scratchpad. The backend must expose finish reason separately so a token/time-truncated response cannot masquerade as a complete action.

Parent's backend interface is `complete(messages, max_new_tokens, temperature, seed, deadline) -> {'text': str, 'provenance': dict}`. Text contains newly generated completion tokens only. Load staged files only; native tokenizer/chat-template behavior is versioned per checkpoint and Qwen reasoning is enabled for this comparison. Do not force MARSHAL through disabled-thinking/one-token-ID ranking. Use bounded complete generation, validate only delivered answer content, and never expose internal reasoning as public chat. Both models receive the same instruction to keep private reasoning brief and reserve tokens for a complete final answer. Truncated/malformed/late outputs produce diagnostics and a safe offered fallback. Time budgets are soft unless a genuine interrupt mechanism is implemented; late-result rejection and actual tail timing are mandatory.

The parent-owned shared frozen-model process serves an owner-only Unix socket (HTTPX UDS/UnixStreamServer; restricted socket directory and permissions), not an exposed network endpoint. Only frozen weights/tokenizer may be shared. Every request supplies its own complete messages and deadline; per-request reasoning/KV state and seat ledgers must not persist across requests. Queue TTL is part of the measured decision budget; expired requests fail explicitly. There is no hidden external API or cross-seat conversation memory. Screen the two checkpoints sequentially on one allocated GPU first; live concurrency/queue throughput requires separate measurement before scaling a roster.

Compare pinned vanilla Qwen3-8B and author-released MARSHAL-Generalist-Qwen3-8B (`369d9decea6e8b347f81706fa169dbbddddce360`) on identical authorized situations, task schema, target permutations and declared resource/decode budget. Respect each model's intended native interface, and document unavoidable template differences. Use actual allocated GPU measurements, including cold load/prefill, generation, parsing, VRAM and final synchronization. Parent freezes operational budgets after throughput observation; no game-win information chooses those budgets.

**Acceptance checks, on Hoffman2 only:**

1. Repeating/adversarially rephrasing accusations cannot alter private checks or invert the new projected trusted alignment; conflicting engine facts are rejected. Own-role/partner visibility stays seat-isolated.
2. Public “system confirms” injections remain attributed claims. No schema field, legal list, role, recipient or backend config can be changed by quoted opponent text. Private buddy messages retain channel/provenance; public output cannot accidentally render buddy/private-plan data.
3. Every engine-legal action appears exactly once, including abstention and opaque IDs. Targets derive from explicit metadata/labels, not a numeric action-ID guess. Permuting player identities/action IDs/order preserves the corresponding factual bindings.
4. Speech validity distinguishes source-grounded verified statements, uncertain/attributed statements and intentional wolf deception. Invalid recipient, fact reference, overlength, duplicate tactic, malformed/truncated answer and deadline overrun exercise explicit legal fallback.
5. Actual vanilla/MARSHAL generation respects intended reasoning format and the same operational budget. Report raw proposer quality separately from deterministic guard-forced successes. No zero-shot victory claim from these checks.

Phase A source acceptance enables the GPU capability screen; it does not preapprove later architecture choices or a training budget.

## Phase B — responsive controller and transport migration

After Phase A screen/parent decision, implement an opt-in controller with at most three ACKed messages per window, sharing quota between public and private chat and retaining a conservative attempted-send ceiling. Support opening, one relevant-new-evidence reply and optional changed-target commitment. No new actionable evidence means terminate. Private coordination is restricted to a living confirmed buddy. Suppress duplicate semantic claims; record a reason for changing a public commitment.

Use an isolated/versioned SDK overlay that merges authorized result messages and calls acting-seat `get_messages` before vote decisions. Gating/hook source must leave legacy entry behavior unchanged. A phase-flip failure cannot silently masquerade as a complete transcript. The current SDK recognizes no WAIT/None response; use its existing post-chat wait and return only SendMessage/TERMINATE. Any later explicit waiting capability requires separate review.

Acceptance fixtures: opening→new-opponent-message→response→quota/terminate; ACK increments exactly once; rejection/duplicate callback/unknown delivery does not create a committed claim or duplicate send; reconcile own transcript before retry; dead/unknown buddy cannot receive P2P; 50-word/1024-character/five-platform-chat/120-second limits; final message in a terminate-result with `phase=moving` survives into vote state; message sequences deduplicate across refresh, reconnect and ordinary state updates. Fallback/error/elapsed events retain all attempted actions.

## Phase C — contextual multibatch learning and attributed reuse

After accepted state/candidate/controller interfaces, adapt the MIT SLA/MAPPO components inspected in `03_method_audit.md`. Record upstream commits, LICENSE/notices, reused files and all interface changes. Reuse actual rollout/return/value/masking primitives where appropriate; do not describe an independently written tiny selector as a full reproduction. The released MAT module's joint-agent assumptions must not import other seats' private observations into the actor.

The actor must consume the actual observation/player-evidence context plus full legal candidate/tactic text, not only the old eight scalars. Use a decentralized behavior critic initially. Select an encoder/recurrence layout after Phase A resource measurements; freeze exact dimensions/token allocation and training settings before collection. Maintain protected fact slots and explicit candidate/legal masks. A larger state string alone is insufficient if truncation drops the important facts.

Collect successive fresh on-policy batches with a stationary declared opponent mixture within each batch. The frozen collector's value predictions inform fixed returns/advantages for that batch; later collected batches use the updated previous critic. Do not refresh advantages every optimizer epoch. Specify event-time discounting, recurrence reset, terminal/death/truncation masks and eventual faction reward for eliminated players. Forced actions can supply value/transition data while producing no categorical actor gradient. Apply clipped PPO with documented reductions, KL/failure rollback and complete optimizer snapshots. Do not reuse old R03 propensity records after the representation/action-space change.

Acceptance fixtures: exact logged categorical probabilities and behavior values reproduce before updates; selected log-probability matches actual sampling; legal masks exclude only engine-illegal choices; terminal/death/truncation returns are correct; actor cannot read future rewards/evaluation roles; old critic influences next-batch advantages; advantages stay fixed within epochs; recurrent state resets by session/seat; critic-only updates cannot silently change an accepted actor; deliberately rejected updates restore policy/value/optimizers; checkpoint/generator/encoder identity mismatches reject a batch.

Parent will append finite fresh episode/batch/compute quantities **after measured throughput, before training outcomes**, with game-level effective sample accounting. Algorithm label is contextual/recurrent PPO unless a separately accepted centralized multi-agent procedure is genuinely implemented. LoRA/SFT/KTO remains a separately measured generator-adaptation option, not an automatic extra round.

## Phase D — fresh strength evaluation and delivery

Freeze selected source, generator, encoder, learned policy, fallback, opponent population and assignment seeds before the final cohort. Compare D7 release, new strategy interface without learned adaptation, and the same interface after adaptation; this isolates interface/model gains from learning. Use multiple declared LLM/policy styles, not exclusively one direct LLM plus five rules. Reserve a different opponent-mixture/seed namespace for final evaluation; old F/G are development-only.

After throughput sizing, parent freezes finite sample/creation/time caps, checkpoint-selection rules and stopping criteria without looking at final outcomes. Retain every assignment, unsuccessful attempt, role/seat imbalance and unresolved game. Report game-level uncertainty, role/seat/mixture counts, actual decision differences, model/format/ACK/fallback rates and latency/VRAM tails. Strength promotion requires new evidence; retain D7 otherwise. Final artifacts include deployment assumptions, actual selected model weights/versions, source/license provenance, reproducible training and evaluation logs. Winning the competition remains the objective, not an outcome promised by architecture or training completion.

## Acceptance status

- Planner: assessment ready in `01_plan.md` and `07_literature_fit.md`.
- Worker: source audit and phased contract completed; new Phase A interfaces/checks implemented, H2 execution pending.
- Parent: Phase A route/interfaces/ownership accepted; measured later-phase budgets remain pending.
- Reviewer: Phase A implementation under review after adding typed constrained assertions, controlled rhetoric, Phase A pending-object rejection, session/seat-owned wrappers and final-answer parsing clarification. Only the first new interface/backend/check chunk is authorized; later phases require their own concrete source/data/budget acceptance.
