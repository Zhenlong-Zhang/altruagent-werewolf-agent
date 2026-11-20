# Phase B: owned dialogue, bounded memory and transcript transport

Worker contract, 2026-11-20. **Parent and Reviewer accepted Phase B implementation; runtime verification and live collection remain separate gates.** Phase A remote source `75a6cd625def57bb039cd61ad3aae0df6267dcfe` and its model-screen protocol remain immutable. This contract is independent of which screened checkpoint is selected. It adds capabilities, not a promise of stronger play. Later learning is owned by the parent under a separately accepted Phase C contract.

## 1. Isolation and ownership

Worker owns new `strategy_memory.py`, `strategy_dialogue.py`, `strategy_candidates_v2.py`, `strategy_policy.py`, `strategy_entry.py`, focused controller/transport checks, and an isolated SDK overlay. Parent owns backend/model-service/harness changes and the later `contextual_selector.py`/`train_contextual_ppo.py`. No global SDK installation, old remote runtime, D7 artifact, user WORKPLAN or user image is modified.

The base owner is `(session_hash, acting_seat)`. A discussion owner is `(session_hash, acting_seat, day, window_kind='day_vote_discussion')`; the official game has one discussion window per day. Actor seat/actual role come from the assigned SDK context and own observation, not an LLM assertion. Different owners cannot share an in-memory ledger, journal or pending delivery. Use one exclusive writer per owner; reject concurrent/conflicting ownership rather than merge agents. The frozen-model service retains no seat memory.

Full authorized raw envelopes and controller events are append-only, owner-private records (directory0700/files0600). Preserve exact sender, sequence, recipients, type, content and source receipt metadata. Reject private envelopes unless this actor is sender or recipient. Sequence gaps do not imply missing messages: unseen other-seat private messages may use global sequence numbers. Conflicting payloads for the same visible sequence are an error. Cross-window transcripts remain available, but old pending/quota records never become the new window's counters.

## 2. Typed mutable state; immutable decision snapshots

`StrategyMemory(owner, journal_path)` provides `observe_state(state, context, observed_at)`, `merge_visible(envelopes, source, complete, observed_at)`, `apply_delivery(delivery_id, result, context, observed_at)`, and `compile_packet(state, context, token_counter, input_budget)`. Source callbacks are trusted transport events, not model commands. Every method validates owner/window and scalar/container types recursively.

Active window state contains only: owner, server-window identity, accepted chat count, attempted chat count, accepted private-buddy count, accepted message sequence/delivery-ID sets, last consumed relevant-event ID, stage, known termination status, pending record orNone, server remaining quota, last observed deadline/seconds-left and monotonic receipt time. Stage is one of `opening`, `response`, `commitment`, `terminated`; stages are not extra chats outside the shared quota. Unknown prior-window records remain in the raw journal with their original owner.

A pending record has exactly:

`{owner, delivery_id, decision_id, candidate_id, intent, target, recipients, canonical_text, assertions, assertion_realizations, packet_hash, send_floor_seq, attempted_at, status}`.

All fields have explicit types: seats0..6 orNone, recipients empty/single legal buddy, canonical text string within platform limits, finite monotonic time, known status enum (`prepared`, `unknown`, `accepted`, `rejected`), canonical assertion objects and realization markers already validated against the immutable before-packet. No arbitrary nested dictionaries, reward/winner/evaluation fields, model memory or unvalidated free prose enter this structure. Attribution clarification records do not become asserted propositions. Pending snapshots are persisted before the RPC; later receipts append events and create a new state version without rewriting the logged before-packet or sampled request.

Each decision record stores exact before-packet/candidates, owner/window IDs and hashes. ACK-driven state changes are future observations, not corrections to historical policy inputs. Raw generator reasoning is internal audit material, never public text or an authoritative fact.

## 3. Finite communication policy

Initial bounds: **at most3 accepted chat messages and4 attempted chat RPCs per window**, public/private combined; at most one accepted private coordination chat. Honor a smaller server `remaining.this_turn` immediately. A rejected attempt counts toward4 but not3. An unknown attempt counts toward4 and locks further chat until reconciliation; no blind resend. Terminate is not a chat and does not consume the platform's quota. Termination is idempotent, and the overlay suppresses redundant terminate RPCs once this seat's termination is known.

Opening may be a public strategy message or a private message to a living engine-confirmed wolf buddy. There is no hard requirement that an arbitrarily ordered private candidate outrank a model's public preference. Subsequent chat requires a **new relevant event after the last decision's consumed cursor**: a new claim/source reference mentioning this actor or a legal target, new authoritative evidence, a direct buddy response, or a changed feasible vote commitment. Repeat identical source/text/recipient/tactic does not create a new event. Repetition alone cannot rewrite facts or count as new independent evidence. Event detection is a logged deterministic relevance heuristic, not a claim of semantic entailment.

Allow one evidence-triggered response and, if still within caps, one materially changed-target commitment. Record why a commitment changed; do not force an obsolete target after new facts. No useful update, known quota exhaustion, unavailable budget, invalid state or unresolved pending delivery means terminate. A selected Pass also terminates. Return only `SendMessage` or `TERMINATE_MESSAGING`; no `None`/invented WAIT. The existing runner waits after a successful chat. After known termination, the overlay waits/ingests updates until phase transition without requesting new chat decisions.

Count accepted messages once by delivery ID and reconcile their actual sequence when available. A duplicate callback or previously observed message does not create another accepted chat, tactic or training decision. Known rejection does not record a public commitment. An accepted ACK may confirm the exact pending canonical message even if its envelope is absent; later transcript reconciliation binds its sequence without a second count.

## 4. Unknown delivery and reconciliation

The isolated runner passes an explicit locally assigned `delivery_id` with every result. A positive `accepted=True` without an error confirms that pending request; an explicit protocol rejection marks rejected. Missing/ambiguous results, disconnects and unknown delivery errors remain `unknown`.

Before any further chat, refresh this seat's authorized transcript. A candidate accepted message must match **sender=self, exact canonical text, exact recipients and sequence>send_floor_seq**. Match exactly one previously unassigned message to exactly one pending request; then mark accepted idempotently. An old matching message at/below the send floor is not evidence for the new request. Multiple matches/conflicting receipts remain ambiguous, not multiple accepted intents. Absence from one refresh is not proof of rejection, so never resend merely because it was not found. If still unknown, terminate and preserve that uncertainty; later snapshots/ACKs may resolve the original window only.

For Phase C, log the sampled **request** with its original probability, plus the executor's delivery/fallback observations. A rejected/unknown request is not silently relabeled a deterministic fallback action. Unknown delivery alone need not censor a known-terminal game if the request/log-probability and replayable authorized next observations are intact (REQUEST-MDP). Missing actor input, irreconcilable ownership, absent propensity or unreplayable transitions are explicit ineligibility reasons. Raw ambiguous episodes remain archived.

## 5. Phase clock and queue budget

Use the server window deadline or conservative `seconds_left`, with the monotonic time at which that snapshot was received. Subtract elapsed local time on every decision; repeated use of a stale120-second value cannot reset the window. Queue time, serialization/token counting, model generation, validation and delivery share the same deadline. The initial minimum delivery reserve is3seconds; parent may increase it from operation-only RPC/timing measurements before collection. Task-specific model/selector caps are frozen after Phase A throughput, not tuned on win outcomes.

Do not call the model when remaining time cannot cover the frozen cap/reserve. Use a validated cheap canonical option if there is sufficient delivery time and it supplies a nonduplicate useful request; otherwise terminate. Results that arrive late are rejected, not sent after phase flip. These are checked soft budgets unless actual preemption is implemented; record full elapsed time and overshoot. A new game day creates a new clock/window, never revives an expired old request.

The pre-send gate checks the latest known owner/window/phase and the aged deadline immediately before the RPC. It cannot atomically prevent the server from flipping phase while that RPC is in flight. Preserve the original attempted request if this race occurs and record the authoritative rejection/unknown result; do not claim an atomic phase guarantee.

## 6. Isolated authorized-transcript overlay

Create a private versioned SDK copy under `runtime/round04_sdk/altruagent/` (or equivalent parent-approved overlay), retaining upstream notices and a manifest of original/modified file hashes. Activate it only for the new strategy entry with an explicit environment/PYTHONPATH in its isolated job. The shared official/starter package and Phase A/R03 running copies remain untouched.

Opt-in policy hooks:

- `requires_visible_transcript_flush = True`.
- `on_visible_update(state, context, envelopes, source, complete, observed_at)`: merge authorized new state/result/refresh envelopes and clock observations; no model call.
- `before_message_request(delivery_id, decision, context, observed_at)`: bind/persist a prepared request before the actual transport call.
- `on_delivery_result(delivery_id, result, context, observed_at)`: update the typed pending state and merge result envelopes **before** selecting the next get-state/wait branch.

Before `choose_action` for `day_vote`, the overlay calls the **same acting transport's** `game.get_messages(since=highest_visible_seq)`; initially use -1 to establish history, then exclusive sequence cursors. Merge returned messages with the last state and any terminate-result payload using sequence deduplication. The refresh cannot access another seat's transport or global hidden game state. A terminate result reporting `phase=moving` is ingested before `get_state`, so its final chat survives even when the vote state's `new_messages=[]`.

A refresh error/incomplete transport result marks `transcript_complete=False` and is logged. It must not silently advertise complete memory. The policy can use the authoritative current observation and a declared canonical fallback; no fabricated recovered claim is inserted. Capture the exact completeness/error indicator in the before-packet. Live unknown-delivery reconciliation uses the same authorized refresh hook, not a separate credentialed client. Policies without the opt-in flag retain the legacy runner path.

## 7. Bounded model view, persistent exact ledger

Phase A's all-text `attributed_claims` view is not a long-game memory solution. B introduces `strategy-observation-v2` and a versioned deterministic projection; never send the whole raw ledger repeatedly until the8192-token backend rejects it.

Reserve protected slots for own identity/role, all verified facts (capacity32), seven player records, full engine legal actions and typed active-window state. Protect current own claims/commitment and separately tagged bluff records. Contradictory trusted facts or protected-capacity overflow produce an explicit unsupported-state diagnostic and fallback, not silent deletion.

The protected own ledger has capacity64 distinct typed semantic entries, keyed by `(assertion, recipients, intent)`; vote commitments use a typed synthetic `vote_commitment` entry. Repeated identical semantics retain their original and latest ACK metadata (at most128 projected source records), while the private raw journal retains every message receipt. Assertion-free rhetoric is not a protected factual claim. Records retain owner/day, delivery ID, accepted sequence and original before-packet hash. This matches the Phase C semantic deduplication key; two different assertions in one accepted message are two semantic entries. Overflow is explicit and never silently slices away old commitments or bluffs.

For each player retain up to4 attributed source slots: two latest unique messages, one possible-conflict/fact-reference slot, and one older relevant claim/commitment reference. Tie-break by sequence deterministically and deduplicate source IDs across slots. A “possible conflict” flag is uncertain retrieval metadata, not verified logical contradiction. Use exact bounded source excerpts with channel, sender, sequence, mention pointers and a full-raw-record hash; do not turn a negated/quoted mention into an affirmative summary. No model-written summary is promoted to a fact. Private excerpts remain visible only to their owner and are never automatically publicly quoted.

Provide at most6 recent unique dialogue excerpts, deduplicated against persistent slots. Use a frozen per-slot token cap and exact native-template input-token counter; parent backend may expose a counting-only request (no model inference/state) or a local pinned tokenizer may count it. Protected fields are admitted first, then deterministic relevant slots until the complete request fits the configured input budget including system prompt. Record retained source IDs, excerpt/truncation flags, dropped counts and selection policy version. If protected fields alone cannot fit, fail explicitly. Raw evidence remains in the private append-only journal; the bounded prompt is not described as complete-history equivalence.

Before live collection, freeze exact slot/token caps and tie-breaks after the operation-only long-history check. The Phase C network may encode these slots separately; neither native inference nor learned inference may reconstruct an unlogged, richer context during replay.

## 8. Actual teacher preference and later selector interface

New schema/generator identity `strategy-candidates-v2` adds speech response `{candidates:[three typed plans], preferred_index:int}`. Validate preferred_index is an integer0..2 and points to a validated generated candidate. Obtain it in the **same native completion**, with no additional teacher call and no assumption that candidate0 is best. Preserve Phase A's strict v1 parser/validator and frozen remote source; B uses a separate adapter. Nonlanguage teacher responses already supply the actual `action_id`, while all engine-legal candidates remain available.

Teacher mode uses the actual validated preference when transport/controller permits that request. Log `teacher_choice`, `teacher_source` (actual generation or explicitly named heuristic fallback), and the executed/gated result separately. Never record a rejected, overridden, timed-out or reordered first plan as an actual teacher demonstration. Teacher-generation demonstrations are BC data, not fake PPO trajectories with invented categorical probabilities. No usable preferred_index means explicit format error/fallback or a separately labeled heuristic choice.

Keep the actual preferred teacher request in the raw log even when the controller rejects or overrides it. BC eligibility/target and executed choice are separate fields; a gated request is not relabeled as a different teacher preference. Do not transfer a BC eligibility filter to later PPO REQUEST-MDP trajectories, which retain original sampled requests and their propensities.

Proposed Phase C selector seam: `selector.choose(packet, candidates, deadline=...) -> {index, probabilities, log_probability, value, checkpoint_id, metadata}`. Parent owns its final implementation. In learned mode validate shape/sum/finite values and selected log-probability, preserve the full candidate order and log the behavior critic value before transport. Mode/config/serializer/generator identities are explicit. Transport gating does not rewrite the original requested policy distribution; deterministic fallback/executor records remain separate. Pass and canonical fallback are available candidates, while padded indices are never legal choices.

## 9. Required Hoffman2 verification and acceptance boundary

No local inference/tests/gameplay. Before live B collection, run:

1. Cross-session/seat/day/window injection; unauthorized private envelope; malformed nested pending/assertion/status/time payload; stale state/reconnect and exclusive owner writer.
2. Three-ACK/four-attempt and shared private/public quota caps; accepted private coordination cap; duplicate ACK/old matching message cannot increment; known rejection leaves no public commitment; unknown delivery matches only exact new own message; no match/multiple matches never trigger blind retry.
3. Opening→relevant new claim→response→changed commitment→terminate; duplicate or unrelated update does not manufacture new evidence. Missing/dead/suspected buddy cannot receive P2P. No fake WAIT/None return.
4. Stale-clock aging, UDS queue delay, reserve exhaustion, phase flip and late completion; preserve actual deadline overshoot/fallback diagnostics.
5. Isolated SDK public `run_match` fixture: terminate result contains final opponent chat and `phase=moving`, next state has empty new_messages, final vote packet still contains that authorized message exactly once. Refresh failures remain visible; no other-seat client is queried. Legacy no-hook path unchanged.
6. Adversarial long history beyond input budget: every verified fact survives, per-player relevant older source slots survive declared retention, private/public channels remain distinct, untrusted negation/quotes never become facts, overflow/truncation is explicit, and the complete native prompt is counted before generation.
7. Actual teacher preferred_index selects the correct validated plan after a nontrivial permutation; invalid/missing/Boolean index rejects; actual action_id preference maps exact IDs. BC records do not fabricate PPO probabilities. Learned-selector stub reproduces logged request probabilities/value, and rejection/fallback leaves those immutable.

Then measure actual shared-service operation with the frozen selected model and controller under realistic concurrent seat requests, while recording cold/warm cost, queue/phase deadlines and fallback/ACK behavior. This is readiness, not a win-rate claim. Parent sets the finite first fresh collection budget before game outcomes; any next policy/source change creates a new lineage. Phase C cannot start by retrofitting the old136 coarse-state choices.

## Status

- Reviewer and parent accepted Phase B implementation after the race/teacher-log clarifications above; this is not live-collection acceptance. New modules and isolated SDK overlay are written; static review and Hoffman2 execution remain pending.
- Phase A remote and model-screen results remain untouched; no model winner is assumed here.
- Source/API and fixed memory/clock constants must be recorded in the isolated B manifest before gameplay. Later learning/evaluation needs its separately accepted contract and fresh data.
