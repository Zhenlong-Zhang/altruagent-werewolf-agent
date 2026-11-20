# Method audit before Round04 implementation

Worker source inspection, 2026-11-20. Baseline: `fe359e42053cce7edd194d2a6d8f9bf9ede013e3`. No implementation, model execution, local tests or new game-log analysis was performed for this audit. R03's already-reported 1/98 observed greedy-choice change motivates inspection; it does not identify a cause by itself. Read alongside Planner `01_plan.md` and `07_literature_fit.md`.

## Main finding

R03 optimized a constrained selector over a narrow, mostly fixed action vocabulary, using a representation that omitted the social evidence needed to distinguish strategies. More parameters or epochs cannot recover an omitted candidate, an unseen claim or a missing final message. The PPO calculation was valid fixed-batch terminal-return learning, but the fitted critic supplied no learned baseline to that actor update. Several actual source-level correctness paths also need repair before attributing new performance to learning.

## Source-grounded findings

### 1. Immutable evidence can be inverted in heuristic scores — correctness bug

`werewolf_agent/beliefs.py:59–62` initializes a privately checked nonwolf at -8; lines 98–110 then add 0.35 for each matching accusation, with only a final clamp at line 113. The comment at line 112 promises that weak claims cannot overpower an investigation, but the code does not enforce that invariant. Twenty-five accusations produce -8 + 25×0.35 = +0.75. Five other living players can each send five chats under `werewolf.md:275–279`; repeated claims across windows are also retained. This is a source-level counterexample, not an executed test or an assertion that it caused an R03 loss.

The separate `known_roles` map stays correct. However, the mutated score enters candidate ordering, fallback utility and speech target selection (`hybrid.py:150–157`, 161–176). Repair the representation: immutable private/public-engine facts plus separately accumulated uncertain evidence, with final hard-fact projection where appropriate. Repetition must not create additional independent evidence. Do not turn this into a blanket honesty rule for a wolf's deliberately deceptive public speech.

### 2. Discussion-to-vote can drop final visible messages — latent integration bug

`policy.py:33–40` ingests only `state.new_messages`. `hybrid.py:416–417` handles message results through `_delivery`, which ignores their transcript payload. The SDK runner at `starter/altruagent/runner.py:921–924` switches directly to `get_state` when a send/terminate result says `moving`. The official snapshot documents that `new_messages` is reset at phase change (`werewolf.md:301`); an authorized late message present in that transition result can therefore be absent from the next vote state and never enter policy memory. `runner.py:592–604` reads result sequence numbers for waiting, not policy history.

Most ordinary mid-window messages are ingested on later callbacks, so this is not a claim that every final message is lost. A transition-safe merge and acting-seat `get_messages` flush before voting are needed. The transport method exists at `starter/altruagent/mcp_game.py:169–176`, but the policy's `choose_action(state, context)` does not receive that transport. Use an isolated SDK overlay/hook with sequence deduplication; do not mutate a shared SDK or fetch another seat's transcript. New provenance must identify the overlay.

### 3. Speech consumes an attempt before acknowledgement — correctness/recovery gap

`hybrid.py:389–392` sets `_last_messaged_day` before selection or delivery. Rejected, unknown or late delivery does not undo it; `on_message_result` records acceptance but does not reconcile the speech state. A subsequent call terminates regardless of new information or actual delivery. Replace the day latch with per-window pending/accepted records, an attempted-send ceiling, transcript reconciliation for unknown delivery and quota accounting from acknowledgements. Rejected/ambiguous sends must not silently become an accepted public commitment or cause duplicate retries.

### 4. The semantic actor is largely blind to the actual game state — confirmed limitation

The proposer receives `_view`, but the learned selector receives only `features`, candidate labels and `{day, phase}` (`hybrid.py:249–275`). `semantic_selector.py:54–80` explicitly encodes eight coarse context numbers, and lines 147–168 attend over label embeddings and numeric rows. Investigation identities, attributed claims, contradictions, public deaths, recent exchanges and intended coalitions are not context tokens. Candidate text such as “Vote Player2” cannot tell the network what Player2 said. A correct `known_alignment` scalar does not reconstruct all other players' evidence.

Distinct histories can map to identical policy inputs. This is a representation limitation, not a failed attention implementation. A richer query must encode actual authorized observation, persistent player/evidence summaries and relevant visible dialogue. Keep fixed facts in separate protected slots, rather than putting everything into a string truncated at 256 wordpieces. Whether MiniLM's embedding of a numeric JSON array is useful is a hypothesis, not a measured conclusion.

### 5. Proposer ranking removes choices before RL can select them — action-space limitation

The local IDs path in `variants.py:239–253` scores one ID token and returns the top two. `_propose` keeps that set plus the leading rule option (`hybrid.py:210–236`). The neural selector cannot choose any omitted legal action or speech. Nonlanguage actions should expose every engine-legal ID, with explicit target and action kind, including abstention. `_actions` currently removes targetless alternatives whenever targets exist (`hybrid.py:113–114`, 145–146; `_target` maps ID7 to `None` at `policy.py:21–30`). This prevents strategic wolf abstention even though three consecutive nonelimination days can win the game for wolves.

Partner protection and positive seer results are currently policy filters, not engine legality. Preserve verified knowledge, but distinguish legal masks from utility priors; deliberate wolf bussing must not be declared engine-illegal. Changing coverage changes the action distribution and generator identity. Old propensities cannot train the new full-action policy on previously omitted alternatives.

### 6. Available speech does not express the required tactics — confirmed limitation

`hybrid.py:159–178` offers a generic question, a request to one heuristic target, a generic lean, a pass, and sometimes one positive seer disclosure. It cannot render negative-check disclosure, a response to a particular contradiction, a negotiated vote, a targeted defense, a deliberate wolf role claim, or an evidence-aware withholding decision. An LLM ranks these templates; it does not author context-specific language. The learned head cannot invent missing tactics.

Generate diverse typed speech intents and grounded realizations from the current seat view. Separate intent, target, recipient, asserted claim, source references and surface text. Validate verified claims against fact IDs; keep deliberate wolf bluffs in a separate assertion ledger, never the private-fact ledger. Safe canonical templates remain delivery fallback, not the entire language action space.

### 7. One broadcast per day cannot implement interactive play — confirmed limitation

`hybrid.py:384–397` terminates on the next same-day callback and constructs `SendMessage(option.label)` with no recipient. Yet the actual SDK supports one-recipient messages (`runner.py:250–265`), waits after each chat (`runner.py:913–924`), and permits five chats/50 words/120 seconds (`werewolf.md:275–303`). This leaves both responsive discussion and confirmed wolf-pair private coordination unused.

A bounded three-ACK opening/response/commitment controller is a concrete next interface. A response needs genuinely new relevant evidence; otherwise terminate. Private messages must address only a living confirmed buddy, never a suspected teammate. The SDK currently accepts only `SendMessage` or `TERMINATE_MESSAGING` (`runner.py:580–587`), so returning `None` or inventing a WAIT sentinel is a real API error. Existing post-chat waiting is sufficient for an initial responsive controller; a new waiting action would require a separate SDK change.

### 8. Channel/time/source information disappears before reasoning — confirmed limitation

`_all_messages` retains index, recipients and move index (`policy.py:33–40`); `_view` strips these fields and takes the last ten envelopes before filtering chat, with 240-character truncation (`hybrid.py:94–95`). Public evidence and a private wolf plan become indistinguishable quoted text; earlier important claims can disappear. Retain an append-only authorized transcript and immutable per-player claim summaries with channel/message/day/source IDs. The model may consume a bounded retrieval view, but protected facts and commitments must survive truncation. This is not evidence of cross-seat unauthorized reads: the inputs remain whatever this seat received.

### 9. The trained critic does not inform the existing actor update — intended algorithmic limitation

`train_semantic_ppo.py:192–201` requires the initial value to be zero and freezes `advantage = terminal_score - old_value`; lines 216–225 reuse those fixed advantages for all epochs. The critic is fitted separately at lines 92–111. Its pooled representation is detached, and runtime inference discards the returned value (`semantic_selector.py:168`, 254). There is no second collection batch that uses the fitted critic. This is valid Monte Carlo PPO with a zero baseline, not evidence of a useful learned actor–critic baseline or GAE. Recomputing advantages every epoch from the newly fitted critic would not be the correct repair.

Use successive fresh batches and persist the collector's previous policy/value checkpoint. Store values and terminal/death/truncation masks at collection; compute fixed returns/advantages for each batch using that frozen behavior critic. Death disables future actions but must not discard the eventual faction reward. Prefer audited rollout/return/value-learning code over another isolated optimizer. Start with a decentralized critic; do not claim MAPPO without actual multi-agent/centralized training and reviewed information boundaries.

### 10. Features and training unit are restrictive — limitation, with credit-assignment hypotheses

`selector.py:9–22` uses four candidate scalars multiplied by five context conditions. `hybrid.py:250–257` supplies ordinal LLM rank, clamped heuristic score, cumulative prior-vote share, known alignment and coarse bits. `public_support` aggregates historical votes (`hybrid.py:85–88`, 245–252), not current expressed intent or source credibility. The MiniLM head adds semantics to the same limited candidate pool; the LLM and encoder remain frozen. Initial-prior reproduction and later KL control are correctness checks, not strength guarantees.

Every sampled speech/action in an episode receives the same faction outcome; the actor objective sums decision surrogates per game and averages games (`train_semantic_ppo.py:29–33`, 63–71). This estimates the declared terminal objective, but high variance and longer-episode contributions remain plausible learning problems. Thirty games are not 136 independent game outcomes or 32 new batches. Do not label every winning speech desirable: teammates, roles and later actions affect the result. A diverse training population and more fresh episodes are needed before selecting a larger learning method.

### 11. Native strategic-model inference is incompatible with the legacy screen — interface risk

`variants.py:232–253` disables thinking and ranks one token rather than generating a complete answer. Applying only this path to a released strategic-reasoning checkpoint could suppress the capability being tested. Screen vanilla Qwen and MARSHAL through their intended tokenizer/chat template and bounded complete generation, with matched authorized state, budgets, format/target validation and actual GPU latency. Distinguish valid intended wolf deception from accidental corruption of verified knowledge. A native-interface screen is not a reason to remove legal/clock checks or accept truncated reasoning as an action.

### 12. Old logging cannot supply the new training state — migration requirement

`hybrid.py:346–368` logs eight features/labels/choice/probabilities, not the entire selector-visible packet, source-linked beliefs, value prediction or next-state masks. Action logging separately records observation and current-window messages (`logging.py:44–54`), while message logging lacks the private observation (`logging.py:56–63`). Reconstruction cannot recover unreceived final messages or the exact persistent context that never existed. Old R03 logs can support legacy audits and filtered supervised examples, not pretend to be full-state on-policy PPO trajectories for the new architecture.

## Public code/model reuse assessment

No external package, checkpoint or training script was executed. Repository commits and source were inspected read-only.

| Reference | Verified contents / permission | Reuse boundary |
|---|---|---|
| [SLA](https://github.com/zelaix/SLA), `c81f4500b0be74e4a270370ba35bb94caceb915b` | MIT LICENSE; recurrent MAPPO infrastructure and separate MAT attention implementation. | Reuse attributed rollout/mask/value modules and inspected attention primitives. `r_mappo/algorithm/r_actor_critic.py:32–39` is MLP/RNN; attention is in `algorithms/mat/algorithm/ma_transformer.py`. Do not assume the whole MAT actor is a seat-isolated candidate scorer. Its multi-agent input/slot assumptions require adaptation. |
| [Official MAPPO](https://github.com/marlbenchmark/on-policy), `de66d7a4b23fac2513f56f96f73b3f5cb96695ac` | MIT LICENSE. `r_mappo.py:179–205` consumes value-based advantages; `shared_buffer.py:179–203` implements returns/GAE; actor supports legal masks and recurrence. | Suitable learning-core reuse after asynchronous-turn/death/terminal and padded-candidate adapters. Use local critic inputs; do not introduce hidden opponents' private observations into actors. Preserve LICENSE/notices and record exact modified files. |
| [ONUW](https://github.com/KylJin/Werewolf), `4daf873e83fe1e9b029cb3d39c7e57f3baa5da10` | Real `discussion_policy.d3` exists. `core/dpins.py:23–33` embeds full history+belief; `training/train.py:20–27` is DiscreteCQL. No repository LICENSE found. | The six honest/deceptive tactic categories are informative design ideas. The policy expects 1536-d OpenAI ada embeddings (`onuw/utils.py:34–54`), not MiniLM384, and one-night role-swapping rules (`roles/base.py:25–29`). Do not transplant weights, pad embeddings, or vendor unlicensed code as a drop-in agent. |
| [MaKTO](https://github.com/ReneeYe/MaKTO-Werewolf), `32a57f04b371e5c88c4d77422f175e5e899fed9f` | Public extraction/interface code; no top-level LICENSE found. README:167–169 promises model release and uses the nonstandard model-license string `CC BY-NA-SA 4.0`. Runtime hardcodes nine-player/Chinese conventions (`makto_agent.py:45`, 49–58). | Not verified author pretrained Werewolf weights. Reuse concepts and independently written adapters; clarify source permissions before copying. Nine-player Witch/Guard/Hunter transitions are incompatible. |
| [MaKTO public dataset](https://huggingface.co/datasets/ReneeYe/werewolf_game_reasoning) | Official HF card declares MIT; instruction/prompt/response/meta data. | Potential later SFT/KTO data, after acting-view/privacy, role/rule/language filtering and game-level split. Outcome-derived desirability is not ground-truth strategic quality. |
| [MARSHAL generalist](https://huggingface.co/nics-efc/MARSHAL-Generalist-Qwen3-8B), SHA `369d9decea6e8b347f81706fa169dbbddddce360` | Parent/Reviewer verified author-linked Apache-2.0 strategic-RL release. | Genuine strategic pretrained contender, not verified Werewolf champion. Screen its native generation interface against vanilla8B before collecting training data. |

The official [TRL license](https://github.com/huggingface/trl/blob/main/LICENSE) and [LlamaFactory license](https://github.com/hiyouga/LlamaFactory/blob/main/LICENSE) are Apache-2.0. These are concrete reuse options for later LoRA/SFT/KTO rather than a new handwritten token trainer. Pin an API-compatible revision; source/model/data licenses remain separate. A third-party public-data SFT card was inspected but not selected: its own description does not claim complete MaKTO or held-out strength, so it is not a replacement for the matched author-release screen. No further third-party weight search is required for this review.

## Recommended repair order and interfaces

1. Correct the immutable-fact score invariant, transcript transition and ACK state machine before training. Version both the isolated transport overlay and observation serializer.
2. Create a full acting-seat `ObservationPacket` and `CandidateSet` interface. Keep verified facts, attributed claims, deliberate self-bluffs, full legal actions, dialogue channels and budgets separate. Preserve all engine-legal nonlanguage actions. Encode current player evidence and context in the selector query, not just a coarse numeric array.
3. Run the matched vanilla/MARSHAL native-generation GPU screen. Freeze selected backend/template/revision and validated budget before strength comparisons. No new training needs to precede this capability test.
4. Generate diverse contextual speech under explicit tactics, then allow a bounded number of ACKed, evidence-triggered responses and confirmed-buddy coordination. Test real platform quotas and terminal boundaries.
5. Collect new source-complete, on-policy trajectories and adopt a licensed recurrent/value-learning core. Compare an untrained new interface against the same adapted interface to separate candidate/model improvement from RL. Keep D7 as the actual release until fresh evaluation justifies a change.

Parent acceptance and Reviewer approval of `02_contract.md` are required before implementation. This document is a diagnosis and design proposal, not an assertion that the proposed architecture will win.
