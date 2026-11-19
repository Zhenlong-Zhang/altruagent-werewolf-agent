# Next intervention — responsive discussion and wolf coordination

This proposal is for a separately versioned Phase C. The current primary A/B comparison stays frozen. The change is a bounded discussion state machine with grounded message intents, initially capped at three accepted chats per window. Three is a revisable engineering choice. The platform permits five chats, 50 words per message, a 1024-character limit and a 120-second open-discussion window. Private chats consume the same chat quota.

## State machine

Store state by match, seat and discussion window/day. Track accepted chat count, pending delivery, processed message sequence, own public vote intention and termination status.

1. **Opening.** A village-side player broadcasts one grounded evidence statement or targeted question. A wolf with a living, engine-confirmed partner first sends a private coordination proposal to that partner. The next eligible callback sends a public cover statement, subject to remaining time and quota.
2. **Response.** Allow one response to material new information. Triggers include an accusation contradicted by an own seer check, a new seer/check claim, a direct question, a partner proposal, or evidence sufficient to change the intended target. Select from source-bound reply intents. Repeating an old message or generic chatter does not qualify.
3. **Commitment update.** If a target changed and clarification still helps, state the updated public vote intention or acknowledge the partner's target. This uses a remaining slot. The intention is a record of the current plan. New evidence may justify a different vote, with the reason logged.
4. **Terminate.** End when no useful response is available, the local/official quota is exhausted, the window is ending, or the remaining time cannot cover another measured inference and delivery. Record the terminate acknowledgement and keep collecting visible messages for later voting.

For a wolf, private coordination plus public cover normally uses two slots, leaving one for a reply or changed-target update. For the village side, opening, reply and optional update use at most three slots. Only accepted chat deliveries advance the counter. A rejected or ambiguous delivery must not silently consume the next plan step.

## Message intents and truth rules

| Intent | Eligibility and renderer | Example |
| --- | --- | --- |
| Share positive/negative own check | Actor is the seer, target and alignment match a real own-check fact ID. A negative check means nonwolf alignment. | “I am the seer. My investigation found Player2 is not a wolf. Please reconsider the accusation against Player2.” |
| Ask for evidence | Cite a received public claim or ask a neutral question when no specific evidence exists. | “Player3 says Player2 is a wolf. What check or voting evidence supports that claim?” |
| State suspicion | Mark uncertainty and cite an actual public event/claim. A villager cannot assert an own investigation. | “I lean toward Player2 because of the claim from Player3, but that claim is unverified.” |
| Coordinate with wolf partner | Recipient is the single living partner identified by the actor's private role observation. Proposal targets are legal/live and distinct from the partner unless a separately tested strategy allows otherwise. | Private to partner: “I propose voting Player2 today. Player5 claims to be the seer; consider Player5 tonight.” |
| Update intended vote | Link to the earlier own intention and new supporting information. | “I am changing my intended vote to Player4 after that explanation. My earlier preference was Player2.” |

The renderer binds target IDs and evidence references programmatically, checks word/character limits and sets recipients explicitly. It must distinguish public statements from private coordination. A shared model service receives only the acting seat's own legitimate observation and visible messages.

A wolf's deceptive public speech is a legitimate optional strategy. A concrete later candidate is a counterclaim such as “I am the seer. My check found Player2 is a wolf.” Mark it internally as a deliberate bluff, record the target in a separate **public-claim ledger**, and preserve the true private role/check ledger. Future reasoning must know that the actor's own fake claim is a bluff. This candidate needs an explicit role gate and its own ablation before routine use. The initial responsive intervention can measure truthful/uncertain replies and partner coordination first. A generic prohibition on unverified statements must not accidentally remove all legitimate wolf deception.

## SDK behavior and immediate feasibility

The current runner already calls `wait_for_update` after each chat, then invokes `choose_message` again on a fresh state. Reply-on-next-update therefore fits the existing flow. `choose_message` currently accepts only `SendMessage` and `TERMINATE_MESSAGING`. Returning `None` or the realtime action `WAIT` sentinel would raise a decision error. Under the first version, an update with no actionable response leads to termination. Continuing to listen through irrelevant updates would require a separate bounded messaging-wait extension in the SDK.

Use the authoritative `remaining.this_turn` from message acknowledgements when available, together with accepted-count state. Leave a pending step unresolved until delivery is known. Reconcile own transcript entries before retrying an ambiguous send. If certainty cannot be recovered, avoid generating duplicate assertions and record the ambiguity. Do not assume termination can be undone while the window is still open.

The guide says `new_messages` resets at phase changes and is empty during voting. `mcp_game.get_messages(since=...)` is available, but the runner does not currently refresh the full transcript before a vote. Add a separately versioned, seat-visible transcript refresh at the discussion-to-vote boundary and deduplicate by message sequence. A last message arriving with the phase flip must remain available to the voting policy. Preserve the source and visibility of each message.

Derive the inference budget from `messaging_seconds_left` and measured tail latency plus delivery reserve. A critical own-check message can use its verified template directly. Extra model calls are unnecessary for formatting a known fact. Stop before an inference is likely to overrun the window.

## Observability and acceptance

Log the speech-policy version, intent, target, channel/recipient, supporting fact/claim references, triggering message sequences, window time, accepted chat count and server quota. Keep pending delivery and acknowledgement events linked to the selected intent. Record public commitment and later vote target, including a reason for any change. Count direct and broadcast messages together. Record pass/terminate decisions so that the transcript available at vote time is reconstructable.

Targeted checks should cover an own-check contradiction, an unsupported public seer claim, repeated/irrelevant messages, a dead or absent wolf partner, private/public channel separation, deliberate bluff versus private truth, rejected and ambiguous sends, quota boundaries, phase flips and reconnect reconstruction. The test for a phase flip should inject a final visible statement and verify that voting receives it.

Measure accepted useful replies, coordination agreement, evidence-grounding failures, timing/quota errors and terminal wins by role. Preserve the same language model and selector for the first speech comparison. A practical next official roster is one frozen direct-language opponent plus five rules agents, with one focal candidate. This requires two model instances under the current process design; the parent's proposed 64 GiB single-match allocation needs an actual peak-memory check.

Compare the one-message incumbent, responsive replies and replies with partner coordination through explicit versioned arms or staged ablations. Any later selector update uses a separately versioned on-policy cohort because the offered messages, channels and information timing have changed. The current frozen primary A/B and its logs must retain their original identities.

Source checks are the local platform guide `werewolf.md`, section 8, plus `starter/altruagent/runner.py` message validation and wait behavior and `starter/altruagent/mcp_game.py` transcript support. This document proposes the intervention and its tests. It contains no implementation or measured win-rate claim.
