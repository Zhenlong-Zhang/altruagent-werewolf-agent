# Historical focal-seat failure cases

Read-only review of round-one training records, 2026-11-19. Only focal seat-zero logs and the public rules-policy source were inspected. The prospective comparison cohort was left untouched. These are development diagnoses. An episode loss alone cannot establish that a particular move was wrong.

## Verified-innocent ranking near-miss

[Seer loss, decision at line 9](/Users/zhangzhenlong/Desktop/werewolf_agent/data/qwen3-8b-round1/decisions/match_c5ae81de08f30209_seat_0.jsonl:9).

The observation explicitly records that the seer investigated Player1 and found a villager. Nevertheless, the preceding `hybrid_choice` ranks Vote1 above Vote2 and assigns Vote1 probability 0.562. The features themselves correctly mark Player1's known alignment as -1 and its rule score as -1. The actual sampled choice was Vote2, an unknown target. This is an observed proposal/selection-distribution defect and an avoided bad vote in this episode. The record does not show the agent actually voting for the verified villager.

Hypothesis to test: role-appropriate exclusion of confirmed villagers when unknown targets remain can prevent this specific failure. Current `_actions` prioritizes confirmed wolves for a seer but leaves confirmed villagers eligible when none are known. A better rationale proposer might also improve ranking, but that effect remains unmeasured.

## Specific accusation with no recorded evidence basis

[Villager loss, speech at line 2 and vote at line 5](/Users/zhangzhenlong/Desktop/werewolf_agent/data/qwen3-8b-round1/decisions/match_3be713860e47a0a0_seat_0.jsonl:2).

At day one, the actor sends a message leaning toward Player1 and subsequently votes for Player1. The logged observation says the actor knows no other roles. Recorded history has no deaths or prior ballots, the action's `new_messages` is empty and suspicion scores identify only the actor's own role. Offered speech alternatives are a generic evidence request and the Player1 lean. Offered votes are Player1 and Player4, with equal rule evidence and a larger LLM-rank feature for Player1.

The record supplies no grounded explanation for the particular target. Player1's true role is unavailable from the acting-seat evidence, so this review cannot label the vote factually incorrect. Earlier callbacks can also cache messages that are absent from the final decision delta. Hypotheses are target-index preference and weak evidence-conditioned communication. Future logging should preserve the legitimate compact proposer view or its evidence references to distinguish them.

## Public target and vote diverge

[Wolf loss, speech at line 2 and vote at line 5](/Users/zhangzhenlong/Desktop/werewolf_agent/data/qwen3-8b-round1/decisions/match_7a25413268812cb3_seat_0.jsonl:2).

The wolf publicly leans toward Player1, then samples Vote5. Its observation identifies Player4 as its wolf teammate. The vote slate contains Player1, Player5 and Player2, so the teammate is correctly excluded. The speech slate contains a generic request and the Player1 lean. No private coordination message or updated public recommendation appears in this focal log. The recorded rule score for Player1 is slightly negative, which may reflect prior messages.

Switching targets can be strategically justified. The log does not establish that Vote5 caused the loss. It does identify a coordination question and a limited offered speech set. Test whether explicit vote coordination or target-update intents help against varied partners before claiming a remedy.

## Fixed seat-zero opponent bias

[Example wolf loss](/Users/zhangzhenlong/Desktop/werewolf_agent/data/qwen3-8b-round1/decisions/match_634c619c155fbe98_seat_0.jsonl:5) and [rules seer selection](/Users/zhangzhenlong/Desktop/werewolf_agent/werewolf_agent/policy.py:76).

Inspection of the 30 focal logs found eight wolf games, all losses, with the evaluated actor always seat0 and the other six agents using rules. In this setup the seer must be a different seat whenever the focal actor is a wolf. The rules seer's first investigation breaks equal scores by the lowest eligible target. With no first-night public evidence, that target is seat0. The peaceful first night therefore exposes this particular focal placement to a systematic investigation.

The first-investigation preference follows from the roster, platform roles and policy source. The precise downstream contribution to each loss would require further legitimate trajectory evidence. Prospective seat rotation and role-stratified comparison are necessary to separate target-order exploitation from general model ability. This review makes no claim that the new proposer changes the pattern.
