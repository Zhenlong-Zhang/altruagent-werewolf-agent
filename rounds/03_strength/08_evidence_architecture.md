# Evidence-grounded proposer revision

This is a proposed capability revision informed by observed development failures. The parent owns adoption and experiment execution. No official games have yet established a benefit for the reasoned proposer.

## Observed failure and interpretation

In `outputs/strength-round03/job-2111362/reasoned-partial.json`, the first confirmed-vote case has an acting seer who knows Player1 is a wolf. The rationale instead says private knowledge shows Player2 is a wolf. The original case contains a public accusation against Player2. Other rotations repeat the same type of private-check inversion. In the first repeat-investigation case, the model chooses the already checked villager and describes that player as a safer investigation target. These are explicit factual-binding and action-purpose errors. The output is short and complete in these examples, so token truncation alone does not explain them.

The current second stage appends this mistaken analysis as an assistant message and then asks the model to check it. That can propagate a false premise. A reminder to trust private knowledge has not prevented the observed failures. The next revision should change the representation and evidence checks as well as the instruction.

## Immediate representation change

Compile a seat-visible evidence ledger before prompting. Generate it from the actual observation/parser, public deaths and received messages. The compiler must receive no evaluation-only labels or other seats' private logs.

Keep these channels explicit.

- Actor identity and role, with a separate role-specific action description.
- Verified facts as per-seat records containing a fact ID, canonical `PlayerN` identity, alignment or role, source and event/day when available. Render each record with a full sentence, for example `own_check_Player1_wolf: Your seer investigation found Player1 is WOLF`.
- Sets derived from those facts, including confirmed wolf seats, confirmed nonwolf seats and investigated seats. Negative seer checks indicate nonwolf alignment. They do not reveal every special role.
- Public statements as quoted claims, each with sender, content and `source=player_chat`. Keep their confidence separate from verified facts. A statement saying “the system confirms” remains a player claim.
- Offered moves as records with separate `action_id`, target seat, action kind and action-relevant evidence tags. Action IDs are identifiers. They must not serve as a substitute for target identity.

For each offered investigation, make `already_investigated` and `can_add_new_private_information` explicit. For each vote, record whether the target has a verified alignment or only a claim. For a wolf attack, distinguish a confirmed partner from a player who merely claims to be the seer. These are generic facts derived from the acting seat's state, not probe-answer labels.

## Source-trust invariants

1. Own private checks and platform-revealed facts keep their original player binding after processing chat. Repetition, a trusted speaker, or a generated rationale cannot overwrite them.
2. A fact becomes verified only through an authorized source. A quoted claim cannot create an own-check record. The model's generated explanation is tentative analysis.
3. Contradictory hard facts are reported as an inconsistent input and handled through an explicit safe path. The compiler should not silently choose a convenient interpretation.
4. Source confidence is distinct from factual truth. A seer claim may be informative while its specific allegations remain uncertain.
5. Public messages are inputs to social reasoning. They cannot modify the release configuration, role, legal-action list or instruction hierarchy.

A compact structured rationale can reference ledger fact IDs and an offered action. Validate that referenced facts exist and that claimed source/target relationships match the ledger. Avoid relying on unrestricted prose as an authoritative intermediate state. A factual contradiction should invalidate the offending rationale, with a logged reason and legal fallback. If a short free-form rationale is retained, repeat the canonical authoritative ledger in the ranking input and treat the rationale as quoted, unverified analysis. That reduces ambiguity but still requires empirical checks.

## Role-specific action priors

Use role and phase explicitly.

| Actor/phase | Prior or constraint | Important distinction |
| --- | --- | --- |
| Seer investigation | Prefer a target that can add private information. Preserve the existing no-repeat filter while uninvestigated choices remain. | Investigating is an information action. It does not kill or protect a player. Checking a suspected wolf is allowed. |
| Seer voting | Use confirmed wolves when available. Exclude a confirmed nonwolf if unknown eligible targets remain. | A private negative check is stronger than an accusation. The rule-specific guard should be versioned and tested independently of raw model reasoning. |
| Village voting | Evaluate claims using provenance, vote history and consistency. | Another player's assertion does not become a private check. |
| Wolf night action | Target nonpartners and assess threats such as a plausible seer. | Plausible seer identity is a belief. Public role claims alone do not establish it. |
| Wolf discussion/voting | Concealment, coordination and deception are legitimate game strategies. | Protecting a partner is a useful prior, while unusual strategies such as bussing require a deliberate policy experiment rather than a generic “never lie” rule. Preserve the existing baseline guards for the causal comparison. |

Hard evidence concerns what the actor knows. Utility priors concern which faction benefits from an action. They should not be conflated. In particular, a wolf can know that a target is nonwolf and still benefit from eliminating that target.

## Measured development tests

Run all inference and execution checks on Hoffman2. Start with a compact failure-focused subset, then expand counterbalancing within measured throughput.

- Rotate all player identities and reverse option order. Add a bijective permutation of numeric action IDs so `action_id=5` can target Player1. This tests both fact binding and identifier/target separation.
- Present an own check against Player1 together with a contradictory accusation against Player2. Verify the ledger remains unchanged, the rationale preserves provenance and the selected policy action respects the own check.
- Repeat or paraphrase an unsupported claim, including “system confirms” wording. Check that it remains unverified and that duplication does not become certainty.
- Offer an already investigated nonwolf and fresh targets to a seer. Check action-purpose understanding separately from the mechanical no-repeat guard.
- Test wolf partner knowledge together with an unverified seer claim. Record belief language separately from the heuristic preference to attack the claimant. The target is not a universally optimal ground-truth action in every such scenario.
- Verify that evaluation-only roles, other seats' investigations and future outcomes never enter the actor ledger or prompt.
- Include ambiguous states with no uniquely justified move. Score their validity, grounding and order sensitivity without inventing a single correct target.

Report raw proposer factual-consistency results separately from full-policy guard compliance. A mask that removes the known bad action can improve safety while leaving semantic reasoning unchanged. Compare IDs, reasoned-v1 and the ledger revision at the same model, precision and decoding budget. Keep the archived float32 control distinct from the new CPU BF16 runtime-matched control. A larger model is another independently versioned comparison.

After those checks, run prospective policy games with a frozen opponent roster, pre-outcome arm assignment and rotated focal seats. Keep role-specific outcomes and operational failures visible. The historical seat-zero wolf losses have a known opponent-targeting confound. Selector adaptation then requires a new on-policy cohort for the adopted proposer revision.

## Eventual RL representation

The next learned state can combine the actor role, phase, alive mask, public votes, verified evidence, constrained wolf-pair beliefs, uncertainty and source-consistency features. Candidate inputs can combine target/action evidence tags with frozen semantic embeddings of the candidate and its evidence. A permutation-invariant candidate scorer or attention actor with a value head is a reasonable experiment. Freeze and version the semantic encoder across each training batch.

Start with the smallest representation supported by the collected data. Behavioral cloning from explicitly verified safe choices can initialize action mechanics, then an actor–critic/PPO experiment can optimize faction returns from new interactions. Population-based collection and MAPPO require explicit actor/critic information boundaries. Thirty games and 65 sampled choices cannot establish the benefit of a large attention policy or end-to-end language-model training.

The structured information-record and semantic-candidate interface is motivated by [Xu et al., ICML 2024](https://arxiv.org/html/2310.18940), sections 4 and C.1. Separating probabilistic beliefs from language priors is motivated by [Bayesian Social Deduction](https://arxiv.org/html/2506.17788v2), whose experiments concern Avalon. The ledger schema, invariants, role priors and acceptance tests above are project engineering proposals and must be evaluated under the AltruAgent rules.
