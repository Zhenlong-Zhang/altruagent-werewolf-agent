# Round 03 closeout — bounded experiment completed

The contracted training, Phase F and bounded Phase G evaluations are complete. The incumbent remains the delivery choice: checkpoint `d7ebf7670444067a`, `hybrid_greedy`, evidence guard `none`. The semantic checkpoint is retained as an evaluated research artifact. **Completing this experiment does not mean the broader objective of winning the competition has been attained.**

## Completed work and evidence

- Preserved the IDs incumbent after broader generated-reasoning and grounded-ranking development variants failed to justify promotion. The independent verified-innocent guard changes only seer daytime voting; all-known legal fallback remains available.
- Implemented a frozen-Qwen proposer plus frozen-MiniLM semantic candidate attention actor/critic. There are 30,421 actor parameters, including the 20 base coefficients, and 1,345 critic parameters. This is focal small-head PPO, not token fine-tuning, centralized multi-agent training or full MAPPO.
- Trained on the frozen E batch: 30 completed games, 136 acknowledged sampled choices. Initial logits exactly reproduce the behavior policy using the current Python float64 accumulation path. All 32 PPO epochs were accepted; final mean/max training KL was 0.005938/0.035033. The separate saved-policy audit passed source-log, artifact, parent, encoder and match-set checks. Training-state mean TV was 4.09%; 4/136 greedy argmax choices changed.
- Hoffman2 verified the focused regressions, schema-2 reload, actual encoder and neural inference, and mixed-roster readiness. The original failed exact-logit gate and the plain-Python audit import failure remain in the trace; both were repaired and rerun without changing outcomes or relaxing the gates.
- Completed prospective F: 53/53 games resolved. Incumbent greedy: 7/14 wins; guarded greedy: 8/14; old guarded sampled: 6/12; semantic guarded sampled: 5/13. Role and seat imbalance remain visible. F operations audit passed with no protocol issues, E/F overlap, model/selector fallbacks or unconfirmed focal deliveries. No strength improvement from the sampled NN was demonstrated.
- Completed G: 54/54 assigned games resolved, including all hash-assignment oversampling. Incumbent greedy won 16/21 (76.2%; 95% Wilson 54.9–89.4%); old guarded greedy 7/12 (58.3%; 32.0–80.7%); semantic guarded greedy 11/21 (52.4%; 32.4–71.7%). Their villager/seer/wolf counts were 13/3/5, 7/2/3 and 12/1/8. These small, role-imbalanced groups do not establish an intrinsic ranking, but they provide no demonstrated NN gain supporting promotion.
- G operations passed: 239 focal decisions acknowledged, zero model/selector fallbacks or errors, no controller/configuration issues. The split audit confirmed E/F, E/G and F/G match disjointness. The frozen candidate was not retrained or tuned using these outcomes.
- The actual same-observed-slate diagnostic passed. Recorded NN greedy choices differed from old guarded D7 choices in 1/98 eligible decisions, one seer speech choice in 1/21 NN games. One forced singleton was excluded. This is a conditional comparison of recorded choices on identical candidate features, not a simulation of alternative states or counterfactual wins.
- Finalized the concise English report at `outputs/strength-round03/strength_report.html`. The sampled F and greedy G results remain separate. Read-only audits and observed-slate diagnostics do not fit models or alter checkpoints.

Primary computed evidence:

- `outputs/strength-round03/job-2123930/training-report.json`
- `outputs/strength-round03/job-2123942/semantic-update-audit.json`
- `outputs/strength-round03/job-2123942/final-summary.json`
- `outputs/strength-round03/job-2123942/final-operations.json`
- `outputs/strength-round03/job-2133983/final-summary.json`
- `outputs/strength-round03/job-2133983/final-operations.json`
- `outputs/strength-round03/job-2133983/split-audit.json`
- `outputs/strength-round03/job-2133983/greedy-action-differences.json`

## Final decision and handoff

G used the predeclared fresh seed `r03-semantic-greedy-heldout-v1` and unchanged `wolf-policy-v1` assignment mechanism. It finished with 21 incumbent, 12 guarded and 21 NN games, satisfying the minimum of 12 per arm without extending the 75-game/three-hour contract. G was declared before the aggregate F readout; one F canary had already been inspected for operational validation.

Retain the original unguarded greedy incumbent. Neither sampled F nor greedy G demonstrated a benefit from promoting the semantic head. Broad intervals and different role mixes prevent interpreting the observed rankings as proof of intrinsic superiority. The semantic artifact remains available for reproducibility and future research, rather than being relabeled a winning competition policy.

The final G analysis succeeded on Hoffman2 as audit job 2141810. A later duplicate run regenerated a byte-identical summary, then was refused by the immutable-output guard; the successful audit files remained intact. The verified final G summary SHA256 is `38b947e01356d70b14ef80081feeb278903727147046da70b4732b31e1b47c42`. The parent owns the final source/artifact handoff, project-memory consolidation and delivery review; no new model computation is required to reproduce the report from the supplied JSON.

## Workflow and ownership

Used the `round-based-research-workflow` skill in project-round mode with real Planner (`planner_strength`), Worker (`worker_strength`) and Reviewer (`reviewer_strength`) roles; the parent owns synthesis, Slurm execution, final selection and delivery. The Worker implemented contracted isolated changes and read-only audits; Reviewer accepted source boundaries before Hoffman2 execution. The UCLA server workflow confined model computation, training and gameplay to allocated Slurm nodes. This final report formats existing computed JSON only; no local model analysis, inference or gameplay was run.

Core implementation spans the contracted proposer/guard/production paths, semantic selector/trainer and focused checks. The complete source and experiment manifests remain authoritative. Read-only helper changes and their hashes are tracked separately from frozen runtime files. The Worker finalized the HTML and this closeout; the parent consolidated README, round memory and the release manifest. User changes to WORKPLAN and the existing image remain untouched.

## Limits and unresolved questions

Thirty training games remain thirty terminal observations despite 136 choices and 32 epochs. Wilson intervals in F/G are broad; platform roles were uncontrolled, and the one-direct-plus-five-rules roster cannot establish tournament-wide generalization. Training-state KL/TV and parameter changes verify the update, not its playing strength. Private evidence grounding and legal guards are distinct from learned social reasoning. Responsive dialogue and cyclic ranking were not activated in the final candidate. F/G are policy-selection evidence, not an untouched postselection final test. No further tuning occurred within this frozen selection pilot; future improvement iterations must use fresh evaluation rather than adapting to these held-out outcomes.

The broader competition-winning objective remains unresolved. Future strength improvements are within the user's broad project request but require new, clearly separated training/evaluation evidence. Any competition deployment must use the verified production entry and exact retained incumbent artifact, with its operating assumptions visible.
