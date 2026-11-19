> Superseded after the prospective loss audit; see13_winning_incumbent_pivot.md. The first exploratory episode is retained.

# New on-policy cohort and checkpoint evaluation

Recorded before the new training cohort. Phase B remains a separate frozen proposer comparison against six rules opponents. Its greedy logs will not be used as a sampled training set.

## Training specification

- Isolated phase-D source manifest: outputs/strength-round03/phase-d-source.json. Slurm job2114248 starts only after job2113134 ends; the controller confirms old runtime termination and the new runtime's actual connection before creating more games.
- Qwen3-8B at revision b968826d9c46dd6066d109eabc6255188de91218, CPU BF16, grounded-v2 with order balance none. Cyclic aggregation remains experimental: full-context supported K7 exceeded30seconds and its wolf heuristic diagnostic improved only7/14to8/14.
- Allocate16CPU128GB for at most two concurrent games. Each has one focal model and one direct-LLM opponent, so at most four model processes. Explicit Torch intra-op, OMP, MKL, OpenBLAS and NumExpr caps are4 per process. Measure actual memory/latency rather than treating the allocation as proof of readiness.
- Fixed roster: one direct Qwen3-8B opponent and five rules policies, ordered by nonfocal seat. The direct slot is not shuffled. Preserve only each actor's own authorized observations and history. Report actual model fallback/delivery rates for the opponent as well as its configured label.
- New collection seed r03-grounded-mixed-onpolicy-v1; hash the session ID with namespace wolf-collect-v1 to select one focal seat before roles or outcomes are read. Use the exact competition factory at that seat, in hybrid_collect sampling mode.
- Start from checkpoint d7ebf7670444067a. Freeze the proposer, roster and starting selector throughout collection. Target30 eligible completed sampled focal episodes; retain controller receipts for every assigned match, including failures and games excluded from the gradient.
- Apply one REINFORCE update with step0.5, predeclared as an experimental tenfold increase over the previous0.05 step. Validate unique terminal, checkpoint/generator identity, exact sampling propensities and acknowledged sampled deliveries. Output includes parent, step, eligible match hashes and role counts. This updates the20 selector coefficients; it does not fine-tune LLM token weights and is not PPO/MAPPO.

## Fresh evaluation specification

Use separate matches after fitting, the same model/precision/thread caps/roster and the same sampled mode for both incumbent and adapted selectors. Hash arm and focal seat from session ID with namespace wolf-selector-v1 and a new seed. The checkpoint factory logs the chosen arm and actual selector ID before observation. Enable evaluate-only so sampled evaluation cannot trigger training.

Initial target12 completed games per arm is a pilot. Report all assigned/unresolved games, role/seat counts, acknowledged decisions, fallback/latency, Wilson intervals and role-standardized outcomes when all roles are observed. Verify zero intersection with training match hashes. Do not interpret small aggregate differences as tournament superiority, and do not assume sampled-policy findings transfer to greedy deployment. Retain the incumbent when evidence does not support promotion.

## Interpretation boundaries

A new checkpoint establishes a completed update. The controlled held-out results assess its effect under this roster. The direct opponent's realized validity/fallback rate, narrow roster and small role-specific samples constrain conclusions. Responsive multi-message speech remains design-only in10_responsive_speech.md; no social intervention is silently part of this cohort.
