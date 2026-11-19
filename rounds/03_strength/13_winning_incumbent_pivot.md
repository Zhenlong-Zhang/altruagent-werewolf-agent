# Pivot following prospective game evidence

## Evidence and decision

Phase B enrolled25 matches: legacy IDs won9/12 and grounded won5/13. All113 focal decisions were acknowledged; neither arm had a model fallback or delivery error. All25 controller receipts match assignment hashes/arms/seats/namespace/seed. One controller UUID receipt was recorded after a delayed terminal snapshot and is explicitly flagged. Role-standardized rates were0.743 and0.325, but the samples and role/seat cells are small; grounded had four seat-zero games versus one for IDs. These are pilot findings, not a precise causal estimate.

The independent posthoc audit found that the learned head selected the model's first proposal in every model-backed greedy decision. Grounded changed speech from mostly generic questions to mostly unsupported specific-target preferences. Six later grounded votes differed from the preceding preferred target. A tentative preference can legitimately change when information arrives; these sequences motivate consistency/communication analysis rather than proving each change irrational. The underlying speech shift and lower observed return make grounded unsuitable for automatic promotion despite42/42 synthetic evidence-constraint checks.

Retain legacy IDs as the competition incumbent. Replace the next broad-prompt adaptation with a narrow, separately versioned guard: on a seer's day vote, keep confirmed-wolf priority; otherwise exclude privately verified nonwolves when unknown eligible targets remain. Keep the original ID prompt, ranker, speech and other roles/phases. Count actual slate changes, not merely the guard-reason label.

## Preserved attempts and superseded plan

Job2114248 passed31+9+5 checks and completed one sampled grounded mixed-roster game. That exploratory episode, its receipt and raw logs are retained. No gradient update was made because the30-game threshold was never reached. Its runtime was stopped after its only game had completed. The pending original head-only evaluation job2114507 was cancelled before execution or match creation. The original future plan in11_onpolicy_contract.md is superseded by this document; its records remain historical.

No social-policy code is part of this pivot. Responsive conversation and wolf coordination remain proposed work in10_responsive_speech.md. Cyclic aggregation remains off after its full-context budget failure.

## New sampled training cohort

Freeze a new source snapshot and cohort identity before creating games. Use Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218, BF16 CPU, IDs proposer, verified-innocent-v1 guard, hybrid_collect sampling and checkpoint d7ebf7670444067a. Collect30 eligible completed sampled focal games with seed r03-ids-guard-onpolicy-v1 and namespace wolf-collect-v1. Keep all assigned/failed/excluded games in the controller census at data/round03-ids-guard-onpolicy/controller_census.json.

Use one direct-model opponent plus five rules policies, ordered by nonfocal seat. The initial16CPU128GB request had no eligible node with16 free cores (maximum10) and an evening start estimate. Its unstarted train/evaluation jobs2116519/2116555 were cancelled. Allocate8CPU64GB and one match/two HF processes, with4 intra-op and OMP/MKL/OpenBLAS/NumExpr threads per process. This preserves per-model resources and reduces concurrency. Verify realized opponent model validity/fallback and focal delivery; configurations alone do not establish a strong opponent population.

Make one on-policy REINFORCE update with the predeclared step0.5 and the existing20-coefficient selector. Freeze the proposer and starting selector throughout sampling. The update is experimental; it does not fine-tune the language model. Preserve parent ID, generator identity, exact propensities, terminal validation, role counts and training match hashes.

## Fresh complete-policy comparison

The next untouched match cohort compares four profiles under the same model, precision, thread caps and opponent roster:

| Arm | Guard | Selector | Mode |
|---|---|---|---|
| incumbent | none | d7ebf7670444067a | greedy |
| guarded | verified-innocent-v1 | d7ebf7670444067a | greedy |
| sampled | verified-innocent-v1 | d7ebf7670444067a | sampled |
| adapted | verified-innocent-v1 | newly trained child | sampled |

This includes the best observed deployed baseline and separates the guard, exploration and update comparisons. It replaces the inadequate comparison of two grounded heads. The profile factory hashes arm and focal seat before observations using namespace wolf-policy-v1 and a fresh seed. It logs the actual guard, mode and selector. Evaluate-only mode prevents this sampled comparison from triggering training.

Target12 completed matches per arm as a pilot, retain all assignments and censoring, and verify no overlap with the training hashes. Report role/seat counts, Wilson intervals, actual interventions, model configuration, latency and delivery/fallback for each arm. Different modes are intentional profile components. The comparison is held out from fitting and is used for policy selection; it is not an untouched final test after selection. Small cells, multiple comparisons and the narrow roster constrain any strength claim. Keep the incumbent unless the combined evidence supports a change.
