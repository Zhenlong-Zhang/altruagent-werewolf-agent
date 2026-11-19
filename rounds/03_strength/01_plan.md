# Strength round plan

## Objective and starting evidence

Improve measured Werewolf performance while making the chosen policy executable in tournament context. The parent agent owns final choices. Preserve the Qwen3-8B round-one checkpoint and its collected data as a control.

The archived CPU float32 ID proposer scored 21/42 preferred choices on the small counterbalanced development probe, equal to the 21/42 first-option control. New H2 job 2111362 reproduced all 42 archived float32 rank lists exactly. The earlier 20/42 ID and 4/42 Yes/No results were GPU bfloat16 measurements; precision and device context must accompany these comparisons. These are diagnostic findings. Several probe scenarios would be forced by production hard-evidence filtering, so their accuracy cannot substitute for game outcomes. The current selector uses 20 hand-feature coefficients and one REINFORCE update from 30 games against six rules opponents. Its three fresh operational wins establish integration.

The user explicitly prioritizes competition score, permits stronger models, and asks for implementation following the reviewed papers. A model-size or algorithm preference is an experimental choice. The user-modified WORKPLAN.md remains outside this round's edits.

## Priority intervention

Implement one bounded reasoned-ranking proposer as a distinct experimental backend. Give it the acting seat's legitimate structured evidence and offered action or speech alternatives. Generate an auditable bounded rationale, then rerank the offered IDs conditional on that rationale. This first implementation retains offered-ID token scoring for the final ranking and therefore requires direct tests that reasoning reduces numeric and order bias. Log the rationale, resulting legal IDs and any fallback. Preserve the old immediate-scoring backend unchanged.

Begin with offered safe language intents so the first test isolates semantic reasoning and ranking. Context-specific free speech is eligible as a subsequent separate change, with explicit provenance for factual claims. A larger neural selector, MAPPO and CFR/DPO remain further hypotheses. The current dataset is small for such additions.

CPU BF16 on the measured Xeon6 family is a runtime candidate. Measure supported execution, proposal quality and latency on Hoffman2 before adopting it. Compare baseline and new proposer at the same precision when assessing algorithmic effects. If CPU generation misses the platform budget, obtain a suitable GPU allocation or revise decoding within observed limits. The outcome is a usable, measured configuration.

## Implementation and verification sequence

1. Worker and Reviewer agree the round contract. Record version identifiers for proposer, model revision, precision, prompts, feature schema and selector. Keep existing logs and checkpoint readers compatible.
2. Add the bounded proposer and meaningful diagnostics. Measure candidate validity, reference choice, order sensitivity, rationale grounding and latency on identity-rotated and order-reversed states. Include adversarial accusations and private-knowledge boundaries. Label development probes accurately.
3. Add a dedicated competition factory. It must activate the selected policy at every assigned seat and in tournament context, independently of focal-seat practice routing. Exercise both action and speech paths. Keep practice table behavior available for comparison.
4. Establish a frozen prospective comparison. The primary arms are old ID proposer plus round-one selector and new proposer plus the same selector. Hold model revision, precision, selector mode, opponent roster and execution budgets fixed. This estimates the effect of the proposer and tests transfer of the existing selector. Record assignment before roles/outcomes are inspected. Use arbitrary/rotated focal seats.
5. Run an initial official comparison batch through Slurm. Choose its size from actual throughput and feasible uncertainty, recording the choice before observing outcomes. Match counts are revisable execution choices. If a batch is too small to establish a win-rate difference, report uncertainty and continue the comparison within available resources.
6. If the new proposer is credible operationally, collect a separately versioned on-policy batch for selector adaptation. Compare its new checkpoint on fresh games with the unchanged-selector arm. Historical logs from different proposers never silently become the same on-policy batch.
7. Add a second frozen opponent mixture or population before claiming tournament-strength improvement. Rules-only results establish performance against that roster. Broader mixtures should include LLM or historical opponents when inference serving supports them.

## Acceptance evidence

- Legal/private-information constraints hold through the normal policy entry point. Verified seer results and known wolf teammates retain their role-appropriate treatment.
- Existing ID baseline and archived checkpoint still reproduce their documented behavior.
- New proposer outputs, parse failures and timeouts have explicit logging and a deterministic legal fallback. No valid action is inferred from malformed truncated output.
- The dedicated competition factory uses the configured model policy for any seat and tournament-marked action/message requests. Release identity and model readiness are checkable.
- All training, inference and gameplay runs are Hoffman2 Slurm work. Local activities are edits and inspection. Authentication and unrelated jobs are handled by the parent.
- Official prospective results include all assigned games, actual role and seat, arm, policy/proposer identity, terminal outcome or censoring reason, accepted delivery, latency and fallback. Per-game outcomes are the independent units.
- Report raw role counts and role-stratified/standardized outcomes using the platform's 4/7 villager, 1/7 seer and 2/7 wolf mixture where estimable. Keep no-shows, server incidents and policy failures visible.
- Promotion requires measured competitive benefit and reliable operation. A successful build, development probe or three-game winning streak alone does not establish it.

## Outputs and risks

Expected outputs are the contract, improved source, reproducible Slurm invocation, targeted verification, versioned comparison records and an independent review/closeout. Specific deployment activation remains contingent on candidate selection and tournament-context validation.

Generation latency, malformed JSON, evidence hallucination, positional bias, narrow opponent mixtures and scarce role coverage are the main risks. Diagnose these separately. A failed intervention should retain its evidence and leave the measured incumbent available.


## Reasoned-v1 evidence update

Partial development output from job 2111362 contains explicit errors binding private checks to player IDs, plus repeated investigation of an already checked villager. Longer preliminary text has therefore not established stronger play. The evidence compiler, source hierarchy, role-affordance priors and next tests are specified in `08_evidence_architecture.md`. Implementation of the next proposer remains a distinct experimental version. A parallel 14B CPU bfloat16 probe is a model-feasibility experiment; the current causal proposer comparison keeps the 8B model fixed.
