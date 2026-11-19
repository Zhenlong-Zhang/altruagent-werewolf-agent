# Execution decisions and provenance

## Phase A: development diagnostics

- Job 2111205 failed fixture construction before model inference. Missing required SDK action input fields were corrected.
- Job 2111362 runs isolated source based on50a4fb0, Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218, CPU BF16,4CPU48GB. Its CPU FP32 control reproduces all42 archived CPU rank lists exactly(21/42 preferred); the earlier GPU BF16 control was20/42.
- At matched CPU BF16, legacy IDs scores20/42 historical and27/42 fresh scored presentations. The fresh first-option control scores31/42. Fresh unlabelled claim cases are excluded from accuracy.
- Reasoned-v1 development cases reveal private-fact inversion and redundant investigations, independently audited in04_review.md. It will not be promoted or used to collect new RL data on this evidence. There have been no new official games in this phase.
- Job 2111774 screens Qwen3-14B revision40c069824f4251a91eefaf281ebe4c544efd3e18, CPU BF16,4CPU80GB, same30second soft budget. IDs scores10/42 historical and12/42 fresh scored presentations. Preliminary reasoned proposals exceed budget. The diagnostic was stopped early and partial results retained; this rejects the tested configuration, not every possible14B setup. It is not a matched model-size experiment for game strength.
- Job 2112497 passes4 accounting checks. The completed-reason allowlist was subsequently tightened and awaits rerun with the next source snapshot.

## Phase B: grounded revision

The reasoned-v1 result changes the next intervention. Add explicit evidence/target binding, role-aware candidate annotations, independently versioned seer constraints and a one-pass proposer. Preserve IDs and reasoned-v1 controls. Run raw proposal diagnostics separately from policy guard checks. Additional ID/target permutations test that the model follows targets rather than numeric identifiers.

Frozen comparison will use IDs versus grounded, identical model revision/precision and old selector checkpoint, with arm and focal seat hashed from session ID before seeing role/outcome. Both arms use the exact competition factory at the focal seat and six fixed rules opponents. Only the grounded arm receives its versioned evidence guard: this is a whole-policy capability comparison, not an isolated prompt-only effect. Guard-specific diagnostics report that intervention separately.

The earlier proposed40completedgames/arm is superseded before any new match creation. Initial target is12completedgames/arm, chosen for observed platform throughput and a feasible live allocation. This is a pilot with wide uncertainty; it cannot establish tournament superiority. Retain every controller creation receipt, worker failure and unresolved assignment. Report roles/seats and Wilson intervals; no promotion based solely on unadjusted aggregate wins.

No mixed-proposer greedy evaluation logs will become an on-policy training set. A subsequent adaptation cohort, if justified by operation and quality, must use sampled choices, its actual generator identity and a separately recorded checkpoint lineage.

Phase B submitted as job2113134:8CPU64GB3hours, Xeon6family, source archive SHA2569ef888f64b47b33be5b2f9a6a0090a6568fbb3c809faad4f0083f25e271b7f6e. Source fingerprint is outputs/strength-round03/phase-b-source.json. Benchmarks explicitly include mapped suite. All23focused+6comparison+4accountingchecks run before model staging. Arm seed r03-grounded-v2, old selector d7ebf7670444067a, CPU BF16 pinned8Brevision unchanged, exactcompetitionfactoryatonefocalseat. No official match created at submission.

Phase C diagnostic job2113721 uses a separate code snapshot (phase-c-source.json),8CPU48GB1h, no official runtime. It passes28focused+6comparison+4accountingchecks. The preselected development suites isolate fullcyclic meanofferedlogp vsnone. Social changes remain design-only.

A separate full-context latency diagnostic runs as an overlapping Slurm step inside2113134 while allfour existing matches have completed and newcreation is paused. It imports phaseCcode fromr03c, uses the allocation8CPU and its staged8B weights, and includes six-target votes, seven offered legal moves including abstain(supportedbound; normalproduction excludesabstain), and actualfive-optionwolf speech withten240char messages. Onepass acceptsallthree at7.761/4.955/4.821seconds. Cyclic acceptsK6at28.797andK5at23.828seconds, butK7rejectsat32.929seconds aftersixpasses; no partial rank used. Seven-optionbudget failure and narrowK6margin preclude blanketcyclicdeployment undercurrent30secondbudget. Artefact outputs/strength-round03/wide-2113134.json.

Controllerreceipt4was recorded on a delayedsnapshot that already showedcompletion. Allfourcreationrequests are retained; no outcomebased exclusionoccurred, and runtimeassigned arm/seat beforeobservations. ThislateUUIDreceipt is flagged in ignoredcontrollercensus; do not describeeverycontrollerUUIDrecord as preoutcome.
