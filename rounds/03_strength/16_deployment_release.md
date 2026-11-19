# Competition release route

Keep the competition incumbent until the fresh four-profile comparison supports a change. The semantic checkpoint is a trained candidate, not automatically the selected release. Training and serving use separate output directories and allocations.

The official starter's current CLI supports `--tournament` for tournament assignments, `--match` for Testing and both flags in one process. Use `werewolf_agent.competition_entry` for the contestant. Practice table factories deliberately fall back to rules for tournament contexts and must not be deployed as the competition agent.

The tested official starter is commit be680b1435c12569beb8ea1d239538efff8d8f82 with the two checked-in settlement/message-result callback patches applied. [SDK provenance](../../outputs/strength-round03/starter-runtime-source.json) records the commit, patch hashes and modified-file hashes. Preserve those callbacks when reproducing delivery and terminal-result accounting.

Freeze these fields together before the competition window:

| Component | Verified setting |
|---|---|
| Language model | Qwen/Qwen3-8B, revision b968826d9c46dd6066d109eabc6255188de91218 |
| Runtime | Python3.12.13, Torch2.7.1 CPU, Transformers4.51.3 |
| Precision / threads | BF16 model inference,4 threads per HF process |
| Incumbent selector | schema1 checkpoint d7ebf7670444067a |
| Semantic candidate | schema2 checkpoint6ca4df8fd98821e6, [artifact](../../outputs/strength-round03/job-2123930/selector.json) plus adjacent selector.pt |
| Semantic encoder | all-MiniLM-L6-v2, revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41; staged files must match checkpoint hashes |
| Candidate generator | legacy IDs; cyclic aggregation off |
| Guard and sampling | Choose the complete evaluated profile; preserve its guard and greedy/sampled mode together |
| Deadline behavior | Semantic stage checks remaining30s, capped5s, with logged legal fallback; these are posthoc checks, not hard interruption |

Inside a running compute allocation, stage model files and install the pinned environment. Set `WEREWOLF_MODEL_PATH`, `WEREWOLF_MODEL_DEVICE=cpu`, `WEREWOLF_MODEL_DTYPE=bfloat16`, `WEREWOLF_MODEL_REVISION`, `WEREWOLF_CPU_THREADS=4`, `WEREWOLF_HYBRID_WEIGHTS`, `WEREWOLF_HYBRID_MODE` and `WEREWOLF_EVIDENCE_GUARD`. For schema2 also set `WEREWOLF_SEMANTIC_ENCODER` to the staged local encoder directory. The training artifact's temporary encoder path expires when its allocation ends, so this override is required on a new host. Keep candidate generation and model identities identical to the selected release.

Source the private official-key environment only within the allocation; do not put credentials in Git or a release manifest. From the official starter checkout, the serving command is:

```bash
python -m agent --tournament --agent werewolf_agent.competition_entry
```

Maintain one runtime per Official Agent Key. End the Testing runtime before starting the tournament runtime. Slurm jobs continue independently of the SSH terminal, but allocation expiry ends the agent. The allocation must already be running, connected and ready before the announced tournament deadline. A queued job is not an available agent.

The measured8CPU64GB setup supports the tested one-game roster with two actual8B HF processes. The synthetic readiness probe had6 calls per process with no fallback; maximum full-call times were17.015s direct and9.030s semantic. The live Testing canary confirmed model selection and5/5 platform acknowledgements. This does not establish concurrent tournament capacity, mid-game reconnect recovery or actual tournament-context connectivity. Verify those against the tournament schedule and simultaneous-assignment rules before final entry; no tournament registration or competition connection was performed by this experiment.
