# Semantic PPO execution amendment

Frozen before any Phase F heldout match, 2026-11-19.

Phase E remains unchanged: job2116767 collects30 eligible completed sampled games under the legacy IDs proposer, seer guard and parent d7ebf7670444067a. The earlier REINFORCE child is still produced and independently reproduced as a reference. Its training outcomes are not evaluation results.

The pending four-profile evaluation job2116775 was cancelled **before execution and before any F match**. A new isolated source snapshot will replace its adapted arm with the semantic attention/PPO policy specified in14_semantic_ppo_contract.md. It uses exactly the same frozen E cohort and original behavior parent, rather than the REINFORCE child. The sampled control and adapted arm share proposer, evidence guard, mode, opponents, precision and thread limits. Their contrast estimates the complete semantic-PPO policy change; it does not separately identify architecture versus optimizer effects. The other two controls remain incumbent greedy and guarded greedy. Namespace wolf-policy-v1 and seed r03-ids-guard-heldout-v1 remain unused by any training cohort.

Final preimplementation choices clarify the contract: entropy coefficient0.001; actor loss is a sum over eligible sampled decisions within each episode, then a mean over episodes; fixed unnormalized advantages use the terminal faction score and the frozen old critic. Critic MSE is a mean over decisions. Thirty episodes remain30 terminal outcomes. Full actor, critic and optimizer snapshots are restored on a KL-bound violation, and training stops at the previous accepted state.

The frozen MiniLM encoder represents candidate labels and coarse contemporaneous context only. It does not recover unlogged conversational history, train Qwen weights or implement centralized multi-agent PPO. New code and staged encoder are isolated from the active E writer. H2 regression checks, propensity parity, artifact integrity and actual cold/warm runtime checks must pass before a heldout match is created. No F result may choose the learning rate, epoch count or encoder.

The12-per-arm target is a pilot for policy selection, not a definitive tournament win-rate estimate. Keep all receipts, failures and role/seat counts. Training loss, KL movement or synthetic correctness cannot justify promotion without prospective game evidence.
