# Competition objective and joint LLM–RL policy

Optimize official faction score on seven-seat mixed tables. Round 01's 2,400 surrogate games trained a separate action scorer. Their weights and results are ineligible for the joint policy.

## Evidence translated to design

- [Xu et al., ICML 2024](https://proceedings.mlr.press/v235/xu24ad.html): use the same language model to propose a short slate, then learn which offer to play from faction reward. Train against an opponent population rather than identical copies.
- [Xu et al., ICML 2025](https://proceedings.mlr.press/v267/xu25h.html): start with a compact set of speech intents. Expand the set from match transcripts when it omits useful strategies. Consider preference fine-tuning only after separately measuring performance and collecting trustworthy preference pairs.
- [Bayesian Social Deduction](https://arxiv.org/abs/2506.17788): use confirmed seat-visible evidence as constraints and keep social evidence uncertain. The published experiment is Avalon, so validate this choice under Werewolf rules.
- [Exploring LLMs for Communication Games](https://arxiv.org/abs/2309.04658) and [Yang et al., EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1628/): test role-conditioned retrieval from development games and distinguish an accusation from the credibility of its author. Keep evaluation games outside any retrieval corpus.

## First trainable prototype

One LLM shortlists two legal actions or role-specific speech intents. The selector sees candidate features and their interactions with role, alive count and speech phase. Confirmed seer information constrains play. A fallback makes a deterministic rules decision if the model cannot shortlist. The first CPU model is for rollout plumbing and throughput measurement.

Collect one focal seat per game. Freeze proposer, initial selector, opponent population and log directory for a collection batch. Sample from the selector distribution and record the slate, probability vector, delivery acknowledgement and official faction return. Train one on-policy REINFORCE update per batch. Capture collection fallout and latency by role. Missing acknowledgements and mismatched policy/proposer versions are ineligible for this first estimator. Report censored games separately and review whether failures are policy-dependent.

## Competing for the tournament

Benchmark candidate acceptance, speech intent diversity, wolf-finding accuracy and tail decision time on role-balanced development states. Compare a stronger instruction model against the small CPU model on Hoffman2 under the same action deadline. Use a frozen population of distinct policies in training, then keep fresh opponent mixes for final play. Start with the compact intent set and test evidence tracking, stronger proposals, role-specific retrieval and free-form realization singly before combining gains. Protect against opponents that manipulate or flood chat by using verified history and bounded transcript context.

Evaluate four controls on fresh matches: rules, the same proposer with LLM-first selection, the initial selector and the trained selector. Label stochastic and greedy policies separately. Match up opponent teams and rotate seats and roles where the platform allows. Report official faction score by role, overall score standardized to the tournament role mix, model fallback, failure rate and latency. Promote weights only when the held-out outcomes and delivery support them.

Development and evaluation gameplay and inference run on Hoffman2 through Slurm. This round uses the research workflow, with independent implementation and review. The final competitive assessment remains open until real collection and evaluation are complete.
