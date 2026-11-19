# Competition architecture and deployment review

Reviewed on 2026-11-19. The objective is official tournament score and held-out playing strength. The present evidence is one 30-game Qwen3-8B collection, one selector update and three operational checks. Choosing the competitive model and policy remains an empirical decision.

## Paper findings and proposed experiments

| Primary source | Finding relevant to this project | Proposed experiment |
| --- | --- | --- |
| [Xu et al., ICML 2024, Strategic Language Agents](https://proceedings.mlr.press/v235/xu24ad.html), [full text, sections 4, C.1, C.3 and D.5](https://arxiv.org/html/2310.18940) | The method combines role deduction, diverse language candidates, and an attention-based selector trained with MAPPO against a policy population. The selector embeds both observations and candidate text. D.5 reports transfer of a frozen selector to three different LLMs. | Compare a semantic candidate selector with the current feature selector. Use varied frozen teammates and opponents. Test frozen-selector transfer and adaptation separately when changing the LLM. |
| [Xu et al., ICML 2025, LSPO](https://proceedings.mlr.press/v267/xu25h.html), [full text, section 3](https://arxiv.org/html/2502.04686) | LSPO clusters generated discussion into latent strategies, optimizes an abstract game using CFR, and uses DPO to improve the language model and expand strategies over iterations. | First expand and measure distinct speech strategies. Evaluate a structured intent policy with contextual language realization. A fuller CFR/DPO implementation needs a validated abstraction and preference data. |
| [Xu et al., Exploring LLMs for Communication Games](https://arxiv.org/html/2309.04658v2), sections 3–4 | Retrieval and reflection provide a learning mechanism with frozen LLM weights. Increasing experience-pool size did not yield a consistently increasing win rate. | Retrieve a bounded set of role-relevant development experiences. Compare retrieval enabled and disabled. Keep final evaluation games outside the retrieval corpus. |
| [Rahimirad et al., Bayesian Social Deduction](https://arxiv.org/html/2506.17788v2), sections 3–5 | GRAIL combines structured probabilistic inference with language-derived priors. Its experiments concern Avalon. | Adapt constraints to our Werewolf rules, track evidence provenance and compare calibrated beliefs with the current suspicion heuristics. |
| [Yang et al., EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1628/) | Their benchmark finds sensitivity to accusations and to trust in the accuser, including for larger models. | Add tests involving false seer claims, repeated unsupported accusations and contradictory statements. Check source credibility alongside accusation content. |

The paper configurations include roles and rules that differ from AltruAgent. Both ICML papers use a Doctor. Our platform has two wolves, one seer, four villagers, a peaceful first night and bounded discussion. Those differences affect transitions, information and strategy value. Rule details currently come from the checked-in platform guide `werewolf.md`; the live game-rule URL could not be fetched during this review.

## Gaps in the current implementation

- `variants.py` ranks offered action IDs with next-token scores. `hybrid.py` uses a compact, manually specified speech set. The current proposal mechanism therefore has limited scope for discovering new language strategies.
- `selector.py` learns 20 coefficients over eight features and their interactions. It receives candidate rank, rule evidence and context features. Semantic observation/action embeddings and an attention-based actor–critic remain proposed components.
- `train_hybrid.py` implements one on-policy REINFORCE update. MAPPO, behavioral cloning, a learned critic, population training and LSPO preference optimization are further experiments.
- The completed batch used six rules opponents. Broader teammates, adversarial language and different frozen policy versions require separate collections and evaluations.

Our present implementation is a trainable prototype inspired by candidate selection. The paper transfer result motivates testing existing weights on a different LLM. Transfer performance for our feature-based interface is still unknown. For each new proposer, compare unchanged weights, an adapted selector and the LLM's own selection under the same evaluation conditions.

## Competition-oriented development order

1. Preserve the present checkpoint as a reproducible control. Evaluate it against LLM-first and the initial selector on new games, including stochastic and greedy selection. Rotate focal seats and record actual role assignments.
2. Introduce a fixed population of distinct frozen policies. Include conservative, aggressive and deceptive language styles and historical checkpoints. Reserve additional opponent mixtures for final evaluation.
3. Screen stronger LLMs on deduction, candidate diversity, legal delivery and latency. A larger model can be selected when the complete system performs better within the platform deadlines. Each candidate gets a versioned proposer configuration.
4. Add semantic candidate representations, explicit evidence tracking and richer speech generation as separately measured changes. Start with a small contextual selector and an actor–critic/PPO experiment when rollout volume supports it. MAPPO requires a deliberately defined joint collection protocol.
5. Investigate LSPO-style strategy expansion and preference training if candidate coverage remains a dominant failure. Its counterfactual game-solving approach requires suitable simulation or abstraction support. The live platform trajectories alone provide no state-forking interface for CFR traversal.

Report wins and uncertainty by role, the competition role mixture, latency tails, failed deliveries and no-shows. A larger held-out batch should determine promotion. The three completed validation wins establish operation. Comparative strength remains open. The final language model, selector, sampler and prompts will be selected together.

## Deployment route

The current registration uses Self-hosted. The [official agent guide](https://platform.altruagent-game.com/tournament/agent-guide) documents running the starter continuously with `--tournament`, maintaining one runtime per Official Agent Key, and connecting within the tournament deadline, four minutes by default. It also documents the platform-hosted Markdown option. Self-hosting accommodates our Python policy and checkpoint directly.

The proposed serving arrangement is an official starter supervisor with independent per-game workers, a warmed language-model service, a selector, and a legal-action fallback. Shared model weights can reduce repeated loading. Match and seat memories remain isolated. Each model request includes only the acting seat's authorized information. The inference service should maintain no shared conversational history across seats.

For Hoffman2, the supervisor and inference process run inside an active compute allocation covering the announced tournament window. The allocation must be running and the model ready before the games start. The measured Xeon6 CPU configuration is a viable candidate for the present 8B model. Larger models need their own measured allocation and latency evidence. A continuously available host is an alternative if scheduled cluster coverage cannot be secured, with hosting costs and availability considered before provisioning.

Training stays separate from the serving release. Freeze a release manifest containing the model and tokenizer revision, precision, prompt and candidate schema, feature or embedding version, selector hash, sampling mode, retrieval-corpus version, code commit and deadline/fallback settings. Preserve the selected distribution when stochastic play wins the comparison. Choose greedy play through the same comparison process.

## Deployment work still required

The current experiment entry is practice-only. Code inspection found three relevant paths.

- `hybrid_entry.py`, `HybridTable._policy`, selects `WerewolfPolicy` when `context.tournament_id` is present.
- `hybrid.py`, `choose_action`, delegates tournament actions to the rules parent.
- `hybrid.py`, `choose_message`, also delegates tournament discussion to the rules parent.

A dedicated competition factory should construct the selected policy for every assigned contestant seat, with tournament-compatible action and message handling. It should support the chosen model backend and verify the release manifest at startup. The practice table assignment and opponent configuration remain useful for evaluation.

Before registering the final release for play, verify tournament-context routing, arbitrary seat positions, checkpoint identity, model readiness, action/message delivery, reconnect behavior and concurrent assignments. Derive a per-decision inference budget from the remaining platform time, validate the chosen action against fresh legal actions and use a measured legal fallback on inference failure. The final canary should use the exact competition factory in Testing. An actual tournament-context connection is a further verification when such an assignment is available.

## Decisions and open items

- Adopt a richer deduction/candidate/selection interface as an experimental direction. Promotion depends on measured playing strength.
- Include zero-shot selector transfer as an experiment. Adaptation is an additional candidate.
- Use Self-hosted Python deployment as the working route. The final runtime host, language model and sampling policy remain to be selected.
- Confirm the tournament schedule, maximum simultaneous assignments, available serving resources and permitted inference services before sizing the release.
- This review changes the research and deployment plan. The competition factory, semantic selector and expanded training have implementation and validation work remaining.

## Workflow record

The round-based-research-workflow skill was used in light continuity mode for a literature review and deployment audit of the existing round. The main agent reviewed primary papers, the live platform agent guide, local entry points and completed-run metadata. This task produced a plan and code findings. It involved no new experiments or deployment. The user-modified `WORKPLAN.md` was consulted as context. Evidence and remaining work are recorded above.
