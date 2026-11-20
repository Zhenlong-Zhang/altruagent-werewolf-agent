# Primary literature and available reuse

Reviewed2026-11-20 using primary papers, author-linked repositories and local read-only source copies. Findings are distinct from proposed adaptations. No paper result establishes performance on AltruAgent.

## Closest architecture: strategic language agents

[Xu et al., ICML2024](https://arxiv.org/html/2310.18940), sections4/C: hidden-role deduction, diverse generated language actions and a state/action attention policy trained in a population. Their state includes information record and deduction, unlike our coarse PPO context. AppendixC reports behavior cloning then10M MAPPO steps, four learners with three sampled population players, three candidates,1536-wide encoders and12 attention heads; gamma and GAE lambda are0.95. This is a different scale and information interface from30 terminal outcomes.

The [author homepage](https://zelaix.github.io/) links the actual [SLA repository](https://github.com/zelaix/SLA), commit `c81f4500b0be74e4a270370ba35bb94caceb915b`, MIT. It contains environment/agents, attention actor–critic, runner and training entry points. Its README is an old Python3.6/Torch1.5 setup; GPT/ada API dependencies and Doctor rules need replacement. No ready AltruAgent checkpoint was verified. Reuse the MIT learning machinery with attribution and a new platform adapter; do not execute its environment unchanged or label our adapter an exact reproduction. The separately inspected [MAPPO upstream](https://github.com/marlbenchmark/on-policy) is MIT and provides return/mask/value-learning reference code.

## Strategy expansion: LSPO

[Xu et al., ICML2025](https://arxiv.org/html/2502.04686), section3/appendixB: cluster discussion into latent strategies, solve the abstract game with Deep CFR, then DPO-expand the generator. Their construction collects1000 games with three candidates; embeddings are1536-dimensional. Deep CFR uses1500 training iterations and a500k role buffer; DPO uses two epochs. Our live platform exposes no counterfactual state-forking/traversal interface, so replaying recorded outcomes is not CFR. Doctor and other transition differences matter. Author/publisher pages reviewed did not supply a verified LSPO code/checkpoint release. Reuse the abstraction/strategy-expansion idea, not fictional pretrained weights; full reproduction is deferred.

## Most concrete tactical-policy release: ONUW

[Jin et al., NeurIPS2024](https://arxiv.org/html/2405.19946v2), sections5/E: visible history plus belief embeddings condition a CQL discussion policy. Six tactics cover honest/deceptive evidence, accusation and defense. The reported dataset is120 GPT4 games/1800 discussion transitions; discussion-policy training took about2.5 hours on a3060Ti. Those modest counts do not guarantee adequacy for our different game.

The [official project](https://one-night-ultimate-werewolf.github.io/) links [KylJin/Werewolf](https://github.com/KylJin/Werewolf), commit `4daf873e83fe1e9b029cb3d39c7e57f3baa5da10`. Actual source includes `training/train.py` (d3rlpy DiscreteCQL) and a roughly16MB `onuw/agents/models/discussion_policy.d3` checkpoint. Thus a trained policy exists. It expects1536-dimensional OpenAI ada embeddings of history/belief and ONUW role-swapping dynamics; replacing them with384-dimensional MiniLM or merely padding vectors is not valid transfer. No repository license file or API-recognized license was found. Keep it as a read-only reference; implement ideas independently or clarify license before copying code/weights. [d3rlpy itself](https://github.com/takuseno/d3rlpy) is MIT. ONUW has one night, role swaps and one final vote; AltruAgent has repeated eliminations, fixed roles and faction parity, so its simulator/checkpoint is not a drop-in agent.

## Actual reusable strategic LLM: MARSHAL

The author-linked [MARSHAL code](https://github.com/thu-nics/MARSHAL) and [generalist Qwen3-8B weights](https://huggingface.co/nics-efc/MARSHAL-Generalist-Qwen3-8B) are Apache2.0. Pin weights to `369d9decea6e8b347f81706fa169dbbddddce360`. The model card says it was initialized from Qwen3-8B and self-play trained on Tic-Tac-Toe, Kuhn Poker and Mini Hanabi. Its turn-level credit assignment and agent-specific normalization address multiaction games, but Werewolf is not its stated training task. This is an actual RL-trained model worth an equal-interface vanilla8B comparison, not an available Werewolf champion. Measure memory/decode deadlines on allocated GPUs before selecting it; inference transfer does not require reproducing its full token-RL training stack.

## Token adaptation alternative: MaKTO

[MaKTO paper](https://arxiv.org/html/2501.14225), appendixE, reports25k SFT samples and20k preference entries;14B training used eight A100s and72B used32. This is not a two-GPU reproduction budget. The [author code](https://github.com/ReneeYe/MaKTO-Werewolf), commit `32a57f04b371e5c88c4d77422f175e5e899fed9f`, contains preference-extraction scripts and says model release is future; no released MaKTO14B/72B weights were verified. Its model-license string is written `CC BY-NA-SA4.0`, which is insufficient to assume an applicable standard permission.

The actual [reasoning dataset](https://huggingface.co/datasets/ReneeYe/werewolf_game_reasoning) is public, ungated and card-marked MIT; metadata revision `7d7d516f780879dbe9f512ef29655f51548f7578`. It has instruction/prompt/response/meta fields and English/Chinese plus action/speech/vote splits. Those are potentially reusable training data, not pretrained weights. Nine-player examples include Witch/Guard roles; rules, language and acting-seat information require filtering, and game-level splits must precede LoRA training. This is a later generator-adaptation option after the initial released-model screen, not a reason to spend scarce rollouts reproducing irrelevant roles.

## Decision and nonclaims

Recommended near-term route: released strategic8B versus vanilla8B, meaningful evidence-conditioned candidates and responsive communication, then a properly contextual multibatch RL policy using attributed MIT components. Full LSPO requires counterfactual infrastructure; ONUW checkpoint transfer requires matching embedding/rules and license; MaKTO-scale full training is not established feasible. Paper strength, network size and training completion are not promotion criteria. New prospective AltruAgent games against varied frozen policies must decide playing strength.
