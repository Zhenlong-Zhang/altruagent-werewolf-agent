# WolfReasoner training and deployment evidence

## Implemented

The mixed table runtime can select one experimental seat and uses the same rule policy for the other six. It offers direct language model decisions, wolf-probability estimates with the same model and prompt, and practice-only transfer of frozen search or REINFORCE weights. The tournament entry continues to use the rule policy. The round used the research workflow and Hoffman2 skills. Planner wrote the verification contract, Worker implemented the simulator and trainer, and Reviewer independently replayed the frozen evaluation.

## Offline results

Hoffman2 job 1872013 completed in 8 seconds. The benchmark and weights were byte-identical to the initial run. REINFORCE optimized the terminal team reward over 2,400 simulated training games. Train, validation and test seeds were disjoint. Validation chose REINFORCE before the test run. There were 840 test seeds reused across three fixed opponent mixes, yielding 2,520 seat-games per method and 840 seed clusters for paired differences.

| Policy | Test wins | Test rate | Role-standardized rate |
|---|---:|---:|---:|
| Simulator rule proxy | 1,133 / 2,520 | 44.96% | 45.09% |
| Parameter search | 1,061 / 2,520 | 42.10% | 42.22% |
| REINFORCE | 1,135 / 2,520 | 45.04% | 45.17% |

The paired, role-standardized difference for REINFORCE was +0.084 percentage points with a 95% interval of −0.706 to +0.873 points. The simulator uses synthetic discussion, a proxy rule policy and simplified legal choices. In particular it omits abstention and voting for the wolf ally, and excludes public death evidence from scoring. This experiment does not train a language model or establish relative platform strength. The selector weights control a linear score over four observations, with parameters learned from simulated faction outcomes. See [Xu et al.](https://proceedings.mlr.press/v235/xu24ad.html) for the related approach combining language-model candidate generation with RL.

## Official Testing

- The first direct-model pilot completed with two model calls falling back to rules. It is an integration failure for the LLM path, not a model win. [Replay](https://platform.altruagent-game.com/replay/b820af51-5795-41f6-87fa-f5492c472009)
- A scheduled Hoffman2 practice runtime loaded the frozen REINFORCE artifact in seat 4, with rules in the other six seats. The focal seat was a villager and its faction won. The runtime delivered the structured result and was stopped after this match. Slurm job 1872250. [Replay](https://platform.altruagent-game.com/replay/86682c7a-2c1b-41fd-a1cd-f66c394c90ee)
- Subsequent gameplay, inference, simulation and training run on Hoffman2. Setup job 1872758 installed the CPU runtime and downloaded Qwen2.5-0.5B-Instruct.
- The model initially emitted free-form action responses. CPU inference now scores the offered action IDs directly, which constrains these decisions to legal choices. Discussion still uses rules when the response cannot be parsed.
- The direct model played seer and won in seat 1. The model made both action decisions. Its one discussion attempt fell back to rules. Slurm job 1873392. [Replay](https://platform.altruagent-game.com/replay/f46e3e78-90a4-410b-8a01-7798ebb3abfb)
- The belief-assisted model played seer and won in seat 5. The model made all three action decisions. Its one discussion attempt fell back to rules. Slurm job 1873501. [Replay](https://platform.altruagent-game.com/replay/d2978634-53bc-4a32-8495-dd84d7b357cb)
- Search weights loaded in seat 3. The focal seat played wolf and lost. Slurm job 1873774. [Replay](https://platform.altruagent-game.com/replay/5b46ec61-298f-41df-9988-d5c520166e4e)

The four Hoffman2 transfers are individual integration games. Their single-seat scores establish that the variants load, issue actions and receive the authoritative terminal result.

## Fresh official comparison

The five new games ran together on Hoffman2 job 1873986 after freezing the
assignment manifest. Each variant occupied seat 2 in one game, with rules
in the other six seats. The platform drew roles naturally. All 35 seats
received terminal results, with zero rejected actions in the five focal
logs. The runtime was stopped after settlement.

| Focal policy | Role | Result | Model actions | Discussion fallbacks |
|---|---|---|---:|---:|
| Rules | Villager | Loss | — | — |
| Direct Qwen | Seer | Win | 4 | 2 |
| Belief + Qwen | Villager | Win | 2 | 2 |
| Search | Villager | Win | — | — |
| REINFORCE | Wolf | Loss | — | — |

The two LLM policies delivered all six requested actions. The four
discussion attempts used rule fallback after model output could not be
parsed. Maximum recorded model-call latency was 21.941 seconds for direct
and 38.873 seconds for belief. This block tests the same seat and teammate
and opponent policy configuration. The focal roles differ across policies.
With one game each, the outcomes cannot rank the variants.
[Official results and replay IDs](../../outputs/official/heldout-1.json)

## Reproduction

Run `python scripts/offline_train.py --out-dir outputs/offline` in a CPU Slurm allocation. Compare the resulting files with [benchmark.json](../../outputs/offline/benchmark.json) and [weights.json](../../outputs/offline/weights.json). The files are synthetic aggregates and parameters. Raw platform logs remain private. The runtime installation requires the starter patch described in the README. Set `WEREWOLF_CANDIDATE=reinforce`, `WEREWOLF_CANDIDATE_SEAT=3`, and `WEREWOLF_WEIGHTS` to the artifact path, then run the starter with `--match --agent werewolf_agent.entry`.

## Remaining acceptance evidence

The first frozen model and evaluation package ships with rules as the
tournament default. A larger official comparison is still needed to rank
variants with role coverage, rotated seats, and multiple opponent pools.
The REINFORCE weights were trained on simulated games. Training on official
game trajectories and improving language generation remain separate
experiments.
