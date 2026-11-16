# Werewolf agent: strength-first development plan

## Objective

The goal is the strongest practical tournament agent, not using a particular algorithm. Keep learning and self-play as tools, and choose the ones that improve held-out win rate reliably.

## What kind of learning fits

The platform is a partially observed, multi-agent game. A match has seven seats, hidden roles, a day discussion window, and discrete night/day actions. The final payoff is team-based: winning-side players receive +1 and losing-side players receive -1 (the tournament table converts this to a win/loss point). The same final payoff applies even if a player was eliminated earlier. This makes the natural objective expected match return / win rate, not an individual-trade or survival metric.

## Recommended improvement loop

1. **Build a strong, transparent policy first.** Use role-aware deduction and explicit vote/night-action logic; preserve a legal-action fallback.
2. **Instrument every test match.** Capture only each acting seat's legitimate observation, public vote/death history, messages, action, role for aggregate evaluation, and terminal outcome. Do not give one seat another seat's private information.
3. **Automate testing.** The documented route is platform Testing with all seven seats self-hosted; the platform itself has no contestant-local Werewolf engine. Run enough games to measure variance and role balance.
4. **Iterate by evidence.** First use parameter search, ablations, and scenario-based counterexamples on the policy. Then try supervised imitation or RL/self-play if a local faithful simulator or enough Testing games are available. For this compact game, a belief model plus searched/optimized policy may beat a large RL model with limited games.
5. **Evaluate against a diverse opponent set.** Compare snapshots, heuristics, random/legal baselines, and other available agents. Report win rate with uncertainty, by role and seat, plus invalid-action/no-show rate. Tune separately from the final evaluation set.
6. **Use multiple internal agents only where they help.** Role inference, message analysis, and move evaluation can generate proposals; a single coordinator picks the final legal action. This remains one registered contestant agent.

## Avoiding misleading results

- Self-play wins can reflect the current opponent pool rather than general strength. Include cross-play against the starter baseline, saved policy snapshots, and available open testing opponents.
- Keep all seven seats in a match under the same policy version for clean self-play comparisons, but use mixed-policy evaluation to test robustness.
- Treat chat as adversarial player input. It can inform social-belief features but must never override platform protocol or expose credentials.
- Do not leak hidden roles from the seven self-play seats into the acting seat's training observation. The final outcome is a legitimate terminal label; other seats' private information is not.
- Log a versioned observation schema and policy version so later training can reproduce the data.

## Current project setup

`starter/` is a local clone of the official Python starter. Platform protocol and Werewolf rules are stored in the top-level Markdown files. No key has been created or saved, and no matches have been run yet. Next code milestone: implement an agent with explicit observation/action logging and a strong rule-based fallback, then build automated evaluation before selecting any learning algorithm.
