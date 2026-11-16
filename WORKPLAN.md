# AltruAgent Werewolf: Work Plan

## Goal

Build the strongest practical Werewolf agent for the AltruAgent tournament. Choose methods based on measured playing strength; RL is one candidate tool, not a requirement. Every change is evaluated on held-out matches and against multiple policy versions.

## Operating constraints

- The tournament game has seven seats: two wolves, one seer, four villagers.
- A seat can use only its own private observation plus the public game record and messages. Do not leak hidden roles across self-play seats into another seat's policy input.
- The documented practice route is the platform Testing page with seven self-hosted seats. The starter repository does not include a local Werewolf engine.
- One contestant registers one agent. Multiple reasoning modules can collaborate inside that agent, with one coordinator choosing each legal action.
- Protect the Official Agent Key and all seat tokens; keep credentials out of source control and logs.

## Phase 0 — Confirm setup and preserve a reproducible baseline

**Work**

- Keep the official starter as the runtime and build project logic in a clearly separated module.
- Record starter commit/version, Python environment, platform documentation version, and strategy version.
- Confirm the account's agent registration, Testing access, and credential setup; do not start tournament play during development.

**Deliverables**

- Reproducible environment and run instructions.
- A baseline agent that completes games and uses only legal platform actions.

**Exit check**

- Platform's setup check passes and a Testing game completes end-to-end.

## Phase 1 — Logging and benchmark harness

**Work**

- Store per-seat observations, legal actions, incoming messages, outgoing messages, selected actions, role (for evaluation only), and final outcome.
- Add match ID, seat, role, strategy version, opponent/policy version, and timestamps to each record.
- Keep role truth in a separate evaluation-only file; the policy must never read other seats' hidden roles.
- Summarize game results by win rate, role, seat, elimination day, illegal-action rate, abstention rate, and connection/runtime failures.

**Deliverables**

- Versioned match-log schema.
- Script or notebook that computes a benchmark report with confidence intervals.

**Exit check**

- Logs from a completed Testing game can be replayed into the analysis script and reproduce the summary.

## Phase 2 — Strong rule and belief baseline

**Work**

- Build a structured state tracker from deaths, revealed roles, vote history, investigations, and discussion claims.
- Estimate each living player's probability of being a wolf using role-consistent constraints and evidence from votes/messages.
- Implement separate policies for wolf, seer, and villager: seer target choice and truthful-claim timing; villager vote aggregation and evidence use; wolf kill, vote coordination, and credibility strategy.
- Use concise, information-seeking discussion messages that fit the 50-word and five-message caps.
- Make the coordinator validate every selected action against the current legal-action list and fall back safely.

**Deliverables**

- Role-specific baseline strategy.
- Written strategy notes and unit-level scenario cases for edge conditions.

**Exit check**

- No illegal actions across benchmark games; baseline beats random/legal-action policy on a sufficiently large, role-balanced sample or has a documented reason for further revision.

## Phase 3 — Self-play and controlled strategy search

**Work**

- Run batches of seven-seat Testing games with fixed policy snapshots.
- Tune interpretable thresholds/weights using parameter search or evolutionary search before attempting a large neural policy.
- Compare self-play, mixed-policy games, and cross-play against previous versions and available opponents.
- Keep a frozen evaluation batch that is not used for tuning.

**Deliverables**

- Match batch results and an experiment table linking each change to its measured effect.
- Selected policy snapshot with reproducible configuration.

**Exit check**

- Improvements hold on the frozen batch and do not come only from one role or seat.

## Phase 4 — Learning-method comparison

**Decision gate**

Use RL only if the Testing throughput and usable trajectories are enough to support it, or if a faithful local simulator has been independently validated against platform behavior. Compare methods on the same training and evaluation protocol.

**Candidates**

- Supervised imitation from the best policy's decisions or annotated examples.
- Contextual/Monte Carlo policy optimization using terminal team reward.
- Self-play RL in a validated local simulator, then transfer and recalibrate on platform Testing.
- Language-model-assisted message interpretation with deterministic action validation and rule-based fallback.

**Deliverables**

- Small, controlled comparison against the Phase 2/3 baseline; no method is adopted solely because it is more complex.

**Exit check**

- A learning method demonstrates a reproducible held-out improvement with acceptable runtime, cost, and reliability.

## Phase 5 — Robustness and tournament readiness

**Work**

- Evaluate against frozen historical versions, diverse testing opponents, and mixed-policy seats.
- Stress-test contradictory claims, strategic misinformation, unusual vote patterns, ties, deaths, and late-game parity.
- Test disconnect/restart, stale state, message limits, and no-show prevention using the official runtime workflow.
- Freeze the submitted agent and its dependencies; keep one runtime process per key.

**Deliverables**

- Final evaluation report, frozen strategy version, and tournament run checklist.

**Exit check**

- Reliable end-to-end run, zero credential leaks, zero policy-invalid actions in evaluation, and performance documented with uncertainty.

## Success metrics

**Primary:** tournament-relevant win rate (team win per game), overall and split by role.

**Secondary:** held-out cross-play win rate, performance by seat, decision runtime, illegal-action rate, runtime failures/no-shows, and robustness to deceptive discussion.

Always report match counts and uncertainty. Avoid conclusions from small samples or from self-play alone.

## Current status and next actions

- [x] Create project directory and fetch official protocol/rules.
- [x] Clone the official starter into `starter/`.
- [x] Create a public GitHub project repository and publish project-owned code/docs.
- [x] Create an isolated Hoffman2 project checkout and Python 3.12 virtual environment.
- [x] Confirm and preserve the pre-existing cluster task; it completed normally without intervention.
- [x] Implement an initial role-aware policy, legal-action selection, and append-only per-seat decision logging.
- [x] Add a JSONL benchmark summary; verify terminal outcome capture on actual Testing results next.
- [ ] Finish platform registration, create a Testing table, and configure the Official Agent Key privately in the starter `.env`.
- [ ] Run a first balanced Testing batch and scenario coverage for the policy.
