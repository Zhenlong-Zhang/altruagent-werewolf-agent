# Round04 — strategy architecture review, then fresh training and evaluation

User: “继续work，改进候选动作和交流策略，再训练和测试。目标还是让agent足够强，最后能赢比赛。” Latest steering: “开始work前先检查看看做法是不是有什么问题。然后你可以去网上抄一个别人的模型那个那么多paper。”

Interpretation: review current method and primary papers/open implementations BEFORE implementation; reuse applicable open-source components with provenance/license, then train/evaluate adapted agent on fresh data. Winning strength is the objective, not completing a fixed number of parameter updates.

Baseline: commit fe359e42053cce7edd194d2a6d8f9bf9ede013e3, D7 IDs/greedy/no guard retained. R03 NN trained on30 terminal observations,136choices; G changed1/98 eligible choices, no demonstrated gain. Prior F/G outcomes are development evidence for this new round, never its final evaluation.

Constraints: preserve acting-seat private information, real platform rules/limits and old artifacts. Compute/inference/tests/gameplay on Hoffman2 Slurm; Mac source edits, metadata inspection and transfers. Leave user WORKPLAN.md edits and old validation-screenshot.jpg untouched. No disruption of existing jobs. No new code implementation until architecture review/contract accepted. No credentials in files or agent prompts.

## Skill Selection
- Candidate/final skills: round-based-research-workflow + ucla-server.
- Operating mode: substantial project round, real Planner/Worker/Reviewer.
- Mapping: Planner=planner_strength, Worker=worker_strength, Reviewer=reviewer_strength (existing agents reused).
- Rationale: source-level architecture audit, paper/code fit assessment and new training/evaluation require independent roles.
- Rejected alternatives: repeated small-head tuning ignores observed action/candidate bottleneck; paper name alone is not evidence of a faithful or effective reproduction.
- Execution result/evidence: method/literature review completed before implementation; opt-in B/C source implemented, independently reviewed and verified by Slurm2189749 (seven suites/51 synthetic/SDK correctness cases). See08_execution.md for preserved failures and exact claim boundaries. New real-model gameplay/training remains pending GPU capacity and measured service readiness; winning strength is not established.
- Resource decisions/sample counts remain agent-proposed until independently justified and frozen before outcome inspection. User authorizes substantial server use; scheduler maxima are not a mandatory consumption target.
