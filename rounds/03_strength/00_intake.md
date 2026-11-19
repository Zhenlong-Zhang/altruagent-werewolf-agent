# Round 03 — Measured improvements to Werewolf play

User instruction, 2026-11-19: “然后请你work来提升模型狼人杀的能力”. The objective is stronger tournament performance. The user permits stronger models and algorithm changes and has authorized Hoffman2 computation and the public project repository.

## Current evidence

The frozen Qwen3-8B ID-ranking proposer preferred the reference action in 20/42 controlled presentations. A Yes/No judge achieved 4/42. The initial 30 official games against six rules opponents produced a 20-parameter REINFORCE selector update. Three independent games validated loading and action delivery. A win-rate advantage has not been established.

The current proposer scores immediate action-ID tokens. The selector uses handcrafted features, and speech candidates are templates. Tournament contexts currently route to a rules fallback. Read the Round 02 literature/deployment review and preserve its frozen artifacts.

## Decisions and operating constraints

- Keep the existing proposer and checkpoint as controls. Evaluate changes prospectively and report delivery and latency alongside score.
- Run inference, gameplay, training and substantive checks through Hoffman2 Slurm. Local editing and metadata inspection are permitted.
- Preserve each acting seat's private-information boundary. A tournament entry must activate the selected policy for any assigned seat.
- Existing local edits to WORKPLAN.md and the untracked validation screenshot belong to the existing workspace and are outside this round's edits.
- Match counts, probe sizes and run order are execution choices. Revise them when evidence warrants. Record the reason for a change and avoid selecting a winner using operational checks alone.

## Skill Selection

- Final skills: round-based-research-workflow, ucla-server.
- Mode: project round with real Planner, Worker and Reviewer agents.
- Mapping: Planner = planner_strength; Worker = worker_strength; Reviewer = reviewer_strength. The tool provides task names as the available labels.
- Rationale: multi-file capability and serving changes require independent planning and review plus experimental evidence.
- Rejected alternatives: light continuity is insufficient for implementation and prospective evaluation; artifact-production skills do not cover this task.
- Main agent ownership: experiment orchestration, hybrid practice routing, scheduler jobs, platform games and synthesis.
- Worker ownership: proposal backend, HybridPolicy changes, dedicated competition factory and focused regression checks.
- Execution and evidence: completed experiment delivery. E trained the semantic actor/critic on30 games/136 choices; F evaluated53 games and G54. Final audits and independent review are recorded in04_review.md and05_closeout.md, with computed evidence in outputs/strength-round03. No demonstrated neural-policy strength gain; the incumbent remains selected. The user's broader competition-winning objective remains unproven.
