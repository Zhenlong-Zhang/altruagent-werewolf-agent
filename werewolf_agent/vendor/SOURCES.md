# Small attributed algorithm adaptations

Both upstream repositories distribute these components under MIT; their
copyright and license are retained verbatim in `MAPPO_LICENSE`.

| Local file | Upstream pinned source and ranges | Adaptations |
| --- | --- | --- |
| slot_attention.py | zelaix/SLA, commit c81f4500b0be74e4a270370ba35bb94caceb915b, onpolicy/algorithms/mat/algorithm/ma_transformer.py lines19–84 | Q/K/V attention and postnorm residual block; variable key/query lengths and explicit key/query padding, single acting seat only, removed joint-agent decoder and causal buffer. |
| event_ppo.py | marlbenchmark/on-policy, commit de66d7a4b23fac2513f56f96f73b3f5cb96695ac, onpolicy/algorithms/r_mappo/r_mappo.py lines52–87,119–142; onpolicy/utils/shared_buffer.py lines185–203; onpolicy/algorithms/utils/distributions.py | Clipped ratio/value objectives and GAE recurrence adapted to asynchronous sampled requests; independent bootstrap and trace masks, explicit dynamic candidate mask, exact categorical KL. No upstream centralized private observations or synchronous team buffer. |

This is contextual PPO with a seat-visible critic. It is not a faithful SLA,
MAPPO, MAT, LSPO, CQL or MARSHAL reproduction. New packet serialization,
controller, rollout accounting, strict protocol validation and checkpoint
transaction logic are project-specific. Source hashes of these local files
belong in each frozen training snapshot manifest.
