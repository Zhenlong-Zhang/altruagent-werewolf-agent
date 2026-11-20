# Round04 execution decisions and receipts

## Preimplementation conclusion

The review identified actual trusted-fact score and late-message paths, alongside substantial action/state/communication limitations. Adopt Planner01's route and the reviewed02 interfaces. Parent/Worker/Reviewer completed source-only review before any new agent/model inference or training. Public asset staging was allowed preparation, not a playing-strength experiment.

## Model and infrastructure preparation

- Public asset setup job2183063 (4CPU12GiB1h) failed immediately, before installs/downloads, because the account's global scratch quota was full. Its script and receipt are retained. No other project files were changed or deleted.
- Read-only quota inspection found home approximately45GiB used of57GiB, insufficient for two16.38GB language models. Retain only a new CUDA environment and small outputs/checkpoints in home; stage large public weights in allocated node TMPDIR. This is an operational choice, not a reason to reduce model capability or request cleanup of unrelated work.
- Environment-only job2183616 (2CPU8GiB1h) completed in7m34s, exit0. Pinned Torch2.7.1+cu126, Transformers4.51.3, HuggingFaceHub0.36.0, Accelerate1.6.0 and HTTPX0.28.1; see outputs/strategy-round04/environment.json.
- Model IDs/revisions/licenses/actual configurations are recorded in model-sources.json. Both contenders really have8,190,735,360 parameters; the MARSHAL8B card's4B frontmatter is a metadata inconsistency, not the actual tensor size.

## Initial native-generation development screen

After final source acceptance, run the same70 synthetic authorized views per model:28 gold-scored seer fact/new-information action presentations,28 unscored wolf/private-public and small-table action presentations, and14 contextual speech-plan capability presentations. Seven seat rotations and both legal-list orders are included for actions. All legal actions, including Abstain, remain present. Gold alignment/candidate coverage is not a claim of optimal strategic play or win probability.

Start with native Qwen thinking, temperature0.6, seed97001+10*seat-shift+scenario-index (identical across order pairs and models),2048 maximum new tokens,8192 input-token ceiling, and30-second soft request budget for BOTH checkpoints on the same allocated GPU. Inputs over the cap are rejected, never silently truncated. These are declared engineering settings for the development screen; timing/format failures may justify a separately recorded revision before live training. Do not call this a final untouched test. Cold loading, complete-output/EOS status, raw generation, validation errors, timing and VRAM remain visible. No outcome-dependent training or match selection occurs in this stage.

Serve only through an owner-restricted Unix socket for later shared inference. Each request carries complete own-seat inputs; only frozen weights/tokenizer are shared, not dialogue or KV state. Readiness follows successful socket bind and permission checks. Per-token deadline stopping is soft; posthoc late rejection remains mandatory.

Delivered speech in PhaseA uses typed assertions and canonical intent/target/audience rendering (typed-canonical-v1). Arbitrary model rhetoric is not executed. This expands the strategic action space beyond R03 but is not unrestricted text generation or an exact SLA reproduction. Responsive control, transcript overlay, long-memory representation and new multibatch training remain subsequent implementation stages under02.

## Resource recovery and independent correctness checks

- Original screen2184358 was confirmed unstarted and cancelled, preserving its receipt. Runtime job2184543 completed successfully and upgraded only the new round04 environment to Torch2.7.1+cu128, enabling official Blackwell support. No prior project runtime was changed.
- Replacement screen2184544 includes eligible97GiB Blackwell devices with the same frozen PhaseA source and paired70-view protocol. It remains pending for actual GPU capacity/priority; an estimate is not evidence of execution. The observed2-GPU/user limit is respected. Full author8-GPU training examples do not establish a minimum hardware requirement for this adapted inference/controller project.
- Independent CPU adapter/interface checks2186290 (1CPU4GiB15min) completed in27seconds, exit0. All16 interface methods and11 backend methods passed. It used job-local SDK dependencies, avoiding environment mutation while GPU work is pending. Logs and result.json are in outputs/strategy-round04/checks-2186290. Source is frozen PhaseA; no LM inference or games occurred in this check. These results establish adapter correctness on the tested fixtures, not playing strength or real-model throughput.
- PhaseB controller/memory implementation and the independently reviewed PhaseC contract proceed while GPU access waits. New source will use separate manifests; the pending PhaseA snapshot is immutable.
- A separately versioned modern-CPU fallback probe measures one native action and one typed speech completion per checkpoint,16threads/BF16/90seconds. This is a latency/format diagnostic, not the frozen70-view30-second GPU comparison, and cannot silently select a model as if hardware/settings matched the latter. Its source overlays and Slurm script are pinned in cpu-probe-source.json. Large weights again use allocation-local TMPDIR. The primitive checks in this snapshot include8 methods after the Reviewer's strict categorical-mass/NaN-padding repair.

## Verified implementation checkpoint (not strength closeout)

- CPU probe2186591 completed9minutes, exit0. Both native8B models timed out on both prompts (four calls, approximately90.12–90.16seconds each), so this CPU configuration is not a usable deployment fallback. Its computed summary is retained with the CPU verification artifacts. This does not establish relative strategic quality or GPU speed.
- CPU verification2187275 failed the actual SDK child fixture because it discarded the job-local dependency path (`dotenv` missing). All other prepared suites passed. The child now inherits dependency paths and the SDK builder copies only an explicit source allowlist, with source/overlay file hashes.
- Repaired earlier snapshot2187511 completed25seconds, exit0; all five suites/36 cases passed. This included actual synthetic PPO updates, whole-network/Adam rollback, game-mean reduction, and hash-bound checkpoint round trips.
- Integrated snapshot2189527 failed one file-backed data test. Its other six suites and five of six data cases passed. A graph-edge tuple became a list on JSON reload, so exact snapshot comparison wrongly excluded an otherwise valid game. Serializer now emits lists and explicitly tests full JSON round-trip identity; numerical graph inputs are unchanged. The failure is preserved.
- Final integrated source snapshot has81 files, archive SHA256 `7fd2ee5b449692738af9db244346a158275b3cb7df15ed79b7acfeb1ab7bc41b`. Job2189749 completed47seconds, exit0. **All seven suites/51 cases passed**:12 controller,4 actual isolated-SDK runner,8 attention/PPO primitive,6 selector,8 PPO/checkpoint,7 runtime entry and6 data/BC cases. Artifacts: outputs/strategy-round04/checks-2189749 and verification-summary.json. BC/PPO optimizer fixtures are synthetic correctness evidence, not new platform training or wins.
- The new source implements complete legal action lists, persistent protected own facts/claims, typed canonical strategy plans, bounded responsive communication, source-pinned teacher/sample/greedy deployment, fresh registered-journal preparation, actual-teacher BC/value warm start, and multi-batch-capable contextual PPO. Model/encoder/controller/source identities, request propensities and delayed faction returns are verified; unknown execution is not silently dropped from PPO.
- A scheduler `--test-only` probe adding V100/FP16 eligibility estimated a later start, even with30minute walltime. No replacement was submitted, and existing2184544 retains its earlier place and frozen source/protocol. Scheduler query IDs are not real model jobs; see scheduler-probes.json.

## Remaining work and claim boundary

GPU2184544 remains pending for shared-campus capacity/priority. It will run the paired native-generation model diagnostic, not new match training. Do not call it running before Slurm says RUNNING, or call its result a win-rate evaluation. Keep the authenticated master open; Slurm work is independent of the interactive connection.

Next empirical work is real-model v2 response/queue/burst readiness, then a frozen population and finite fresh teacher/rollout budgets, C0 imitation, multiple fresh on-policy batches with the learned critic, and a prospective incumbent/C0/adapted comparison. The population launcher/assignment receipts must bind actual other-seat policies; the mixed-router transport path is now covered, but that is not evidence that the intended population has played. Larger language-model tuning remains conditional on measured generator limitations. Existing D7 remains selected; no new strategy/checkpoint is promoted and no tournament-strength gain is claimed.
