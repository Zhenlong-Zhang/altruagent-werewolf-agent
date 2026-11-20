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
