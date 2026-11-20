# Contextual runtime binding

Parent-approved integration binding, 2026-11-20. This connects the accepted Phase B controller to a frozen Phase C checkpoint. It does not approve collection or assert model improvement. Runtime checks run on Hoffman2 only.

`werewolf_agent.contextual_entry.create_agent()` requires `WEREWOLF_CONTEXTUAL_CONFIG` naming an explicit JSON file; passing its path directly is also supported. No model/decode/mode defaults are substituted. Required top-level fields:

```json
{
  "schema": "contextual-runtime-v1",
  "collection_id": "fresh-predeclared-batch",
  "protocol_hash": "64 lower-case hexadecimal characters",
  "mode": "teacher or sample or greedy",
  "checkpoint_id": "full contextual checkpoint hash, or null for teacher",
  "lineage": {
    "rules": "altruagent-seven-fixed-roles-v1",
    "source_manifest": "equal to source_manifest_sha256 below",
    "generator": "fingerprint of the exact generator_spec below",
    "controller": "predeclared controller version",
    "realizer": "typed-canonical-v1",
    "encoder": "exact contextual_selector.ENCODER_ID",
    "serializer": "exact contextual_slots.SERIALIZER",
    "population": "predeclared opponent/population version"
  },
  "source_manifest_sha256": "raw source manifest bytes hash",
  "sdk_manifest_sha256": "raw isolated SDK manifest bytes hash",
  "generator_spec": {
    "model_id": "explicit model repository",
    "revision": "40-character pinned revision",
    "backend_version": "explicit service version",
    "dtype": "torch.bfloat16",
    "device": "cuda",
    "max_new_tokens": 2048,
    "temperature": 0.6,
    "model_seconds": 30.0,
    "input_tokens": 8192
  },
  "paths": {
    "socket": "/absolute/owner-only/socket",
    "tokenizer": "/absolute/pinned/model",
    "encoder": "/absolute/pinned/MiniLM",
    "journal_dir": "/absolute/private/journals",
    "checkpoint": "/absolute/contextual/checkpoint/directory",
    "source_manifest": "/absolute/source-manifest.json",
    "sdk_manifest": "/absolute/overlay/strategy-overlay.json"
  },
  "seed": 97001
}
```

The example uses explanatory strings for hashes and is not an executable config. Lineage has exactly the eight nonempty string keys shown. Generator and source identities must equal the actual declared configuration hashes, and encoder/serializer/rules/realizer must match the fixed implementation identities; an opaque unchanged generator label cannot conceal changed decode settings. Controller and population are declared versions bound to the verified source and precollection protocol. Teacher mode has checkpoint_id null and may set unused encoder/checkpoint paths null; it does not load those resources. Learned mode loads the hash-bound checkpoint once and requires the seven non-population lineage fields to match. Only a newly declared population version may differ between batches; checkpoint identity stays fixed and the new protocol/runtime records that population version. Learned workers use one Torch CPU thread each; the separate model service retains its own declared threading. The frozen encoder and network are shared within that worker; selectors, request identities and sampling are owner-local. Counter-derived sampling uses the persisted decision ID so restarting a worker does not restart its sample stream. The collection registry archives runtime_config_path and its raw-file SHA, allowing the adapter to verify runtime_config_hash and reproduce seed-derived uniforms; no extra journal field is introduced.

The source manifest has `files: {relative_source_path: SHA256}` and must include the complete small werewolf_agent Python package, including lazy and vendor dependencies. Every enumerated source is checked beneath the repository root. The isolated SDK must use `strategy-sdk-overlay-v2`, have valid per-file overlay hashes, and match the actual imported runner location. Hashes are of file bytes. No other-seat client, global SDK mutation or model download occurs in this factory.

Before parsing any model answer, the wrapper checks returned model ID, revision, backend, actual dtype/device, token cap, requested seed/decode settings, native input count and enabled thinking against the declared specification. Late or mismatched replies cannot become candidates; they produce explicit executor fallback diagnostics. The complete native request is counted, including system instructions and chat template. Deadlines remain bounded soft compute budgets plus post-hoc rejection.

The first journal event for a new owner is `strategy_runtime` with automatic owner plus:

`{collection_id, protocol_hash, mode, checkpoint_id, lineage, generator_spec, source_manifest_sha256, sdk_manifest_sha256, runtime_config_hash, source_verified:true}`.

The event precedes all strategy requests. An owner journal may reopen only with the identical event metadata. Missing runtime provenance on an already used journal is rejected. Fresh collection identity is checked again by the parent's protocol/roster adapter; a path or filename alone is never freshness proof.

Terminal audit is append-only and never enters the actor packet. `strategy_terminal` retains `score` and `termination_reason`, adding:

`terminal_audit={official_final_result_present,is_terminal,status,session_consistent,state_version,own_return_source,own_score,verified}`.

`own_return_source` is exactly `final_result.your_return`, `state.returns[context.agent_id]`, or null. Fallback reads only this agent's entry. Verification requires a nonempty official final-result object, a terminal state, matching state/context/result session identity, and a non-Boolean finite own score in {-1,1}. Status is recorded without inventing a terminal status label. Top-level score equals audited own_score. Missing or invalid proof remains visible with verified=false and must not become training return.

Prepared Hoffman2 command: `PYTHONPATH=starter:. python -m scripts.check_contextual_entry`. Seven focused fixtures use fake generation/embedding/network components and actual runtime integration; they do not load an LLM. Existing B/C suites must also rerun after this integration snapshot.
