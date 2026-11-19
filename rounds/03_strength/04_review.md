# Independent review

## Contract review

Reviewer accepted `02_contract.md` for implementation on 2026-11-19.
The intervention is a generated preliminary rationale followed by offered-ID
token reranking. The contract preserves that description and distinguishes
it from a semantic-embedding policy. Defaults for tokens, latency and match
counts are revisable engineering choices.

The implementation review will check the following evidence.

- Preserve historical proposer/checkpoint behavior and separately identify the
  new algorithm, configured precision, revision and reasoning budgets.
- Reject empty/invalid reasoning, nonfinite ranks and output that returns
  after the monotonic soft budget. Include cold loading in elapsed time.
  A model load or forward pass can overrun a synchronous check, so measured
  timing must accompany any operational readiness statement.
- Keep action and speech choices inside the offered legal alternatives.
  Preliminary reasoning uses only the acting seat's observations. Public
  claims remain distinguishable from verified private and public evidence.
- Verify both tournament action and message paths through a dedicated
  factory at arbitrary seats. Preserve the original practice routing.
- Assign comparison arm and focal seat before inspecting roles/outcomes.
  Retain controller-side assignment receipts so launch or authentication
  failures cannot disappear from the comparison.
- Match model, precision, frozen selector, mode and opponent roster in the
  primary A/B. Log zero-choice outcomes, censoring and failed deliveries.
- Separate old development probes, fresh evidence-consistency tests and
  official prospective outcomes. Cases with equally plausible targets use
  preferred sets or an unlabelled grounding audit.

## Starting evidence

The prior GPU BF16 ID probe scored 20/42 against a 21/42 first-option control.
Some known-role scenarios are already constrained by runtime action filters.
A proposal-only gain on those scenarios establishes diagnostic behavior.
Official game results are needed to establish playing strength.

The prior 30-game update and three operational wins do not contain a
prospective comparison against the new proposer. Current review status is
implementation pending. No competition-strength promotion is accepted yet.

## Static implementation review

The Reviewer inspected the new proposer, production factory, comparison
routing, reporter and focused regression checks. The following issues were
raised and repaired in source before runtime acceptance.

| Finding | Repair inspected |
| --- | --- |
| CUDA execution can synchronize after the first post-forward timing check. | The proposer checks the budget again after rank conversion to host values. The regression advances its clock specifically at `Tensor.tolist`. |
| Forced singleton choices could inherit the previous rationale. | Per-call proposal diagnostics are cleared before deciding whether a model call is needed. |
| An HTTP backend could be labeled as reasoned while following a different algorithm. | The reasoned path explicitly rejects unsupported HTTP/API execution, and the competition factory stages a local HF backend. |
| Requested `auto` precision can resolve differently on allocated devices. | The versioned generator digest includes actual loaded precision and staged revision for model-backed choices. A mismatched explicit pinned revision is rejected. |
| Configuration errors could escape fallback when identity logging reparsed an invalid budget. | Legacy logging retains raw configuration strings, and the competition factory validates the inexpensive LLM configuration before play. |
| Parser-only fallback checks did not exercise the public policy entry point. | The added factory-to-action check verifies a legal fallback, its logged cause and the delivery callback for an arbitrary contestant seat. |

The original ID/Judge algorithms and absent-new-flag generator identity
remain available. The production factory enables both hybrid action and
speech handling without assigning a fixed focal seat. Context fixtures
check that seat 6's authorized facts reach the policy and that the selected
checkpoint is logged.

Static inspection supports proceeding to Hoffman2 verification. Model
loads and individual forward passes remain non-preemptive. A measured
soft-budget overrun is therefore an operational outcome, even when late
rankings are rejected. A new real-model/gameplay result is still required
for any statement about candidate quality or tournament strength.

The focused suite now contains 16 cases, including malformed startup
configuration and legacy fallback behavior. Static review found no further
material blocker for a controlled local-HF, explicit-BF16 trial. Runtime
verification remains pending. A true interruptible decision deadline and
fresh-state revalidation remain deployment follow-ups rather than completed
features.

## Runtime verification ledger

- Hoffman2 job 2111205 reached the focused regression suite and failed
  before model diagnostics or gameplay. The parent reported five errors
  from fixtures omitting the SDK's required `LegalAction.input` argument.
  This fixture defect was missed during initial static review. The repair
  supplies `input={}` and requires the checks to be rerun before acceptance.
- The root scheduler driver now repeats the comparison-arm count check
  after its collector loop. This prevents an unexpected runtime exit from
  being labeled successful merely because the total-game threshold was
  reached while one arm was short.
- The parent reported that repaired Hoffman2 job 2111362 passed all 16
  worker checks and five comparison-accounting checks. Its CPU FP32
  historical control scored 21/42, with all 42 rank lists exactly matching
  the archived CPU round-one probe. The previous 20/42 result came from a
  GPU BF16 run. Precision/device conditions must therefore accompany the
  historical score. Independent artifact review is pending transfer of
  the current probe and regression records.

## Independent audit of real-model preliminary reasoning

The Reviewer inspected all 14 rows in
`outputs/strength-round03/job-2111362/reasoned-partial.json`, rather than
inferring rationale quality from the aggregate score. This is a partial,
ordered historical development probe, not a fresh held-out sample. Its
configuration records CPU BF16 Qwen3-8B at immutable revision
`b968826d9c46dd6066d109eabc6255188de91218`, reasoned-v1, 48 generation
tokens and a 30-second soft budget. All 14 rankings were accepted; elapsed
was 12.312–19.147 seconds, with a 15.5135-second median. The first call
includes 2.430 seconds of preparation. This establishes neither an
unbounded-cold-start guarantee nor a competition outcome.

| Rows (zero-based) | Observation checked against the actual case constructor | Assessment |
| --- | --- | --- |
| 0–1 | Seat 0 privately knows Player1 is a wolf. Both rationales instead say private knowledge confirms Player2 is a wolf; row 1 cites Player3's real public accusation as support for that false private certainty. Both rank Player2 first. | Confirmed-private-evidence inversion; public accusation is promoted into private certainty. |
| 6–7 | Rotated seat 1 privately knows Player2 is a wolf. Both rationales instead assert Player3 is the privately known wolf and rank Player3 first. | Same inversion after rotating identities; not merely an isolated parse failure. |
| 12–13 | Rotated seat 2 privately knows Player3 is a wolf. Both rationales instead assert Player4 is the privately known wolf and rank Player4 first. | Same inversion in the third identity rotation. |
| 4–5, 10–11 | The seer already knows the rival is a villager; both ascending and descending rationales acknowledge that fact and still prefer investigating that already checked rival. | Fact recognition succeeds, but action semantics/information gain fail. No fresh information is obtained. |
| 2–3, 8–9 | The acting wolf knows its dead partner's identity. A live player claims to be the seer and gives a matching check. The seer identity itself is unverified. | Treating that claimant as uncertain is legitimate. The diagnostic preferred target is a strategy heuristic, not a verified optimal move. Rows 3/9 overstate the claimed check as confirmed; row 8 also describes killing a plausible seer as protecting them. Ranking changes with offered order in both pairs. |

The first 12 rows score 2/12 against a 6/12 first-option control; the whole
14-row snapshot scores 2/14 against 7/14. Six seer-vote presentations
contradict the supplied private check, and four seer-investigation
presentations choose a redundant check. These categories refer to this
inspected snapshot only. Order changes occur in 2/7 completed paired
presentations. All rank lists are syntactically legal, so successful parsing
and the second-pass instruction to recheck facts did not prevent semantic
errors. There is no evidence here that extra generated reasoning improves
playing strength. This candidate is **not accepted for strength promotion**.

Some historical scenarios are already constrained by production `_actions`:
known-wolf seer voting and avoiding repeat seer checks have filters. Thus
these diagnostic mistakes do not establish that those exact bad actions
would be delivered in production. Conversely, those existing filters mean
that fixing these probe scores alone would not establish an end-to-end
win-rate gain. The no-known-wolf seer-vote branch can still retain a verified
innocent alongside unknowns, as the Planner's archived near-miss documents.

## Next source boundary and feasible acceptance evidence

Preserve the current baseline and partial diagnostics. Start a new,
explicitly versioned intervention rather than editing reasoned-v1 in place.
A sensible next intervention is evidence normalization with role-aware
candidate constraints; it is not yet another unvalidated reasoning
improvement.

1. Build an acting-seat evidence packet from the authoritative observation
   parser: own role/seat, confirmed private checks, confirmed partner facts,
   public revealed deaths, and separately attributed player claims. Use
   explicit records such as `seat=1, role=wolf, source=private_check` instead
   of asking the model to infer sources from a compact numeric-key mapping.
   Text from chat must never populate confirmed facts. Include the actual
   seven-player roster, no Doctor, phase objective and first-night rules.
2. Keep candidate legality and epistemic constraints deterministic. Preserve
   the known-wolf voting and unseen-player investigation filters. Consider
   excluding a verified innocent from a seer's vote when offered unknown
   alternatives exist; label that as a strategic constraint, with explicit
   exceptions/fallback when all offered candidates are known innocents.
   Never invent an unavailable action or claim an inferred role is certain.
   Record pre-filter and post-filter slates plus reasons for reproducibility.
3. Audit the constructed packet and the exact prompt, not only generated
   prose. Probe counterbalanced identities/order, a verified check opposed
   by a false chat claim, an already checked player, all-candidates-known
   cases, absent private checks, and adversarial chat claiming system
   authority. Compare evidence attribution to the acting seat's source
   observation; do not inspect other players' private logs. Ambiguous wolf
   targets and unverified role claims remain manual grounding cases without
   an arbitrary correctness target.
4. Compare normalized IDs and normalized reasoning on identical frozen
   slates/model/revision/precision/selector. This isolates the extra
   reasoning stage. Separately compare the full new constrained system to
   the preserved old baseline if the practical question is system strength.
   Neither comparison should silently mix old and new action filters.
5. Run the focused contract regressions and bounded real-model latency
   probes on H2 before committing official matches. Inspect source-grounded
   rationales manually, and require no contradictions of the confirmed
   facts in the inspected deterministic sanity cases. A larger model that
   cannot return within the measured operational budget is not a ready
   replacement. Once a candidate clears these checks, collect prospective
   roster-specific games with preassigned rotating focal seats, all
   controller receipts, role counts, uncertainty and unresolved outcomes.
   Probe correctness is an execution/grounding gate, not a win-rate claim.

No official comparison games had been created at the time of this audit,
according to the parent. Parent-reported completed 8B BF16 ID diagnostics
(20/42 historical; 27/42 fresh scored versus 31/42 first-option) and 14B
results (10/42 historical ID; initial reasoned outputs rejected after more
than 30 seconds) remain separate from the independently inspected 14-row
8B rationale snapshot. Do not pool these conditions or promote model size
as an improvement.

## Comparison-accounting review

The current `scripts/summarize_strength.py` now rejects duplicate focal
match logs and duplicate controller receipts, excludes stale/cancelled
technical closures from resolved results, retains missing workers as
unresolved assignments, and reports assignment mismatches/orphan logs.
Wilson intervals, role-specific counts, role-standardization only when all
roles are observed, and all-assigned outcome bounds are appropriate
transparent descriptive summaries. This static review does not claim the
new script has passed runtime tests unless H2 evidence is supplied.

Before interpreting any comparison, require empty `protocol_problems` and
inspect per-game mode, checkpoint, opponent pool and proposer configuration;
the summary does not enforce their equality across arms. A configuration
list pooled across all logs is provenance, not proof of a controlled A/B.
In particular, orphan logs currently remain in the descriptive arm totals
while being flagged, so those totals must not be presented as a valid
prospective comparison when such flags exist. Use a dedicated comparison
log directory. The result resolver currently treats any nonempty
non-stale termination reason with a numeric score as resolved: confirm the
platform's valid terminal-reason vocabulary or use a known-positive
allowlist before a new or unknown reason can enter the win denominator.

## Phase A close and phase B static boundary

The Reviewer subsequently inspected the transferred
`reasoned-stopped-development.json`: historical 42/42 accepted, 10/42
preferred versus 21/42 first-option control, median 15.180 seconds,
maximum 19.147 seconds; the deliberately stopped fresh sequence contains
only three presentations, all accepted and none preferred. Its first two
fresh cases explicitly supply Player1=wolf and Player2=villager, yet both
rationales assert that private knowledge says Player2=wolf, matching the
untrusted accusation. The third again chooses to investigate already
checked Player2. Stopping this development run does not create a valid
small-sample strength estimate; it is sufficient to reject the candidate
for the observed grounding defect. The transferred regression transcript
contains the 16-test and five-test successful H2 summaries.

The root's latest phase-B source now resolves outcomes only for the
positive `completed` reason, repairing the denominator concern above.
The optional focal competition factory routes exactly one preassigned
contestant through the production constructor while preserving the six
rules opponents and the practice wrapper's tournament guard. The new
fixture checks the returned policy's competition strategy marker and
tournament opt-in. These are interface checks; live tournament timing and
action delivery remain separate evidence.

The mapped diagnostic changes each numeric action ID by +3 modulo 7,
keeps target labels fixed and maps preferred sets through the same
bijection. This is a useful ID-versus-target counterexample. At initial
review, `--suite all` did not include this new suite, so the phase-B
scheduler must request it explicitly unless that behavior is revised.
Logging the complete ID/target/label correspondence will make these
results independently auditable. Candidate order sensitivity is descriptive:
a change between two equally preferred unknown targets is not a grounding
failure.

Grounded-v2 source inspection is underway as a distinct opt-in intervention.
Its one-pass letter-slot ranking maps back to exact offered IDs and keeps
explicit targets separate; it generates no rationale. This design addresses
an observed failure mode but has not yet established superior decisions or
win rates. The full new guard and evidence compiler require their own H2
regressions and raw-model diagnostics, with forced policy choices reported
separately.

## Phase B final static acceptance

The Reviewer inspected the completed grounded-v2 compiler, dispatch,
candidate guard, metadata changes and 23 focused test methods before the
phase-B H2 snapshot. No material source blocker was found for controlled
H2 verification.

- Explicit offered targets take precedence, and label parsing is the only
  fallback. Numeric or opaque action IDs never establish a player identity.
  Explicit `target=None` keeps a speech alternative unbound even when its
  wording mentions a player. Letter-slot logits map through the offered
  record back to its exact original ID.
- Confirmed facts and attributed unverified chat remain distinct. Seer
  negatives are described as nonwolf alignment. Own-role conflicts and
  private/public alignment conflicts raise errors. The roster/rules prompt
  now uses the documented three **consecutive** days without elimination.
- The production-path fixture uses opaque action IDs and checks that the
  verified-innocent guard retains the unknown target, avoids a model call
  when the result is forced, and logs `model_valid=False` and
  `sampled=False`. It separately verifies unchanged ID-control filtering,
  unseen-player investigation, all-known fallback and an unbound Abstain
  whose numeric ID must not become a target.
- The focused checks cover int/string knowledge keys, public authority
  impersonation as unverified chat, arbitrary-ID mapping without generated
  rationale, nonfinite grounded ranks, and budget expiry specifically after
  host conversion. Actual H2 execution remains required. Cold model work
  still uses the shared non-preemptive loading path and a soft budget.
- Grounded prompt/guard identities and loaded dtype/revision distinguish
  the new distribution from controls. The HF call metadata now uses the
  staged backend name. Guard before/retained IDs and raw model diagnostics
  remain separately observable.

The benchmark now logs each offered ID, target and label. The root's
comparison wrapper, positive completed-outcome allowlist and accounting
checks were reviewed as described above. The next accepted step is H2
regressions followed by matched real-model diagnostics. Any live canary
must exercise the actual factory and delivery path before production
readiness is claimed. No new match result or strength improvement is
asserted by this static acceptance.

## Phase B H2 regression verification

The Reviewer inspected the transferred job 2113134 transcript at
`outputs/strength-round03/job-2113134/regression-results.txt`. It lists
all 23 focused cases as `ok` and records 23/23, 6/6 comparison and 4/4
accounting checks passing. This supersedes the pending-execution status
for those checks. The parent identifies the allocation as n1200,
eight CPUs and 64 GB; the transcript itself records the test outcomes.

The phase-B source manifest is
`outputs/strength-round03/phase-b-source.json`, base commit `50a4fb0`,
archive SHA256
`9ef888f64b47b33be5b2f9a6a0090a6568fbb3c809faad4f0083f25e271b7f6e`.
The Reviewer independently compared all 26 manifest file hashes against
the inspected local files and found no mismatch. This establishes the
source correspondence with the supplied manifest; the transferred test
transcript does not itself independently authenticate the remote archive.
The user-owned WORKPLAN is excluded from the snapshot.

Acceptance now extends to the exercised regression contracts. Matched
real-model grounded/ID diagnostics and mapped-ID behavior remain pending,
as do the actual factory/delivery canary and prospective game results.
The separately scheduled 14B screen must retain its own model revision,
allocation and measured latency; a larger-model diagnostic is not pooled
with the 8B baseline comparison. No official games have yet been created,
according to the parent, and no strength promotion is accepted.

## Completed 8B grounded diagnostic audit

The Reviewer inspected both transferred phase-B artifacts for job 2113134:
`grounded-strength.json` (112 presentations) and `grounded-mapped.json`
(70 presentations). Both identify CPU BF16 Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, grounded-v2 and
hard-evidence-v1. All 182 presentations returned accepted ranks. Offered
ID/target/label records match their compiled candidate annotations, and
returned ranks refer to offered IDs. The compiled verified-fact records
keep chat outside confirmed facts in the inspected outputs.

| Scenario | Historical preferred / presentations | Fresh preferred / scored | Mapped preferred / scored |
| --- | --- | --- | --- |
| Seer votes for privately confirmed wolf | 14/14 | — | — |
| Seer investigates previously unknown player | 14/14 | 14/14 | 14/14 |
| Verified checks override contradictory chat | — | 14/14 | 14/14 |
| Avoid verified innocent when unknowns offered | — | 14/14 | 14/14 |
| Wolf removes plausible seer claimant | 7/14 | — | — |

The historical total 35/42 combines 28 evidence-consistency cases with
14 strategy-reference cases. The latter do **not** have gold-standard
optimal actions: a matching claimed investigation does not verify the
speaker's role. In fact, this model chose the **second offered slot in
all 14** wolf-remove-seer presentations. Its 7/14 reference score therefore
reflects an order pattern rather than demonstrated claimant assessment.
It also chose the second slot in all 14 unlabelled adversarial wolf-claim
presentations, in both fresh and mapped suites. This is a remaining
behavioral limitation even though those cases have no unique correct
answer.

The fresh and mapped totals each score 42/42 constraint-labelled
presentations, compared with 31/42 first-option control. Each additionally
contains 28 unlabelled claim-based presentations, which cannot be counted
as either correct or incorrect from these artifacts. These are raw
proposal probes: all offered alternatives were present during ranking;
normal production guard-forced choices are separate measurements. Correct
mapped cases support ID/target binding on this finite suite, not universal
permutation invariance.

Fresh candidate-order changes total 20/35 paired presentations: two
unknown-investigation pairs, five verified-innocent-avoidance pairs, six
unlabelled village-claim pairs and seven unlabelled wolf-claim pairs.
Mapped changes total 19/35: respectively one, five, six and seven. The
labelled changes are between equally permitted unknown players; none
violates a confirmed-wolf check or selects the known innocent in these
cases. Thus treating every changed top choice as a semantic error would
be incorrect. Conversely, the unlabelled order-sensitive outputs provide
no evidence of strong social reasoning.

Historical median elapsed is 1.7235 seconds, maximum 4.508; fresh median
1.6935 seconds, maximum 1.833; mapped median 1.7225 seconds, maximum
4.159. These are observed timings under this allocation, with initial
calls including model preparation. A 14B result on four CPUs and an 8B
result on eight CPUs compare configurations; they do not isolate model
size as the cause of latency differences.

This finite grounding screen supports proceeding to the planned operational
canary once matched baseline checks complete. It does not establish a
win-rate gain. Prospective official outcomes, unresolved assignments,
fallback/delivery rates and role balance remain necessary before accepting
any strength promotion.

## Phase C cyclic aggregation contract

The Reviewer accepts a separate opt-in cyclic-slot experiment responding
to the observed second-slot pattern. Phase-B remote controls remain frozen;
social-dialogue changes are deferred. With K offered alternatives (K ≤ 7),
run all K cyclic rotations, relabel the displayed slots, compute each
pass's log-softmax over offered slots, invert to exact offered IDs and
average each candidate's log probability. Reject duplicate-ID ambiguity.

Under an additive candidate-score plus position-bias model, every candidate
occupies every slot once and the averaged position contribution is equal.
This does not remove general context-dependent ordering effects or prove
permutation invariance. Exact ties retain original offered order through
an explicitly stable tie rule.

One absolute soft deadline includes loading and every pass, with a check
following host conversion in each pass. Nonfinite values, late completion
or an incomplete rotation set reject the entire aggregated proposal;
partial averages cannot be used as successful output. Default/`none`
behavior, configuration and generator identity must remain phase-B exact.
Only the opt-in receives a new aggregation version and diagnostics.

Focused H2 checks should demonstrate cancellation for position-only scores,
preservation of a true content preference, opaque-ID inverse mapping,
stable ties, whole-proposal rejection after a later-pass failure and
unchanged default provenance. A matched real-model screen must measure
the added latency and remaining order sensitivity before any deployment
or strength claim. This is contract acceptance, not implementation or
runtime acceptance.

## Phase C final static review

The Reviewer inspected `rank_cyclic`, its opt-in configuration/provenance,
the five additional focused methods (28 total), and the root's pre-inference
scenario filter. No material source blocker was found for a separately
frozen H2 diagnostic run.

The implementation validates unique offered IDs and K ≤ 7, runs all K
cyclic rotations, reassigns letter slots, computes offered-slot log-softmax
in float32 and maps each pass back to original IDs before averaging.
An indexed Python sort preserves original order for exact ties. Each pass
checks the shared deadline after host conversion; a final check precedes
returning ranks. Later-pass nonfinite scores or time expiry raise without
returning a partial aggregate. Completed-pass diagnostics remain visible
for failed attempts.

The absent/`none` path retains the original one-pass implementation and
configuration. Only explicit cyclic mode adds the aggregation version to
configuration and generator identity. Fixtures check first-pass prompt
identity, synthetic additive position cancellation, content preference,
opaque IDs, unique-ID/size bounds, second-pass NaN and post-conversion
expiry, and unchanged default identity. These are inspected fixtures;
H2 execution is still pending for this phase.

The proposed focused real-model screen selects wolf-remove-seer,
verified-check-over-claim and unknown-investigation cases before inference,
and records suite/filter in artifacts. This is a targeted development
screen responding to an observed failure, not a new held-out evaluation.
Measure both whole-proposal acceptance and timing, especially the added
cost for six-choice speech or action sets. Frozen phase-B official games
must retain their original source and aggregation setting; do not pool
phase-C outcomes into that comparison.

## Explicit CPU-thread boundary static acceptance

The Reviewer inspected the parent-approved `WEREWOLF_CPU_THREADS` change
and three new focused methods (31 total). No material blocker was found
for H2 verification. Startup validates a positive decimal integer and
rejects values above a supplied valid `SLURM_CPUS_PER_TASK` before any
Torch/model import in the constructor. Explicit settings are applied and
read back on every HF call, including cached-model calls. The original
absent-option cold-load behavior remains intact.

Requested and actual intra-op thread counts appear only when the option
is explicit; the generator digest then includes both. Raw malformed
requests survive legacy fallback logging without reparsing outside its
exception boundary. Fixtures inspect invalid/over-allocation startup,
cached-call set/get application, absent/4/2 identity separation and malformed
legacy legal fallback. Execution of these new checks remains pending.

This is a per-process intra-op setting. It neither limits all auxiliary
thread pools nor enforces the sum across concurrent HF processes; the
Slurm orchestration must budget those processes together. Any timing or
model result under changed explicit thread counts must retain that
configuration rather than being silently pooled with the frozen defaults.

## Rotated collection and held-out checkpoint comparison review

The Reviewer inspected the new collection/checkpoint entry points, optional
routing extensions, reporter, driver branches and added comparison fixtures.
The collection path requires sampled mode and an explicit checkpoint;
it assigns one focal seat from session UUID before observing a role and
logs `collection_assignment`. A fixed roster is shared across that cohort.
The checkpoint path prevalidates distinct incumbent/adapted weights,
hashes the checkpoint arm and focal seat independently of role/outcome,
and passes the selected checkpoint to the same competition factory with
one fixed proposer and evaluation mode. The original comparison namespace
remains the default. Reporting retains zero-choice collection assignments.

The driver distinguishes sampled evaluation from training via
`WEREWOLF_EVALUATE_ONLY`; the evaluation branch checks both total games
and requested arms, including a final arm check after the loop. The
trainer records its predeclared update step, verifies starting checkpoint,
proposal identity and sampled propensities, excludes missing/duplicate
terminals and unconfirmed deliveries, and refuses a batch below its
usable-game minimum. A completed no-sampled-decision game remains visible
in cohort reporting even though it cannot contribute a policy gradient.
Keep that eligibility distinction explicit when reporting training counts.

The planned step 0.5 update after 30 eligible new games is an experiment,
not an established effective learning rate. Keep the new opponent roster,
thread configuration and proposer cohort separate from historical training;
freeze the resulting checkpoint before new held-out evaluation games.
Verify evaluation match hashes do not overlap recorded training hashes.
The sampled incumbent/adapted comparison measures those stochastic policies;
it cannot be presented as an evaluation of greedy deployment.

One accounting repair was requested: compare logged assignment seed and
version to requested values in addition to arm/seat. A different routing
namespace can coincidentally produce the same arm and seat; explicit
metadata must not be ignored. Late census receipts are now visibly flagged,
which should accompany any interpretation of prospective assignment.
Subject to this small accounting check and the pending H2 suites, no
material routing/training source blocker was found. The orchestration
must ensure the previous collector has actually released the platform
worker connection before starting the next cohort; a scheduler dependency
alone does not demonstrate live registration handoff.

## Cyclic real-model results and wide-slate limit

The Reviewer inspected job 2113721's four diagnostic JSON artifacts and
regression transcript (28 focused + six comparison + four accounting checks,
all successful). All 70 focused presentations per aggregation condition
were accepted. The historical wolf-claim reference score moves from 7/14
without balancing to 8/14 with balancing; order-changing pairs move from
7/7 to 0/7. That is evidence the presentation pattern changed on these
cases. A one-presentation gain against a soft strategy reference does not
establish better wolf play.

Both conditions retain 28/28 preferred seer presentations in the focused
fresh suite and 28/28 in the mapped suite. Unknown-investigation paired
choices change more often with balancing: two to six pairs in fresh cases
and one to seven in mapped cases. All remain within equally preferred
unknown sets, so these changes are not errors. Balancing therefore removes
the observed wolf order flip in this finite screen without proving general
invariance or superiority.

The independently inspected `wide-2113134.json` exposes an operational
limit absent from the small-slate screen:

| Offered alternatives | One pass, seconds | Cyclic, seconds | Cyclic result |
| --- | --- | --- | --- |
| Six vote targets | 7.761 | 28.797 | Accepted, 6/6 passes |
| Seven alternatives including Abstain | 4.955 | 32.929 | Timeout, 6/7 passes; no aggregate returned |
| Five actual wolf speech alternatives | 4.821 | 23.828 | Accepted, 5/5 passes |

These are individual stress observations, not latency percentiles. The
seven-choice failure confirms that the deadline is soft and can be exceeded
inside a synchronous pass; whole-proposal fallback works as contracted.
The six-choice success leaves little margin under a 30-second budget.
Broad cyclic deployment is **not accepted**. Retaining unbalanced grounded-v2
for the next training cohort is consistent with the observed speed and
scope of evidence. Further balancing work should budget worst-case slate
size/context rather than extrapolating two-choice timings.

## Larger model screen and next source provenance

The Reviewer inspected job 2113211's 14B grounded artifacts at immutable
revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, CPU BF16.
They report 35/42 historical, 42/42 fresh scored and 42/42 mapped scored,
matching the 8B preferred-case totals. All 182 presentations were accepted.
Historical median is 6.6415 seconds, fresh 6.524 and mapped 6.773, compared
with roughly 1.7 seconds for the 8B configuration. The parent identifies
four CPUs for 14B versus eight for 8B, so this is a configuration-level
quality/latency screen. It neither isolates a model-size latency effect nor
shows a stronger 14B policy. There is no diagnostic basis here to incur
that extra measured cost for improved accuracy.

The summary's requested seed/version repair and its new fixture are now
present in source. Per-arm proposer configurations are exposed for auditing.
The Reviewer compared all 29 files in `phase-d-source.json` with the local
reviewed files: zero mismatches; archive SHA256
`ca79a829fa63afdb46db784afc8a8db371be3afee4cfec5208553896118bbddf`.
This accepts source correspondence with that manifest. Runtime execution
of the new 31/9/5 suites and actual training/held-out comparisons still
require their own transferred evidence. The frozen earlier game cohort,
late receipt flags and later mixed-roster on-policy cohort must remain
separate in reporting.

## Phase B prospective outcome and failure audit

The Reviewer inspected `phase-b-summary.json` and the 25 focal decision
logs under `data/round03-grounded-v2/decisions`. Only focal-seat authorized
observations, emitted messages, choices and public outcomes were used for
this failure review; the six opponents' private logs were not consulted.
The completed cohort reverses the development-probe impression:

| Arm | Wins / games | Wilson 95% interval | Villager | Seer | Wolf |
| --- | --- | --- | --- | --- | --- |
| Legacy IDs | 9/12 (75.0%) | 46.8–91.1% | 4/5 | 3/3 | 2/4 |
| Grounded-v2 | 5/13 (38.5%) | 17.7–64.5% | 4/9 | 1/2 | 0/2 |

Role-standardized descriptive rates are 74.3% and 32.5%. All 113 selected
decisions were acknowledged; neither arm recorded fallback or delivery
errors. The summary has no protocol problems and one explicitly flagged
late controller receipt. Both arms use checkpoint `d7ebf7670444067a`,
the same six-rules roster, greedy selection, model revision and precision;
the proposer identities remain distinct. This supports an operationally
valid pilot comparison, with a census-timing qualification. Wide intervals,
small role subgroups and unequal seats prevent a precise general strength
claim. Grounded has four seat-0 games versus one for IDs, relevant given
the known lowest-seat opponent heuristics. Role standardization does not
correct seat imbalance. Nevertheless, there is no basis to promote grounded
from these outcomes; the point estimates favor IDs in every role subgroup.

### Observable mechanisms rather than inferred causes

Across grounded's 53 choices, the executed selection is the LLM's first
rank every time. Across IDs, all 57 model-backed choices likewise execute
the first rank, with three additional forced choices. The frozen learned
head therefore makes no observed override of the proposer in this cohort.
The result is practically dominated by proposer ranking and generated
slate membership, not an active learned correction of its mistakes.

Grounded selects the specific-target `I lean toward voting for PlayerX`
template in 20/24 speeches, versus 4/25 for IDs. IDs selects the generic
request for suspects/evidence in 20/25 speeches, versus 4/24 for grounded;
its remaining speech is a confirmed seer disclosure. In 6/20 grounded
lean-then-vote pairs, the subsequent executed vote differs from the announced
player. IDs has 0/4 such mismatches. These are descriptive behavioral
changes, not proof that one template caused the win-rate difference. They
show that the grounding prompt changes social behavior well beyond the
verified-innocent guard and can undermine target coordination.

Concrete loss examples:

- `match_fda8424ac7719a3a_seat_5.jsonl`: the grounded seer privately clears
  Player0, publicly leans toward unknown Player1 and votes for Player1,
  while the public tally eliminates the cleared villager Player0. On the
  next day it has also cleared Player6 and again emits an unsupported lean,
  now toward Player2. No private fact is inverted here. The policy fails
  to express protective negative-check evidence; the current speech slate
  offers positive-wolf disclosure but no corresponding cleared-player
  disclosure. More accurate private binding alone does not solve this
  coordination problem.
- `match_602730ef6e38e473_seat_1.jsonl`: grounded villager leans toward and
  votes for Player0 on day one; Player0 is subsequently publicly revealed
  as a villager. On day two it announces Player3 but votes Player5. Similar
  initial Player0 targeting occurs in several grounded villager losses.
  Without evidence of the hidden roles at decision time, those votes cannot
  be called knowingly wrong; they demonstrate deterministic low-evidence
  targeting and inconsistent commitments.
- `match_90a25acc5e2db14b_seat_5.jsonl`: grounded wolf announces Player3
  then votes Player6 on day two, while its partner is eliminated. Its
  preceding acknowledged kill targets Player6, but the public night death
  is Player2, consistent with a pack-target coordination disagreement under
  the documented tie rules. Neither the policy's legal action nor a
  syntactically valid fact packet establishes effective faction play.
- `match_f754ca74c15bfaac_seat_0.jsonl`: the grounded wolf loses after its
  first-day speech and vote. The documented rules-seer lowest-seat check
  makes seat 0 especially exposed; this loss cannot isolate a proposer
  defect from that matchup circumstance.

The logged `new_messages` fields are deltas and frequently empty. They do
not prove that the cached proposer input lacked earlier chat. A future
social-reasoning audit needs the full acting-seat prompt/evidence packet or
message references rather than inferring unseen context from those deltas.

### Recommended decision before more collection

Keep legacy IDs as the incumbent. Prioritize a minimal, separately
versioned IDs-plus-verified-innocent-guard candidate, preserving its original
prompt, ID ranking and speech choices. That isolates the concrete missing
constraint without importing grounded's broader speech/ranking changes.
Run its regression and matched prospective comparison before adopting it.
A replay cannot alone estimate wins because different speech/actions alter
future observations.

The already started grounded on-policy cohort may remain a bounded research
experiment, but adapting a weak proposer is not the strongest current
competition default. The 20-parameter selector sees rank, rule score,
public support, known alignment and context interactions; it has no explicit
speech-intent or commitment-consistency feature. One 30-game update need
not repair this social behavior or recover candidates absent from the
LLM/rules slate. Treat its learning rate and outcome as experimental.

If grounded adaptation proceeds, a head-only incumbent/adapted comparison
answers whether that head improved under the fixed grounded system. It
cannot answer whether the full system beats the current legacy-IDs
incumbent. Before any held-out games, define a separate whole-policy
comparison between legacy IDs and adapted grounded with identical model,
roster, thread configuration and evaluation mode, while logging each
proposer/checkpoint pair. Do not silently replace the predeclared head-only
question or pool greedy Phase B with later sampled mixed-roster results.
No grounded promotion is accepted by this review.

## Narrow incumbent-IDs guard static review

The Reviewer inspected the parent-approved pivot to
`WEREWOLF_EVIDENCE_GUARD=verified-innocent-v1` and four new focused methods
(35 total). No material source blocker was found for freezing and H2
verification. This is an isolated candidate, not an accepted strength gain.

In the incumbent IDs path, the only added action constraint is for a seer
in day voting: normalize its private knowledge keys, retain the existing
confirmed-wolf priority, otherwise exclude verified nonwolves when offered
unknown targets exist. If all targets are known, retain legal candidates.
The legacy action resolver handles opaque IDs through target labels.
Other roles/phases, night investigations, speech templates, model prompt
and ID-token ranking remain unchanged. The grounded branch retains its
separately versioned behavior.

Absent and explicit `none` preserve incumbent configuration and generator
identity. The enabled guard receives its own version; invalid startup
configuration is rejected cheaply, while legacy inference errors retain
legal fallback and raw request provenance. Fixtures inspect prompt/config/
identity equality, the normal opaque-ID factory path, string private keys,
confirmed-wolf priority, all-known fallback and unchanged other role/phase/
speech outputs. H2 execution of these new checks remains pending.

For outcome reporting, distinguish actual candidate removal from a logged
reason: the implementation may log the avoidance reason even when every
candidate was already unknown and the slate is unchanged. Compare before
and retained IDs to count impacted decisions. The new prospective test
must keep the unchanged IDs incumbent as its control and the modified
proposer/checkpoint identities explicit.

## Complete-policy four-arm comparison review

The new policy-profile wrapper fixes IDs as the proposer, selects a UUID-
hashed focal seat and arm before observations, and passes the selected
mode, evidence guard and checkpoint to the exact competition factory.
Other seats keep the fixed opponent roster. The wrapper reads no private
role information for assignment and emits `policy_assignment` without
an evaluation role. The reporter retains that event even for zero-choice
matches, and the summary supports the separate `wolf-policy-v1` namespace.

The proposed four arms isolate incumbent greedy IDs, guarded greedy IDs,
guarded sampled old weights and guarded sampled adapted weights. The
last two estimate the update's effect under the same sampled policy;
the earlier contrasts expose the effects of guard and sampling. Keep all
arms in their frozen predeclared order when hashing UUIDs. A 12-per-arm
pilot is a sizing choice with broad uncertainty, not an automatic promotion
rule or proof of the best tournament policy.

Two specific repairs were requested before source freeze:

1. Reject absent/null/empty or non-file checkpoint paths at profile startup.
   `Selector(mode, None)` otherwise accepts initial weights, followed by a
   runtime environment-update failure when the selected profile tries to
   install `None` as its checkpoint path.
2. Make the profile-routing fixture use a genuinely different adapted
   checkpoint and assert both the selected selector ID and logged
   assignment checkpoint ID. Its initial four profiles all used the same
   file and did not inspect checkpoint identity, so the test could not
   detect routing the wrong weights.

Profile names should also be compatible with the driver's comma-separated
arm interface; disallow commas and surrounding whitespace rather than
allowing silently unmatchable labels. Apart from those validation/test
repairs, static inspection found no additional material routing, mode/
guard override or private-information boundary issue. Acceptance of the
completed implementation awaits those repairs and new H2 checks.

### Complete-policy repair verification

The Reviewer re-inspected the final wrapper and fixtures before phase-E
freeze. Profile checkpoints now require nonempty string paths to existing
files before constructing selectors. Names reject commas and surrounding
whitespace. The adapted fixture writes different weights and checks both
the actual selected selector ID and assignment checkpoint ID; null-weight
profiles have an explicit rejection case. The requested repairs are present.

Static acceptance is granted for the isolated source snapshot and its H2
checks, followed by the predeclared collection/comparison. No material
source blocker remains from this review. Actual runtime, training update
provenance and disjoint held-out four-arm evidence remain pending. The
archived one-game grounded pilot and cancelled former cohort/head-only
comparison remain separate, with no automatic adoption of any arm.

## Phase E H2 checks and resource rescope

The Reviewer independently inspected
`outputs/strength-round03/job-2116767/regression-results.txt`: all 35 named
focused tests are `ok`, and the transcript records successful 35-test,
11-test comparison and five-test accounting suites, 51 checks total,
without a failure traceback. This verifies the transferred check evidence
for the IDs-plus-guard source boundary.

The Reviewer compared all 30 file hashes in `phase-e-source.json` against
the inspected local source: zero mismatches. The manifest identifies archive
SHA256 `303b94743ad557e70d064ecbef0263349ac472cabbfa743fb3d20b1284e6cc74`,
training job 2116767, eight CPUs, 64 GB, four Torch intra-op threads per
HF process and at most one concurrent game. The parent reports earlier
16-CPU attempts were cancelled before starting. One focal model plus one
direct opponent at four threads each fits this revised intra-op CPU budget;
this change preserves per-model threads while reducing game concurrency.
It is an explicitly recorded allocation rescope, not evidence from the
cancelled jobs.

The parent reports the first IDs-plus-guard collection game completed with
five sampled decisions acknowledged, no fallback or delivery error, and one
usable training episode. This is an operational start, not a completed
30-game training batch. The old single grounded pilot remains archived and
excluded from training. The new collection census, eventual selector's
training hashes and subsequent four-arm evaluation must remain separate.
No learned checkpoint outcome or policy promotion is accepted at this
stage; final runtime and held-out evidence are pending.

## Saved-update audit script review

The Reviewer statically inspected the new, standalone
`scripts/analyze_selector_update.py`. No material blocker was found for an
H2 run on the actual frozen fitting input. It independently assembles the
score-function gradient sum, averages by eligible games, applies the
recorded step and coordinate clipping, and compares saved weights, parent,
schema, match set/count and role counts. It hashes input files and eligible
logs. Exclusive output creation cannot replace checkpoints. Reported
coefficient changes and logged-slate KL/total variation/greedy changes are
explicitly in-sample distribution diagnostics, not estimates of match
returns or learning benefit.

This is a reproduction audit with shared eligibility, feature-transform
and probability helpers, not an independent revalidation of those helpers.
It requires the entire directory's eligible game set to equal the saved
training hashes. An additional eligible but unused episode arriving after
fit would therefore fail that strict snapshot audit without alone proving
the saved update wrong. Use the actual frozen fit-input directory, or
separately explain/reconcile extras against the recorded batch. Actual H2
reproduction evidence is still pending; no mirrored synthetic tests or
local model execution were requested.

## Semantic actor–critic/PPO proposal boundary

The Reviewer conditionally accepts an isolated semantic-selector experiment
in new files, pending the Planner's exact written state/interface contract.
The proposed frozen text encoder, shared candidate network, attention pool,
bounded residual actor and critic are a new architecture. They must not
silently replace the incumbent or be described as an established stronger
policy from thirty games.

Preserve the original feature values in float64 for the linear-base logits;
a separate float32 neural representation must not round-trip those values.
Zero-initialized actor residuals must reproduce the logged behavior
propensities before an update is allowed. The frozen local encoder must
use evaluation/no-gradient mode, pinned revision and versioned pooling,
normalization/truncation. Cache identity must include those settings and
candidate text. Exact context fields must use only acting-seat information
available at the decision: no terminal reward, winner, future transitions
or other seats' evaluation roles. Recorded own role can be used because it
was known to that actor.

The initial old-policy equality is necessary for reusing the frozen E cohort
as a clipped-PPO batch. Old probabilities and pre-update advantages remain
frozen across epochs; final evaluation data cannot select steps, architecture
or stopping settings. The proposed KL rollback must restore the full actor,
including its twenty linear coefficients. Optimizer-state handling must be
explicit if training resumes; halting after rollback is an acceptable simpler
boundary. Critic state after rejection must be declared separately.

A SHA-verified weights-only checkpoint, cheap schema validation and explicit
encoder/architecture identities are appropriate. Shared encoder inference
must preserve candidate/ID correspondence and deterministic legal selection.
Runtime integration, latency and new on-policy/prospective evidence remain
separate acceptance steps. In-sample coefficient, value-loss, entropy and
KL diagnostics verify optimization behavior; they do not establish stronger
play, and bounded KL is not a guarantee of improved game returns.

### Final semantic PPO contract choices

The Reviewer read the Planner's `14_semantic_ppo_contract.md`. The parent
then explicitly fixed entropy coefficient 0.001 and actor reduction as the
sum of decision surrogates within each game, averaged over games; advantages
remain fixed, unnormalized terminal-score advantages from the old zero
critic. Critic loss is decision-mean MSE. These final choices supersede the
initial ambiguous negative-mean wording and no-undeclared-entropy statement.
The Planner and Worker were notified to align contract and implementation.

With those explicit choices, the implementation contract is accepted for
new isolated files. Require an unequal-episode-length reduction check and
logged reduction/entropy settings. The original float64 prior equality,
detached critic representation, frozen encoder, whole-module and optimizer
rollback, categorical KL limits and exclusion of F data remain required.
Candidate/context embeddings add semantics only for the recorded coarse
state and offered labels; they do not create conversational evidence absent
from the dataset. Final code, H2 execution and prospective benefit remain
pending independent review.

## Semantic implementation initial inspection

The Reviewer inspected the first `semantic_selector.py` and
`train_semantic_ppo.py` plus root dispatch integration. The model preserves
original Python features in a sequential float64 linear prior and uses a
separate float32 semantic branch. The zero residual/value heads, detached
critic pool, local frozen mean/L2 encoder and explicit coarse-context
serialization match the contract. The trainer checks exact initial logits,
legacy propensity tolerances and zero old value, freezes terminal MC
advantages, applies game-averaged decision sums plus entropy 0.001, and
restores model and both optimizer states on rejected steps. Initial code
inspection found those core mathematics consistent with the agreed design.

A material runtime issue was raised: root's initial call to semantic
`choose` occurred outside a fallback boundary with no selector budget.
Encoder loading, corrupt state, nonfinite output or excessive latency could
therefore abort a decision. The parent agreed to add a semantic-only
exception/validity boundary, remaining total budget plus a five-second
selector cap, pre/post checks and logged selector errors/timing. A fallback
must have its actual deterministic propensity and `sampled=False`, while
proposer validity remains separately visible. Acceptance awaits that repair,
focused checks and measured cold/warm/full-roster latency.

Synchronous model work remains non-preemptive. For this isolated pilot,
a measured soft budget with honest overrun/fallback accounting is a
proportionate boundary; a persistent isolated encoder worker would be
needed for true hard interruption. No hard deadline or production-readiness
claim is accepted from post-hoc timing alone. The current H2 semantic driver
is explicitly a placeholder, not an executable delivery.

The semantic fitting start must be E's behavior checkpoint, not its
subsequent REINFORCE result. Otherwise the old-policy equality gate must
reject E's logged propensities. Shared fit-input eligibility helpers retain
the reproduction-audit limits noted above. Artifact validation and encoder
staging provenance, permutation/critic-isolation/KL-rollback checks and
actual new artifact reload remain pending.

## Semantic completed-source static pass

The completed semantic module adds shape/dtype/finite tensor checks,
weights-only loading, explicit tensor checksums and staged encoder-file
hash verification. It fixes eager fallback-path metadata access and checks
its soft deadline after lazy loading, embedding, neural computation and
host conversion. The eight focused real-Torch checks cover exact prior
logits, nonzero-residual candidate permutation, authorized context,
clipping signs, game weighting/fixed advantages/critic isolation, full
rollback with populated optimizer states, Torch-free artifact validation
and late embedding/host conversion.

Root's runtime repair now validates semantic indices and categorical mass,
catches loading/inference/timeout failures, uses the existing legal rules
alternative with deterministic probability, and marks failed selection as
unsampled fallback while retaining proposer-validity information. The
Reviewer requested a public-policy integration regression for error, late,
malformed and successful semantic selection, including delivery and log
flags; standalone model checks do not establish that boundary. Final freeze
acceptance awaits that focused integration check's source inspection.

The semantic H2 training driver now runs existing regression/accounting
checks, audits the separate linear reference, stages the pinned encoder,
runs semantic checks, fits exactly the thirty behavior episodes and reloads
the new artifact for real encoder/inference checks. The readiness driver
exercises two real HF processes and actual competition construction before
allowing the SDK collector. At this inspection its second worker uses the
hybrid direct-choice factory rather than the actual `LLMPolicy('direct')`
opponent, so it establishes two-model synthetic contention, not exact
mixed-roster gameplay timing. Cold/warm measurements, actual direct-opponent
load and the live delivery canary must qualify broader readiness claims.
No new game-strength or architecture-superiority evidence is asserted.

### Semantic final freeze acceptance

The Reviewer inspected the ninth semantic check, which exercises public
`choose_action` through semantic success, runtime exception, timeout,
post-return lateness, nonfinite mass, wrong-length mass and invalid index.
It verifies a legal rules fallback, actual one-hot fallback probabilities,
proposer-valid versus selector-error flags, `sampled=False` after failure,
accepted delivery callback, exact offered labels and only day/phase context.
Neural layers now explicitly request float32, preserving the separate
float64 base. These source checks close the requested integration gap;
H2 execution remains required.

The readiness benchmark now pairs the semantic competition factory with
an actual `LLMPolicy('direct')` model-call worker. Review found an initial
unconditional `_speech` call on that direct policy, which has no such
method; the root repaired it to construct hybrid speech options only for
the semantic worker. The repaired model-call outputs and logging fields
were re-inspected. No material source blocker remains for the isolated
H2 snapshot, checks and training after the frozen E cohort completes.

This grants static freeze acceptance, not test success or launch/promotion
based on unexecuted code. The intended gates remain successful H2 checks,
real frozen-encoder/artifact reload, initial-policy equality, recorded PPO
trust bounds, cold/warm two-model readiness and a fresh live-delivery
canary. The benchmark uses synthetic observations and model calls; it does
not itself establish SDK delivery or stronger tournament play.

## Semantic evaluation batch pre-submission inspection

The Reviewer inspected `evaluate-semantic.sbatch`: all three incumbent/
guarded/sampled controls point to the original behavior checkpoint, while
the adapted profile points to job 2122369's schema-2 semantic artifact.
The proposer, 1-direct/5-rules roster, CPU BF16 revision and four-thread
per-model setting remain fixed. Global greedy mode plus
`WEREWOLF_EVALUATE_ONLY=1` prevents training even though individual profiles
select sampled mode. The driver checks twelve completed games per arm
both inside and after its loop. Forty-eight is a minimum total: UUID-hashed
arm imbalance may require additional matches to reach all four minima.
Report actual assigned/completed counts rather than asserting exactly 48.
The requested `afterok:2122369` dependency is supplied by submission CLI,
not embedded in this file.

The Reviewer requested a trained-artifact preflight before any F collector:
verify accepted PPO epochs are positive, initial parity checks passed and
saved categorical KL bounds hold. A first-step KL rejection intentionally
restores the original policy and can still yield a valid zero-update
artifact with process exit success. Such an artifact should not silently
be called a trained PPO candidate or trigger a costly strength comparison;
a zero-update control would need explicit separate interpretation.
Submission is otherwise statically acceptable subject to that launch check,
the existing runtime-readiness gate and actual H2 evidence.

All 35 local source hashes in `phase-f-source.json` match the frozen
manifest at commit `8cfcbfcc6511a20906e982c3fed9de439d033a4a`; source
archive SHA256 is
`8f63eb65f50b66b2871071eb658e088c849e147611c6795ac459a6ce527a68b8`.
This is source correspondence, not execution success or strength promotion.

## First semantic fit failure and exact-prior repair assessment

The parent reports job 2122369 passed the 35/11/5 existing checks, nine
semantic checks and the linear-reference audit, then failed before any PPO
update at exact initial-logit equality. All 136 logged-slate probability
parity checks reportedly passed. The failed source and outputs are to remain
archived; no F matches or parameter updates occurred. Independent inspection
of the transferred failure transcript is pending.

The initial fixture used the two-nonzero-coefficient built-in prior and
missed the real twenty-coefficient accumulation difference. This limitation
was missed in the earlier static acceptance. The strict real-data gate
correctly prevented training. Sequential Torch additions are not guaranteed
to reproduce Python's built-in floating sum: Python documents a float-sum
algorithm change in 3.12 ([official Python documentation](https://docs.python.org/3.12/library/functions.html#sum)).
That supports the proposed diagnosis but is not by itself measurement of
the failed rows' residuals.

The Reviewer conditionally accepts the proposed repair: compute the exact
legacy Python primal from **current**, detached float64 weights, and add
`(linear - linear.detach())` as a zero-valued differentiable bridge. The
parentheses are essential: form the zero before adding it to the primal,
rather than evaluating primal-plus-linear-minus-detached in sequence.
The forward value then follows the legacy implementation while the tangent
is the linear design row. This is an explicit numerical/autograd bridge,
not tolerance relaxation or a change to the initial policy target.

Require twenty nonzero coefficients with fractional features, exact primal
checks before and after modifying weights, and a regression verifying base
coefficient gradients. Version the accumulation rule and record Python
runtime identity. Keep the real 136-slate equality and existing probability
tolerances unchanged on the isolated retry. New code and actual retry
results remain pending; cancelling dependent evaluation is appropriate.

### Exact-prior repair final source acceptance

The Reviewer inspected all three repaired files before isolated r03g
retry. `forward` now builds the Python legacy primal from current detached
weights and the untouched design rows, then adds the explicitly parenthesized
zero-valued differentiable bridge. The architecture is versioned v2 and
records the accumulation rule. The added fixture uses archived D7 and
all-twenty-nonzero coefficients with fractional features across seer/wolf/
speech contexts; it checks exact logits, exact design-row base gradients
and exact primal values after changing weights.

The trainer retains the strict real-state `torch.equal` gate and unchanged
legacy propensity tolerances. It separately reconstructs the old sequential
Torch prior across all recorded states, reports its maximum error against
Python before fitting, and records Python implementation/version plus Torch
version. That diagnostic can substantiate the suspected accumulation cause
without changing the policy target. No material source blocker remains for
this new frozen retry. H2 execution and the measured residuals remain
pending; the first failure must not be rewritten as a successful update.

## Successful semantic retry: independent artifact audit

The Reviewer inspected the transferred JSON metadata/reports, tensor-file
hash and log from H2 retry job 2123930 without executing neural computation
locally. The transcript records 35 focused, 11 comparison, five accounting
and ten semantic checks before fitting, and ten semantic checks after
fitting, all successful.

The recorded batch is exactly the thirty hashed matches in the frozen E
controller census and the separate linear-reference artifact. Every one
of the thirty reported focal-log SHA256 values matches its local source.
Those logs contain 142 selected decisions, 136 sampled decisions and 142
accepted deliveries, with zero recorded delivery errors/fallbacks. Roles
are eleven wolf, seven seer and twelve villager games. All sampled choices
share the recorded IDs-plus-guard generator. The starting checkpoint's file
hash matches D7, and the artifact parent is `d7ebf7670444067a`. No F matches
are part of this recorded batch; thirty terminal games remain thirty
observations despite multiple choices/epochs.

All 136 initial logits now match the legacy prior exactly. Maximum initial
probability error is 1.1102230246251565e-16 within the unchanged tolerance.
The separately reconstructed former sequential prior differs by up to
4.440892098500626e-16 on these actual states under CPython 3.12.13. This
confirms the numerical implementation mismatch, consistent with the
previously documented summation explanation; no threshold was relaxed.

All 32 epochs were accepted and satisfy the predeclared training-state KL
limits. Final old-to-new categorical mean KL is 0.005937964267001589 and
maximum is 0.0350332823767123. The artifact records the agreed clip 0.2,
entropy 0.001, game-averaged actor/entropy sums, decision-mean critic MSE,
fixed unnormalized MC advantages, learning rates, seed, full rollback rule
and runtime versions. Reports confirm frozen encoder and unchanged
advantages. These establish optimization and provenance, not improved
playing strength.

The metadata equals the report's artifact copy, and `selector.pt` matches
its recorded SHA256. Independently deriving the checkpoint identifier from
metadata bytes and tensor hash yields `6ca4df8fd98821e6`, matching the
post-training reload report. All 35 phase-G manifest hashes match the
reviewed local files at commit `63bb8ddd0706565cc751d2ae97dbbac476ed6ab9`,
archive SHA256
`9c97d9b2aafcc7aa139c5b7cba6381422bb96c64cff1d36190de44d9283d4098`.

Training artifact acceptance is granted. Cold/cached component timings in
the training allocation are informative only; the separate two-HF runtime
screen, actual delivery canary and prospective held-out comparison remain
required before launch/promotion claims. No policy is promoted from this
training result alone.

## Read-only semantic distribution audit source review

The Reviewer inspected the new `analyze_semantic_update.py` without running
neural computation locally. Static acceptance is granted for the planned
H2 E-only audit with `--training-report`. Before neural evaluation it checks
the exact census/eligible/artifact match set, parent ID/file hash, copied
training artifact and each fitting log's relative path/SHA256. Encoder
relocation preserves pinned revision and recorded file-hash checks.

The script evaluates full categorical old-to-saved-NN KL, total variation
and stable greedy changes on all eligible sampled training slates, split by
role/kind. It reports actual named actor/critic parameter counts, the
zero-initialized head's change, base-coefficient deltas and frozen encoder
identity, then reconciles training KL with declared numerical tolerance.
Role/score labels remain reporting data, not model inputs. Exclusive output
creation does not replace inputs, and no fitting or F access occurs.

The audit reuses the frozen eligibility/probability helpers and is therefore
a reproducibility/distribution audit, not independent validation of every
shared helper. Omitting the optional training report weakens per-log hash
provenance; the actual run should supply it. Parameter counts do not increase
the thirty independent terminal observations, and logged-slate movement
cannot establish counterfactual wins. Actual H2 audit results remain pending.

## F operations helper initial static review

The Reviewer inspected the new standard-library-only
`analyze_strength_operations.py`. It joins supplied assigned-arm/focal-seat
records to all worker logs, checks complete-policy profiles and fingerprints,
keeps E/F overlap empty, validates logged actual probability mass/sampling
flags and counts delivery/selector/direct-opponent errors without removing
assigned games. Guard usage depends on actual before/retained slate change.
It reports integrity failure, incomplete cohort and completed cohort through
distinct exit statuses, with outcome-free status stdout.

One configuration gap was raised before acceptance: the warm-generator
normalization replaces actual CPU threads by the requested count, so a
self-consistent two-thread call with a four-thread request could incorrectly
match the four-thread E counterpart. Require the explicit requested count
of four, actual count four or unloaded as appropriate, and model-backed
choices with actual count four. Digest consistency alone is insufficient
for runtime homogeneity.

Controller assignment integrity is inherited from the supplied summary and
its protocol checks; this helper does not independently recompute raw UUID
assignment. Run against a stable between-games snapshot: an incomplete live
JSON tail is currently classified as integrity failure, rather than merely
an unfinished game. SHA manifests preserve the inspected log snapshot.
No policy selection, fitting or return claim is performed by this helper.

### F operations helper final acceptance

The Reviewer verified the thread repair: requested threads must be integer
four, actual threads are constrained to four or unloaded, and loaded BF16/
model-valid proposals require four. A model-load failure after applying
threads may correctly report actual four with unloaded dtype and fallback.
The helper additionally separates forced-singleton choices, proposer error
types and valid-proposal call latency while retaining all-choice timing.
No material source blocker remains for an H2 run on a stable between-games
snapshot. An integrity-clean partial canary should return exit 2 for
incompleteness; that is not an integrity failure or a basis for outcome-
dependent cohort stopping. Actual run evidence remains pending.
