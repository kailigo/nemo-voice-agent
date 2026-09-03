# Current issues: sft_train8_0826/step-500 on τ-voice airline

**As of 2026-09-03.** Reference basis for what we are trying to fix next. Each issue
has a receipt (episode id + observed call/output). Do not delete claims — mark them
if superseded.

Prior context, still valid where noted:
- Older wide-taxonomy doc from 2026-08-21: [`NEMO_FAILURE_MODES.md`](NEMO_FAILURE_MODES.md).
  Modes described there predate the ckpt this document tracks; use its methodology
  section, treat its per-mode counts as historical.
- Program-plan direction (expand/distill/SFT/RL): [`TAU_VOICE_SFT_RL_PROGRAM.md`](TAU_VOICE_SFT_RL_PROGRAM.md).
- Training-data pipeline for reference: [`SFT_DATA_TO_TENSORS.md`](SFT_DATA_TO_TENSORS.md).

## Scope of this document

- **Model:** `logs/sft_train8_0826/exp/checkpoints/step-500.ckpt`, trained on
  `data/tau2_training_samples/mix_full86/train` (80 airline+retail cuts, 2-20 tool
  calls each). Repackaged for eval as
  `/fsx/home/kai.li/data/voicechat/sft_step500_airline_retail`.
- **Base checkpoint compared against:** `/fsx/home/kai.li/data/voicechat/stt_extracted_lora`
  (extracted from `nemotron_voicechat_11b`).
- **Live batch:** `logs/eval_batch_0901_nemo-{base,sft-500}_t{0..15}` — 16 airline
  tasks, SilenceTTS agent side + real ElevenLabs user side + telephony bandpass,
  complexity `control`.
- **Batch outcome:** base 2/16 successes (t4, t5), sft-500 0/15 successes (t8 missing).
- **Two teacher-forced eval scripts used below:**
  `scripts/eval_forward_id_copy.py`, `scripts/probe_head_coupling_tf.py`.
- **Two free-running probe scripts used below:**
  `scripts/probe_head_coupling.py`, `scripts/mode_a_readout1.py`.

---

## Issue 1 — ID mistranscription on long spelled identifiers

**Failure count:** 8/15 sft-500 failures + at least 1 in base.

**Symptom:** the model fires a tool call with a wrong user_id or reservation_id
argument. The user spelled the correct id letter-by-letter (often 3-4 times); the
model consistently emits a mistranscribed version.

**Receipts (sft-500):**
- t0: called `emma_kim_99`; true `emma_kim_9957`.
- t1: called `raj_sanchez_7734`; true `raj_sanchez_7340`.
- t6: called `sarah_silva_8702`; true `sophia_taylor_9065`.
- t9: called `get_reservation_details({'reservation_id': 'IASFOY'})` → `'OY'` → `'LY'` → `'Y'` — progressively degraded across retries.
- t10: called `liam_khan_225521`; true `liam_khan_2521` (digit dup).
- t12: called `NELLIE_LEDERT_225`, `CHELINA_ELLIS_225`; true `chen_lee_6825` (severe corruption).
- t14: called `mohamed_silva_9226`, `mohamed_silva_filbert_9226`; true `mohamed_silva_9265`.
- t15: called `rav_garcia_1`; true `aarav_garcia_1177` (truncation).

**Also visible under a controlled offline probe** on training-cut audio
(`11_bd8c4d4e`), sft-500 emits `james_patel_9822` for true `james_patel_9828` under
clean 16kHz audio; the same cut telephony-band-limited yields
`james_underwater982`. Real ASR failure on real synthesized speech; audio-band
filtering demonstrably makes it worse.

**Format of vulnerable ids:** `firstname_lastname_NNNN` (13-18 chars, 2 underscores,
mixed letters + digits). All 50 airline tasks and 49/50 use this shape.

**Not fully explained by mistranscription alone:** some models (base t11, base t13)
also emit *placeholder-shaped* ids (`user_1234`, `JAN001`, `HAT030`) with no matching
audio content. This is **argument hallucination**, tracked as [Issue 3](#issue-3--argument-hallucination-without-audio-grounding);
distinct from ASR-level mistranscription but often gets mixed with it in analysis.

**Broader context — this is not a τ-voice-specific weakness.** The base checkpoint's
own reported numbers on Full-Duplex-Bench v3 (a different tool-calling audio
benchmark) are Tool Selection **82.5%** but Argument accuracy **44.2%** — under a
*lenient* gpt-4o semantic judge (e.g. "August 20" == "2026-08-20"), on shorter
alphanumeric ids like `A-BC123`. See [`FDB_V3_REPRODUCTION.md`](FDB_V3_REPRODUCTION.md)
for our reproduction of these numbers. The published example is verbatim our failure
mode: *"track_order({"order_id": "A-BC123"}) for a user who spells 'a B C one two three'
out loud — right tool, near-miss argument."* τ-voice airline ids are longer (16
chars with underscores vs 6), and τ-voice scores argument correctness by *exact
match at the tool server* (the tool returns "user not found" for near-misses; no
lenient judge). Our 0/8 exact-match rate is consistent with (and worse than) the
44.2% FDB baseline given the harder id shape and stricter scoring. **The
audio→argument transcription pathway was never "fine"; it was the weaker of the
model's two tool-calling capabilities at release** — 82.5% naming vs 44.2%
arguments. Fixing this is a pathway improvement, not a τ-voice-shape data patch.

**Reframing:** the initial "add more spelled-id cuts" plan is limited because the
existing 80 training cuts already saturate the `firstname_lastname_NNNN` pattern
(63 unique user_ids in that shape). Adding more cuts of the same shape doesn't
attack the actual weakness — the ASR-into-argument pipeline being unreliable on
spelled content in general.

**Proposed fixes, in order of effort vs. certainty:**

*Fix A — Band-match training and inference audio (cheapest, direct).* Training data
is 16 kHz clean (~8 kHz spectral bandwidth); inference at eval time is 8 kHz
telephony (~4 kHz bandwidth after 300-3400 Hz bandpass). Our offline probe directly
showed the band matters: same cut, same model, wideband produced `james_patel_9822`,
telephony-filtered produced `james_underwater982`. Preprocessing training audio
through the same telephony bandpass + 8 kHz roundtrip so training-time and
inference-time distributions match is a data-preprocessing change; no model or loss
change. Cheap enough to test in one training cycle.

*Fix B — Turn on auxiliary ASR loss during SFT.* The model has an `asr_head` and
`predict_user_text` support. Current yaml has `asr_loss_weight: 0`. Adding this
objective — with clean word-level transcriptions of the user turns as ground-truth
(which we already have from the τ-voice user simulator's LLM outputs pre-TTS) —
gives the audio-fusion layers direct gradient signal to improve internal
transcription quality. Medium effort.

*Fix C — Behavior workaround: confirm-before-call.* Instead of fixing ASR directly,
train the model to read user-provided identifiers back to the user before firing
the tool call, and only fire after the user confirms. Sidesteps the exact-match
requirement — the user catches mistranscriptions in the conversation. Adds one
turn per lookup but matches real customer-service norms.

  **Design principle:** confirm user-*spoken* content, not model-*derived* content.

  | confirm | don't confirm |
  |---|---|
  | `user_id` when user spelled it | `cabin: 'economy'` (model chose from user's request) |
  | `reservation_id` when user spelled it | `insurance: 'no'` (model's decision) |
  | `email` when user spelled it | `payment_method_id` if it came from a prior tool response |
  | dates, dollar amounts, phone numbers the user gave | flight_number the model found via search |
  | passenger `dob`, `first_name`, `last_name` | any enum the model picked between |

  A naive "confirm every parameter" agent would be unusable; the training regime
  must distinguish user-spoken (ASR-vulnerable) from model-derived (not
  ASR-vulnerable) arguments.

  Implementation options for Fix C, ordered by effort:

  - *C-a — tool-schema annotation + harness enforcement.* Add
    `confirm_required: true` to each argument in tool schemas. A wrapper inserts a
    confirmation exchange before the tool fires when any argument is marked.
    Zero training data or model changes; mechanical, stilted, model doesn't learn
    the behavior. Cheapest defensive fallback.
  - *C-b — modify the frontier-model rollout collection (recommended entry
    point).* Update the frontier-model system prompt used during trajectory
    collection to instruct read-back-and-confirm for user-spoken ids. Successful
    trajectories then naturally include the confirmation pattern; our model
    learns it by imitation. One-line prompt change plus a sample check that the
    frontier is following the instruction. Data cost is proportional to the
    number of tool calls per cut × ~5-10s per confirmation.
  - *C-c — post-process existing cuts programmatically.* For each existing cut,
    match tool-call arguments against the preceding user transcript; where a
    match exists, synthesize a new confirmation turn before the tool call and
    splice it into the audio. Cleaner training data than C-b but requires audio
    re-synthesis with a matching voice and Lhotse cut restructuring. More
    engineering.
  - *C-d — RL/rollout with a "confirm-before-call" reward.* Reward tool calls
    only when critical arguments were confirmed with the user beforehand.
    Requires the rollout infrastructure that Issue 2 already needs. Most
    principled, most effort.

*Fix D — Diverse spelled-content pretraining data at scale (out of scope here).*
The truly fundamental fix: mid-training on much more speech involving alphanumeric
spelling (emails, product ids, phonetic-alphabet exchanges). Would improve base
ASR quality generally, benefits every benchmark. Major data + compute effort, not
slottable into an SFT-scale run.

**Recommended path:** Fix A + Fix C-b in parallel. Fix A directly attacks the
distribution mismatch we can measure and costs one training cycle; Fix C-b makes
the model tolerant to residual ASR errors even after A is in place, via a behavior
change that matches real phone-agent norms. Fix B is a cleaner backup if the
auxiliary ASR loss can be wired in without disrupting the existing objectives.

**Question worth deciding before committing to C:** is confirm-before-call a
*permanent* production characteristic (better UX for phone agents, defends against
any residual ASR error) or just a training-time crutch to bump numbers? If
permanent, C-b is straightforward; if crutch, A + B may suffice and the added
turn per lookup may not be worth it.

**Open sub-question retained:** whether the band-match training (Fix A) also
lifts free-running tool-call firing (Issue 2), since the audio distribution at
SOTC-firing moments would also change. Not tested yet — worth measuring after
any A-style retrain.

---

## Issue 2 — Trajectory drift in free-running multi-step rollout

**Failure count:** 4-6 of 15 sft-500 failures show the specific pattern; the
mechanism is likely present in more cases as an amplifier.

**Symptom:** the text head produces coherent, on-policy language including phrases
like *"Let me check your reservation now"*, *"I found your profile"*, *"For a gold
member, I can offer 300 dollars per passenger"* — but the function head emits zero
`<SOTC>` tokens. The narration proceeds as if tool calls happened; no tools were
called; specific policy numbers get fabricated ($300/passenger, $1200 max).

**Receipts (sft-500):**
- t3: 2262 chars of "please spell your user ID... I'm pulling up your profile..."
  across 1619 ticks, zero tool calls.
- t11: 8 words of coherent agent text, 0 tool calls (though 4/15 sft-500 failures
  fire nothing at all; here the function head also stays silent).
- t13: 1 unrelated `search_direct_flight` call, then no further fires.
- t5: 545 words of policy narration, 0 tool calls; explicitly says *"I found your
  profile and past reservation details"* without ever firing a tool.

**Measured mechanism (direct evidence, not speculation).** On cut
`11_bd8c4d4e` (training cut, 8 scripted SOTC positions):

| condition | SOTC top-1 rate at scripted positions | fires in free-running (same audio) |
|---|---|---|
| BASE, teacher-forced | 0/8 (0%) | 1 (free-run) |
| **SFT-500, teacher-forced** | **8/8 (100%)** | **1 (free-run)** |
| SFT-500, free-run (clean 16kHz) | — | 1 fire out of 8 scripted |
| SFT-500, free-run (telephony) | — | 1 fire out of 8 scripted |

Under teacher-forcing SFT-500 predicts SOTC correctly at every scripted position.
Under free-running on identical audio, only 1 of 8 fires. **The training does put
SOTC firing into the head — free-running loses 7/8 of those fires because the
model's own generated prior context drifts from ground-truth just enough to shift
the LLM hidden state past the small SOTC-vs-PAD margin (3-6 logit units under
teacher-forcing).** This is exposure bias, measured on our exact setup.

**Also measured:** SOTC is a top-5 candidate at 75–98% of *all* free-running
positions (base 75.5%, sft-500 98.3%). The head is not silent — it consistently
considers firing, but PAD beats SOTC by ~12-14 logit units median in free-running.

**Proposed fix:** expose the model to its own generated intermediate context
during training. Scheduled sampling on the function-channel path (mix in the model's
own argmax previous function-channel tokens as inputs with some probability) is
the minimal intervention. Full rollout-style / RL training is the maximal
intervention. Both directly attack the teacher-forced-vs-free-running gap; the
current SFT loss cannot see this gap because each head is graded independently on
its own labels at each position.

**Not a fix:** more data, smaller LoRA rank, or fewer training steps. Under
teacher-forcing the head is already 100% correct; less training just makes it less
confident, not more free-running-robust.

---

## Issue 3 — Argument hallucination without audio grounding

**Failure count:** at least 3-4 of the 15 sft-500 failures + several base failures.

**Symptom:** the model fires a tool call whose argument value was never spoken by
the user in any form. The value has a "placeholder" character — round numbers,
generic strings — rather than a mistranscription of something the user said.
Distinct from Issue 1 in that no ASR opportunity exists: the model invents.

**Receipts:**
- Base t11: `update_reservation_baggages({'reservation_id': 'HAT030', ...})` when
  user wants to remove a passenger (not modify baggage) and never mentioned
  reservation `HAT030`. Called 6 more times identically. Stuck-loop terminated.
- Base t13: `get_user_details({'user_id': 'user_1234'})`, then `'user_123'`, then
  `'JAN001'` — all before the user provided any id.
- Base t13: also fired `update_reservation_flights({'reservation_id': 'ZFR04Y'})`
  where `ZFR04Y` is invented; true is `XEWRD9`.
- Base free-running on the offline probe (cut `11_bd8c4d4e`): fired
  `get_user_details({'user_id': 'jane_1234'})` — a placeholder-shaped fake with
  no relationship to the true `james_patel_9828`.

**Why it matters:** more spelled-id training data (Issue 1's fix) will improve
transcription accuracy when the user IS speaking; it will not necessarily prevent
the model from confidently inventing arguments before any spelling happens.
Argument hallucination has a separate cause and needs a separate signal.

**Proposed fix (tentative):** an explicit training constraint that tool-call
arguments must derive from the user's spoken content. Concretely — grader/reward
signal that penalizes tool-call arguments containing tokens that do not appear
verbatim in the immediately-preceding user utterance's transcript. Requires
rollout-style training to apply (Issue 2's fix is a prerequisite). Or — negative
training examples where the model is shown "user has not spoken → do not call a
tool with a fabricated argument, ask for the id instead."

**Open sub-question:** how much of what looks like Issue 3 is actually
underdetermined base behavior that SFT is failing to correct because SFT has never
seen "ask, don't guess" as a labeled action.

---

## Issue 4 — Total-agent-collapse from tick 1

**Failure count:** 2 of 15 sft-500 failures (t4, t7). Distinct from either 1 or 2.

**Symptom:** the model produces zero audio-active ticks and zero words across the
entire episode (611 ticks for t4, 599 for t7). User simulator says *"Hello? Is
anyone there?"*, the agent never responds, `Provider inactivity detected` fires
repeatedly at ~40-second intervals, episode times out at 1800s wall clock.

**Receipts:**
- sft-500 t4: 611 ticks, 0 audio ticks, 0 words, 0 tool calls. Base t4 handled
  the exact same task with 498 words of coherent policy-refusal and won reward=1.0.
- sft-500 t7: 599 ticks, similar profile.

**Not explained by either Issue 1 or Issue 2:** the model never even begins a
generation trajectory. Nothing to drift, nothing to mishear.

**Not measured yet:** cause. Could be an initial-state issue triggered by a
specific opening audio signature; could be a rare interaction between the SFT
weights and a particular user-audio prefix. Small in count but severe when it
happens (0 reward, no partial credit possible).

**Proposed probe (not a fix yet):** feed sft-500 the exact first tick of user
audio from t4 and t7 through the offline streaming path, log the first ~20 LM
positions' text and function head argmaxes. Compare to a base-model run on the
same audio. Does sft-500 emit BOS/greeting on t5 or any other task's first tick?
If yes, what's different about t4/t7's audio in the first 1-2 seconds?

---

## Priorities and sequencing

Ranked by expected reward-move per unit effort, given what's currently measured:

1. **Issue 2 (trajectory drift)** — the mechanism is now directly measured and
   cleanly attributable. The intervention (scheduled sampling / rollout) is the
   biggest architectural move but has the clearest evidence supporting it.
   Prerequisite for Issue 3's proposed fix.
2. **Issue 1 (audio→argument transcription weakness, not τ-voice-specific).**
   FDB-v3 published Argument accuracy 44.2% under a lenient judge already
   confirms this is a base-model limitation, not a τ-voice artifact.
   Recommended entry: Fix A (band-match training/inference audio) + Fix C-b
   (frontier prompt update in rollout collection for confirm-before-call
   behavior). Fix A is one training cycle; Fix C-b is a one-line change to the
   collection pipeline on the gateway machine plus a data sample verification.
   Parallelizable with Issue 2 groundwork.
3. **Issue 3 (argument hallucination)** — dependent on Issue 2's infrastructure
   (rollout-style training) or requires targeted refusal/ask-first training data.
   Third in line unless a cheaper labeled-data workaround is found.
4. **Issue 4 (total collapse)** — smallest count, no measured mechanism yet. Run
   the diagnostic probe before deciding whether it needs its own intervention.

## Cross-cutting questions still open

- **Real-TTS ablation of the batch.** Live batch used ElevenLabs for user side but
  SilenceTTS for agent side. Numbers are internally consistent for base-vs-sft-500
  A/B but the harness's own warning applies to absolute reward. Sanity-check with
  a small NeMoTTS agent-side batch (KEY_3 has ~131k ElevenLabs characters left)
  before final reporting.
- **Audio-band mismatch magnitude.** Training audio is 16kHz clean (~8kHz spectral
  bandwidth). Live-batch audio is 8kHz telephony (~4kHz bandwidth). Direct probe
  showed `patel` transcribes as `underwater` under telephony filtering. Data-side
  fix would be to telephony-filter training audio to match; not yet tested.
- **Whether Issue 3 is a downstream symptom of Issue 2.** Argument hallucination
  and function-head-fires-without-context could both be consequences of the same
  free-running drift. If Issue 2's fix reduces hallucination, Issue 3 becomes an
  observation not a separate item.

## Changelog

- 2026-09-03 — Initial version, based on the base-vs-sft-500 airline batch and
  offline probes. See conversation transcript at
  `/fsx/home/kai.li/.claude/projects/-fsx-home-kai-li-code-nemo-voice-agent/b1e4e3e6-cd5e-4d2e-91e8-1857a0b426fd.jsonl`
  for the evidence trail behind each claim.
- 2026-09-03 (later, same session) — Reframed Issue 1. Original framing
  ("prepare more spelled-id training data") was pointed out to be wrong on two
  counts: (a) the existing 80 training cuts already saturate the
  `firstname_lastname_NNNN` pattern (63 unique ids of that shape), so "more of
  the same" doesn't fix the actual weakness; (b) FDB-v3 published numbers
  (Tool Selection 82.5% vs Argument accuracy 44.2%) show the base model's
  audio→argument transcription pipeline was already the weaker capability at
  release, not a τ-voice-specific gap. Rewrote Issue 1 to reflect this: four
  proposed fixes (A band-matching, B auxiliary ASR loss, C confirm-before-call
  behavior with sub-options C-a/b/c/d, D out-of-scope diverse-content
  pretraining). Recommended path is A + C-b in parallel. Updated the
  priorities section accordingly.
