# Plan: SFT of Nemotron VoiceChat 11B for the τ-voice benchmark

Status: **draft, agreed 2026-08-14.** Supersedes the ad-hoc "6-item data fix list" from the
same day.

Goal: improve `nemotron-voicechat` (our `DuplexSTTModel` 11B) on
[τ-voice](https://sierra.ai/blog/tau-voice-benchmarking-real-time-voice-agents-on-real-world-tasks)
by supervised finetuning on successful frontier-model trajectories collected from the
τ-voice harness itself.

Two repos are involved:

| repo | role |
|---|---|
| `/fsx/home/kai.li/code/nemo-voice-agent` | model, training config, data converter |
| `/fsx/home/kai.li/code/tau-voice-2` | benchmark harness, domains, trajectory collection |

---

## 1. Experimental design

Three arms. All three are required — the cross-domain number is uninterpretable alone.

| arm | train on | test on | purpose |
|---|---|---|---|
| **A — baseline** | nothing (released 11B) | retail, airline, telecom | Is there anything to improve? Where does it fail? |
| **B — upper bound** | retail, airline, telecom (held-out *tasks*) | same 3 domains | What does the recipe buy when domain knowledge is free? |
| **C — cross-domain** | new domains only | retail, airline, telecom | The real claim: does tool-use skill transfer? |

Model selection (checkpoint choice, early stopping, LR) uses a **4th held-out new domain**,
never the 3 test domains. Otherwise B and C stop being held out.

### Success criteria

Primary: task success (reward = 1.0 rate) on the 3 test domains, C vs A.

Secondary, and not optional — the benchmark scores these explicitly
(`tau-voice-2/src/tau2/metrics/voice_interaction_metrics.py:8-19`):

- `response_latency_mean` (L_R) — user turn end → agent response
- `yield_latency_mean` (L_Y), `yield_rate` (R_Y) — does the agent stop when interrupted
- `agent_interruption_rate` (I_A)

Also track: tool-call well-formedness (parse rate, valid tool name, schema-valid arguments).
This is where the largest gain is expected and it is invisible in the reward alone.

### Expected outcome, agreed up front

**Transfers across domains:** telephony-band acoustic robustness, turn-taking and latency,
tool-call syntax and slot-filling discipline, the habit of consulting the policy.

**Does not transfer:** domain knowledge, and specific policy rules. Much of τ²-bench reward is
policy compliance ("authenticate first", "cancel only if pending"); the policy is in the prompt
at eval time, so what must generalize is *reading and obeying an unseen policy* — hard to
instill from tens of hours.

So: expect a large jump in tool-call well-formedness and audio robustness, a real improvement in
latency metrics **if** Phase 2 timing fixes land, and a **modest** gain in task success. A modest
success-rate gain is the predicted result, not a failure.

---

## 2. Measured facts this plan rests on

Everything below was measured on 2026-08-14, not inferred. Cited so it can be re-checked.

**Harness / audio**

- Tick rate is **200 ms**; all label timestamps are `start_tick * tick_duration_ms / 1000`
  (`tau-voice-2/src/tau2/voice/synthesis/conversation_builder.py:176`). Hence every timestamp is
  an exact multiple of 0.2 s. The model's frame is 80 ms (12.5 Hz), so **200 ms = 2.5 frames —
  not an integer.** See Phase 0.
- Label/audio alignment is **exact**, despite labels coming from the tick grid and `both.wav`
  from sample concatenation with `silence_duration_ms=None`. Agent audio is padded to exactly
  `bytes_per_tick` with `TELEPHONY_ULAW_SILENCE`, excess buffered forward
  (`src/tau2/voice/audio_native/adapter.py:58`). Verified: cut duration == last supervision end
  on all 3 cuts, at exactly 998 / 1053 / 787 ticks, zero off-grid boundaries.
- Transport is **8 kHz μ-law telephony**, `telephony_enabled` default `True`
  (`src/tau2/data_model/voice.py:194`), 300–3400 Hz bandpass. Measured on our built shards:
  user track **0.0003 %** of energy above 4 kHz, agent track **0.028 %**. The wideband
  containers (16 kHz / 22050 Hz) are empty above 4 kHz.
- The constant 1.20 s agent→user gap is exactly 6 ticks — a harness constant, not behaviour.

**Teacher trajectories (3 existing cuts, `/fsx/home/kai.li/data/voicechat/tau2_fixed/`)**

- All 3 are **retail task_0** with the same 5–6 tool sequence, from two Gemini versions.
  Effectively one scenario, three takes. 0.158 h total.
- Teacher is audio-native Gemini Live, **not** a cascade. Its 2.13 s mean / 4.60 s max response
  latency is API round-trip plus tick quantisation. Still not worth cloning.
- Tool call → result gap is **0.0 s for all 6 calls** (in-memory DB). `<TOOLCALL>` and
  `<TOOL_RESPONSE>` share a timestamp.
- **One** barge-in event across all 3 cuts.

**Tool schemas — the largest data defect**

`scripts/episodes_to_nemotron_training.py::extract_tool_schemas` reconstructs schemas from
observed calls only. Measured on `0_4922ce6f`:

- **5 tools advertised; retail has 16.** (airline 14, telecom 21.)
- **Zero `description` fields**, on tools or parameters. Real retail `tools.py` docstrings are
  10,588 chars ≈ 2,650 tokens.
- `required` lists every observed argument, so optional params are marked required. The schema is
  *wrong*, not merely thin.

**Model**

- `target_audio` is only "saved for debugging" (`duplex_stt_model.py:651`), never a loss target.
  Agent-track audio quality is irrelevant to STT training.
- `max_fc_total_tokens: 8000` (`examples/speechlm2/conf/finetune/s2s_duplex_stt_11b.yaml:122`)
  **drops the whole cut**, it does not truncate. Current FC totals ≈ 5.0–5.2 K.
- Trainable: 2.41 B of 10.13 B (see memory `stt-11b-trainable-params`). `perception` (0.61 B) is
  trainable, which is what makes telephony adaptation possible.

**Task inventory**

| domain | tasks | role |
|---|---|---|
| retail | 114 | test |
| airline | 50 | test |
| telecom | 2285 | test |
| banking / calendar / healthcare / … (11 new, commit `69264a4`) | 8–10 each | train |

---

## 3. Phase 0 — Evaluation harness (CRITICAL PATH, do first)

**Nothing can be measured today.** There is no baseline number for the released 11B, and no way
to produce one.

### 0a. Register a NeMo agent provider in tau-voice-2

`AudioNativeProvider = Literal["openai","gemini","xai","nova","qwen","livekit"]`
(`src/tau2/agent/discrete_time_audio_native_agent.py:85`), dispatched by an if/elif chain at
`:271-293`. Adding `"nemo"` means: extend the `Literal`, add an `elif` branch, register in
`create_adapter` (`src/tau2/voice/audio_native/adapter.py:372`), and implement a
`DiscreteTimeAdapter` subclass with the abstract methods `connect`, `disconnect`,
`is_connected`, `run_tick`, `_execute_tick`, `_flush_pending_tool_results`.

### 0b. Build an incremental, FC-capable inference driver (the real work)

Neither existing entry point can serve a live eval:

| path | streaming | function calling |
|---|---|---|
| `offline_inference` (`:4911`) | no — whole `input_signal` up front | **yes**, but `function_responses` must be pre-baked from ground truth |
| `online_inference` (`:5072`) | sliding window, but loops internally over all audio | **none** — no FC parameters, never references `function_responses` |

The primitives are all present, so this is a refactor plus a live queue, not new modelling:

- `_init_inference` (`:3597`), `_step_zero` (`:3843`), `_step_inference` (`:3914`),
  `_post_inference` (`:4302`) already carry an explicit `inference_state`.
- `_prepare_function_responses_for_detection` (`:4042`) is already queue-based and the docstring
  confirms **the model chooses the injection position**, not the ground truth. So a live queue
  fed by the τ2 environment on `<TOOLCALL>` detection is a drop-in replacement for the pre-baked
  tensors.

Deliverable: a driver that can be advanced one step at a time, accepts audio incrementally, and
accepts a tool result at an arbitrary step.

Note `online_inference` is also greedy-only — `temperature` / `top_p` / `repetition_penalty`
exist on `offline_inference` only. Carry them over.

### 0c. Resolve the 200 ms ↔ 80 ms mismatch

A 200 ms tick is 2.5 model frames. Options, in order of preference:

1. **Buffer to a 2-tick (400 ms = 5 frame) cadence.** Keeps the 200 ms grid so training data,
   the Gemini baseline, and eval stay comparable. Costs 400 ms output granularity.
2. Change `tick_duration_ms` to 160 ms (2 frames) or 240 ms (3 frames). Clean integration, but
   breaks comparability with the existing data and baseline.

Recommend (1). Also confirm `online_window_size` (default 70 frames = 5.6 s) is adequate for
turns that reference earlier context.

### 0d. Produce the arm-A baseline

Run A on a fixed task subset of all 3 test domains. Record reward rate, all four voice metrics,
and tool-call parse rate. **Acceptance criterion for Phase 0: a baseline table exists.**

This also tells us which failures dominate, which should re-prioritise everything below.

---

## 4. Phase 1 — Data generator fixes

All CPU-only. In `nemo-voice-agent/scripts/episodes_to_nemotron_training.py` unless noted.

**1a. Real tool schemas (highest priority).** Load the full tool list from the tau2 domain
registry instead of reconstructing from observed calls: every tool in the domain, real
descriptions from the docstrings, correct `required` vs optional, enums. Without this, training
teaches "call what's in the prompt" and the cross-domain arm C measures nothing — tool
*selection* from an unfamiliar 21-tool menu is precisely the capability under test.

**1b. Compress tool-response JSON.** Load-bearing, because 1a does not fit in
`max_fc_total_tokens: 8000` without it (adding ~2,650 tokens of retail descriptions plus 11 more
tool entries to a ~5.1 K baseline). Drop unused fields, elide long list bodies.

**1c. Instrument the drop.** Log `fc_drop_info` per domain and fail loudly. The drop scales with
domain size (telecom: 21 tools), so cuts are lost **non-uniformly by domain** and silently. Raise
`max_fc_total_tokens` if 1b is insufficient — but only after measuring.

**1d. Compress inter-turn dead air.** Shift timestamps and cut silence from both waveforms in
inter-turn gaps, targeting 0.3–0.8 s. Exact (both tracks are silent there), needs no TTS,
shortens sequences ~15 %. This is *not* cosmetic: `response_latency_mean` is a scored metric, so
cloning the teacher's 2.13 s mean and ~6.5 s per tool call trains directly against the score and
discards the duplex model's one architectural advantage over the teacher.

**1e. Keep a cheap monotonicity guard.** FC supervisions are appended in tick order
(`all_supervisions = supervisions + fc_supervisions`), so the unenforced monotonicity invariant
in `s2s_dataset.py`'s insertion loop holds by construction today. An assert is cheap insurance
against a future reordering, not an active bug fix. Low priority.

**1f. Do not re-enable NeMo's audio augmentation blindly.** The harness already applies
background noise, bursts, frame drops (`FRAME_DROP_RATE = 0.01`), and muffling. Layering NeMo's
augmentation on top double-applies. Prefer collecting across the 8 `COMPLEXITY_LEVELS`
(`src/tau2/user_simulation_voice_presets.py:443`) — we currently collect only `regular`.

**1g. Do not mix in the wideband synthetic shards as-is.** They would create a bimodal bandwidth
distribution in which the telephony mode — the only one that matters — is the minority. Either
telephony-degrade them (cheap: 300–3400 Hz bandpass + μ-law round-trip) or exclude them. They
contain no tool calls, so they help the audio/text channels only.

**Not doing:** anything about agent-track audio quality or sample rate (`target_audio` is
debug-only).

---

## 5. Phase 2 — Prototype on existing domains

Small, deliberately overfitting, purely to prove the pipeline end-to-end: collect ~20–50 retail
episodes, run Phase 1 conversion, train, and evaluate through the Phase 0 harness. Overfitting is
the expected and acceptable result here.

Acceptance: training loss on the function channel actually moves (today `function_loss` sits at
~0.001 and `val_txt_bleu_tool_call` at 0 on the synthetic shards — the FC channel has never been
exercised), and the Phase 0 harness produces a score for a finetuned checkpoint.

Use ≥ 8 shards so all 8 GPUs get data (`--shard_size` default 1 is correct for small sets; 3
shards caps data-parallel width at 3 of 8).

---

## 6. Phase 3 — Scale up

### The binding constraint is task authoring, not episode collection

11 new domains × ~10 tasks ≈ 110 tasks. At a realistic ~50 % frontier pass rate, 1 trial each →
~55 successful episodes ≈ **3 h of audio**. Target is ~1000 episodes ≈ 55 h.

Repeat trials on the same task do **not** close the gap: they reproduce the same tool sequence,
which is exactly the near-duplicate problem the three retail task_0 cuts already have.

**So: ~50–100 tasks per new domain, not 8–10.** That is the main work item of this phase.

Author for diversity deliberately, since transfer is bounded by whether train-domain difficulty
matches or exceeds test: vary tool count, policy depth, authentication patterns, multi-step
dependencies, and error/failure paths. If the new domains are LLM-generated using
retail/airline/telecom as templates, they will be stylistic clones and arm C will read
optimistically.

### Collection

```bash
bash scripts/run_training_data_collection.sh --num-tasks 20 --domain "banking calendar ..."
```

~3–4 min per episode, `--workers 3`, bottlenecked on Gemini Live API rate limits. ~1000 episodes
≈ 19 h wall clock at 3 workers, plus ~2× that in runs because of the ~50 % pass rate. Budget
Gemini Live audio cost for ~2000 episodes × ~200 s bidirectional audio.

### Deliberately over-sample interruptions

`yield_rate` and `yield_latency_mean` are scored, and our data contains **one** barge-in.
Behavior cloning cannot teach yielding from one example, and successful trajectories are exactly
where interruptions are rarest. Either raise the user simulator's interruption propensity for a
subset of runs, or select for interruption-containing episodes.

### Known selection bias

Filtering on reward = 1.0 keeps the *easy* tasks — the ones the student likely already passes.
Track per-task success so coverage is known. Keep the failures on disk: they are the raw material
for a later rejection-sampling or DPO pass, and re-collecting them is expensive.

---

## 7. Phase 4 — Final runs

Run arms B and C with identical hyperparameters, selecting checkpoints on the 4th held-out new
domain. Report all three arms side by side with the secondary metrics.

Overfitting/forgetting risk: 2.41 B trainable params on ~55 h, with LoRA on only 29 of 56 layers
plus 2.35 B of fully trainable heads/embeddings. Policy compliance depends on general LLM ability
that this can degrade. Mitigations, in order: hold-out-based early stopping, lower LR, then
freezing more (`^embed_tokens\.`, `^lm_head\.`, `^function_head\.`, `^perception\.` — but note
the released 11B was trained with these unfrozen, so freezing changes the recipe).

---

## 8. Risk register

| # | risk | severity | mitigation |
|---|---|---|---|
| 1 | No eval harness → nothing measurable | **blocking** | Phase 0 first |
| 2 | Truncated tool schemas invalidate arm C | high | Phase 1a |
| 3 | Cloning teacher latency hurts scored metrics | high | Phase 1d |
| 4 | Too few tasks → too little data | high | Phase 3, ~50–100 tasks/domain |
| 5 | `max_fc_total_tokens` drops cuts non-uniformly per domain, silently | medium | Phase 1b + 1c |
| 6 | `yield_rate` untrainable from 1 barge-in example | medium | Phase 3 over-sampling |
| 7 | New domains are stylistic clones → optimistic transfer | medium | deliberate diversity in authoring |
| 8 | Overfitting / forgetting policy-following ability | medium | Phase 4 mitigations |
| 9 | Success-filtering keeps only easy tasks | low-medium | track coverage, keep failures |

---

## 9. Open questions

1. Does τ-voice combine reward and the voice metrics into one headline score, or report them
   separately? Determines how much Phase 1d matters relative to Phase 1a.
2. Are the 11 new domains' policies comparable in depth to the Sierra-authored three? Drives
   risk 7.
3. Is telecom's 2285 tasks the real eval set, or is there an official subset? Affects eval cost
   and statistical power.
4. Should arm B hold out tasks or trials? Task-level is cleaner but shrinks airline (50 tasks).

---

## Appendix — pointers

- Model / checkpoint internals: `CODE_WALKTHROUGH.md` §7 (model family), §8 (checkpoint
  composition).
- Launch recipe: `FINETUNING_11B.md`.
- Data format contract and its traps: memory `duplex-stt-training-data-contract`.
- Collection procedure: `tau-voice-2/user_docs/training_data_generation.md`.
- Existing 3-cut sample set: `/fsx/home/kai.li/data/voicechat/tau2_fixed/`.
