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

- **5 tools advertised; retail has 16.** (airline 14, telecom 13 — counted from the registry via
  `export_tool_schemas.py`, not by grepping `def`, which over-counts helpers.)
- **Zero `description` fields**, on tools or parameters. The correct retail tool block is 2,906
  tokens against the 5-tool reconstruction's ~275.
- `required` lists every observed argument, so optional params are marked required. The schema is
  *wrong*, not merely thin.

The fix is not to parse docstrings ourselves: `Tool.openai_schema`
(`tau-voice-2/src/tau2/environment/tool.py:140`) already emits exactly the shape our converter
writes, built from the real signature and docstring. `export_tool_schemas.py` dumps it to a JSON
sidecar; all 15 loadable domains have descriptions on every tool.

**Model**

- `target_audio` is only "saved for debugging" (`duplex_stt_model.py:651`), never a loss target.
  Agent-track audio quality is irrelevant to STT training.
- `max_fc_total_tokens: 8000` (`examples/speechlm2/conf/finetune/s2s_duplex_stt_11b.yaml:122`)
  **drops the whole cut**, it does not truncate. See §2b — this is now the confirmed blocker for
  arm B.
- Trainable: 2.41 B of 10.13 B (see memory `stt-11b-trainable-params`). `perception` (0.61 B) is
  trainable, which is what makes telephony adaptation possible.

## 2b. The FC token budget — measured 2026-08-14, and it blocks arm B

Reproduce with:

```bash
# in tau-voice-2 (needs its own venv: python -m venv .venv && .venv/bin/pip install -e .)
.venv/bin/python scripts/export_tool_schemas.py --output data/tool_schemas.json
# in nemo-voice-agent, voicechat env
python scripts/measure_fc_token_budget.py \
    --shards /fsx/home/kai.li/data/voicechat/tau2_fixed/shards \
    --tool_schemas /fsx/home/kai.li/code/tau-voice-2/data/tool_schemas.json
```

**What counts** (replicated from `_get_fc_cut_total_prompt_tokens`): `1 + tokens(system_prompt) +
1`, plus every segment of `supervisions[1:]`. Critical trap:
`seg_text = (custom.get("function") or sup.text or "")` — with the mandatory
`custom={"function": ""}` on speech turns the empty string is falsy, so it falls through to
`sup.text`. **The whole conversation transcript counts toward the FC budget**, not just tool-call
strings.

**Today the 3 cuts fit; with correct schemas all three are dropped.**

| cut | system | segments | total | vs 8000 |
|---|---|---|---|---|
| `0_4922ce6f` | 1,697 | 4,618 | 6,315 | OK |
| same, full 16-tool retail schema | **4,328** | 4,618 | **8,946** | **DROPPED** |

The other two behave identically (8,649 and 8,766). Per-domain system-prompt cost against a
4,618-token transcript allowance:

| domain | tools | policy | schema | system | headroom |
|---|---|---|---|---|---|
| telecom | 13 | 5,382 | 1,519 | 6,903 | **−3,521** |
| retail | 16 | 1,420 | 2,906 | 4,328 | **−946** |
| airline | 14 | 1,672 | 2,582 | 4,256 | **−874** |
| 11 new domains | 9–10 | 603–1,216 | 1,138–1,730 | 1,858–2,705 | +677 … +1,524 |

So: **all three test domains overflow, every new domain fits.** Arm C's *training* data is
unaffected (new domains only), but **arm B is impossible until this is fixed**, and it would fail
silently. Note the new domains' headroom is only 677–1,524 tokens, so longer conversations there
will overflow too.

**Where the tokens are.** Tool responses are **84 %** of segment cost (3,861 of 4,618); all speech
is 396. So compression targets responses, not transcripts. Largest single response is 1,498 tokens
— `get_product_details` dumping every variant of a product.

**One lossless win, and it is not enough alone.** `content` is *double-encoded* JSON (a JSON string
containing escaped JSON), and `format_tool_response` uses default `json.dumps` separators while the
tool block uses compact ones. Measured over all 3 cuts:

| | tokens | saved |
|---|---|---|
| as generated today | 11,083 | — |
| compact separators | 11,067 | 0.1 % |
| + un-double-encoded `content` | 9,052 | **18.3 %** |

That is **677 tokens/cut, lossless** — but retail's gap is 946, so it closes airline and retail
only in combination with a budget raise. Recommendation: apply the lossless fix **and** raise
`max_fc_total_tokens` to ~12,000 (covers telecom's worst case). The raise is the one item here
that needs a GPU to validate: it lengthens the LLM sequence (a 200 s conversation is ~2,500 audio
frames at 12.5 Hz plus the FC tokens), so memory must be re-checked before trusting it.

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

The template method fits, despite every existing provider being a remote WebSocket API and NeMo
being local and in-process. `_async_run_tick` (`adapter.py:215-292`) already handles tool-result
flushing, audio capping via `buffer_excess_audio`, barge-in, and cumulative state; only
`_execute_tick` and `_flush_pending_tool_results` are ours. Follow the directory convention in
`src/tau2/voice/audio_native/AGENTS.md` (`provider.py` / `events.py` /
`discrete_time_adapter.py`), and reuse `StreamingTelephonyConverter` from `audio_converter.py`
rather than writing a converter — the adapter must emit 8 kHz μ-law and consume 16 kHz float.

Two contract details worth getting right up front:
- Populate `self._utterance_transcripts[item_id]` and tag each audio chunk with that `item_id`, or
  `get_proportional_transcript` (`tick_result.py:463`) silently returns `""` and the transcript is
  empty. Use one `item_id` per agent utterance, minted on the first frame after a yield.
- We know exactly which text belongs to which tick (we decode per frame), so the proportional
  redistribution is unnecessary but harmless — it is driven by byte ratios within an utterance.
- Signal barge-in with `result.truncate_agent_audio(...)` (`tick_result.py:315`); `LiveKit` is the
  precedent for a non-WebSocket provider and the only one that opts out of the template.

### 0b. Build an incremental, FC-capable inference driver (the real work)

Neither existing entry point can serve a live eval:

| path | streaming | function calling |
|---|---|---|
| `offline_inference` (`:4911`) | no — whole `input_signal` up front | **yes**, but `function_responses` must be pre-baked from ground truth |
| `online_inference` (`:5072`) | sliding window, but loops internally over all audio | **none** — no FC parameters, never references `function_responses` |

The primitives are all present, so this is a refactor plus a live queue, not new modelling.
Three findings from reading the code settle the design (2026-08-15):

**1. `online_inference` is already the incremental driver, structurally.** It is "online" only in
the sense of *causal windowing*: `_init_inference` (`:3597`) runs `self.perception` over the whole
signal and preallocates `[B, T, H]`, but the loop body then *overwrites*
`inference_state["input_embeds"][:, t:t+1, :]` with a freshly encoded trailing window and calls
`_step_inference` (`:5155-5170`). The preallocated content for frame `t` is never read once
overwritten. So: preallocate a horizon of silence, extract the loop body into a
`step(t, audio_so_far)` method, and drive it from the adapter instead of a `for` loop. The KV cache
is already incremental. **This is a loop inversion, not a rewrite.**

**2. The FC response queue is a genuine live seam.**
`_prepare_function_responses_for_detection` (`:4042`) builds `response_queue[b] = [(turn_idx,
tokens)]`, consumed when the model emits EOTC, and its docstring states the ground truth supplies
*which* response, **not the position — "the model decides"**. So appending to that queue when a
real τ2 tool result arrives is a drop-in for the pre-baked tensors. No ground truth needed.

**3. The actual blocker is timeline expansion, not the queue.**
`_expand_for_function_calling` (`:4458`) precomputes the whole expansion from ground-truth
`function_call_lengths` / `function_response_steps` / `function_response_lengths` before stepping,
then sets `fc_expand_applied` (`:4864`). Live inference **cannot** supply that: the expansion
depends on when the model chooses to call and how long the real tool result turns out to be.

*Resolution — do not expand. Decouple two counters:*
- `t_lm` — LM position; advances on every `_step_inference`, i.e. on audio frames **and** on
  injected FC tokens.
- `t_audio` — audio frames consumed; advances only as ticks deliver audio.

FC injection then means stepping the LLM over the response tokens **without consuming audio**.
Preallocate `max_audio_frames + max_fc_total_tokens` and skip `_expand_for_function_calling`
entirely. This also sidesteps `_expand_waveform_fc_timeline` (`:4091`), which exists only to
re-align the logged waveform after expansion.

Sim time does not advance during injection, and that is **correct here**: the harness advances sim
time by `audio_sent_duration_ms` (`adapter.py:262`), and τ2 tools are local in-memory Python with
~0 latency. It does diverge from the teacher's ~6.5 s per tool call — deliberately, since that
latency is the duplex model's architectural advantage and `response_latency_mean` is scored.
Wall-clock cost lands inside the tick, which is harmless (see 0c).

Deliverable: a driver advanced one step at a time, accepting audio incrementally and a tool result
at an arbitrary step. Also carry over `temperature` / `top_p` / `repetition_penalty` /
`presence_penalty` — `online_inference` is greedy-only; those exist on `offline_inference` only.

### 0c. Resolve the 200 ms ↔ 80 ms mismatch

A 200 ms tick is 2.5 model frames — non-integer. **Resolved: no cadence change is needed.** Keep a
sample-level residual accumulator and consume `floor(available / frame_samples)` frames per tick.
At 16 kHz an 80 ms frame is 1280 samples and a tick delivers 3200, so the pattern is exactly
2, 3, 2, 3, … — mean 2.5, residual bounded at 640 samples (40 ms), never drifting. This is strictly
better than the previously preferred 400 ms buffering: it keeps the 200 ms grid *and* full
per-tick granularity.

The base class makes this safe. Sim time is `audio_sent_duration_ms`, not wall clock
(`adapter.py:262`), and step 11 only ever *sleeps* to pad a fast tick (`:286-289`) — a GPU forward
pass slower than 200 ms simply runs slower than realtime without corrupting the timeline. Local
in-process inference is therefore compatible with this harness by construction.

Still to confirm: whether `online_window_size` (70 frames = 5.6 s) is adequate for turns
referencing earlier context — it is the one parameter that could force a real design change.

### 0d. Produce the arm-A baseline

Run A on a fixed task subset of all 3 test domains. Record reward rate, all four voice metrics,
and tool-call parse rate. **Acceptance criterion for Phase 0: a baseline table exists.**

This also tells us which failures dominate, which should re-prioritise everything below.

---

## 4. Phase 1 — Data generator fixes

All CPU-only. In `nemo-voice-agent/scripts/episodes_to_nemotron_training.py` unless noted.

**1a. Real tool schemas (highest priority). DONE.** `tau-voice-2/scripts/export_tool_schemas.py`
writes `data/tool_schemas.json` for all 15 loadable domains from `Tool.openai_schema`.
`episodes_to_nemotron_training.py` now takes `--tool_schemas <json>` and selects by the episode's
domain, read from `info.environment_info.domain_name` in `results.json` — which is populated
unconditionally, unlike its sibling `tool_defs` (only set when `get_info(include_tool_info=True)`,
which the orchestrator never passes; hence the sidecar). Falling back to the lossy
`extract_tool_schemas` now prints a `[WARN]`, and `metadata.json` records `schema_sources` so a
corpus built from reconstructed schemas is identifiable after the fact. Without this, training
teaches "call what's in the prompt" and arm C measures nothing — tool *selection* from an
unfamiliar 13–16-tool menu is precisely the capability under test.

**1b. Compress tool responses. DONE (lossless part).** `format_tool_response` now un-nests
dict/list payloads and emits compact separators; `format_toolcall` is deliberately unchanged,
since tool calls are what the model *generates* and their surface form is a learned target, while
responses are only what it *reads*. Scalars and non-JSON results keep the string form, so a bare
`"123"` never silently becomes an int. Measured saving: **662–707 tokens/cut**, matching the
677 estimate. Responses are 84 % of segment cost, so this is the only place worth compressing. If
more is needed, elide long variant/list bodies — but the agent must still pick the right variant,
so that is genuinely lossy and must be measured against task success, not token count.

**1c. Raise `max_fc_total_tokens` to 12,000 and turn on `fc_log`. DONE in config, GPU-unvalidated.**
Measured on the repaired shards with augmentation on: **8,389 / 8,137 / 8,254** — all three over
8,000, so the old value would now discard 100 % of the corrected data. 1b's saving does not close
the gap because 1a costs ~+2,000. 12,000 leaves 3.6–3.9k headroom on retail and is the smallest
round value that also clears telecom (system prompt 6,903 alone). `fc_log: true` is now set —
it defaults to `False`, which is exactly how 8,000 silently discarding everything would have gone
unnoticed. **Still needs a GPU memory re-check:** a canonical retail cut is now ~10.9k sequence
positions (4,478 prompt + 2,495 audio frames + 3,909 inserted FC), up from ~8.4k.

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
| 5 | `max_fc_total_tokens` drops cuts non-uniformly per domain, silently — **confirmed: correct schemas drop 100 % of retail/airline/telecom cuts, blocking arm B** | **high** | Phase 1b + 1c, §2b |
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

## Appendix — state of the world on this box (2026-08-14)

- **The raw episode artifacts are gone, and it cost us almost nothing.** No
  `tau-voice-2/data/simulations`, and a filesystem-wide search for `both.wav` found nothing. Only
  the built Shar survives (`/fsx/home/kai.li/data/voicechat/tau2_fixed/`). The 3 cuts therefore
  cannot be regenerated — but everything the fixes need is already in the Shar, so they were
  **repaired in place** instead (`scripts/repair_tau2_fc_prompt.py` →
  `/fsx/home/kai.li/data/voicechat/tau2_canonical/`, 4–5 tools → 16 documented, responses
  single-encoded, both audio tracks and the timeline verified bit-identical, audio hardlinked so
  the copy costs 30 KB). The only genuinely unrecoverable field is the tool-call *end* times from
  `assistant_tool_calls_labels.txt`, and those gaps measured 0.0 s, so there is nothing in them.
  Dead-air compression (1d) is also still doable from the Shar.
  **Keep raw artifacts from here on anyway** — the next loss may not be this cheap.
- `tau-voice-2/.venv` now exists (`pip install -e .`, tau2 1.0.1, Python 3.12.13) and
  `.venv/bin/tau2` works. `.venv` is gitignored.
- `banking_knowledge` fails to export (`ModuleNotFoundError: rank_bm25`); the other 15 domains
  export cleanly. Install `rank_bm25` if that domain is ever needed.
- **Disk is tightening, not easing.** `/fsx/home` is at **98 % — 2.2 T free of 74 T**, shared
  across all users and still shrinking (3.5 T on 08-13, 3.2 T on 08-14, 2.2 T on 08-15). The
  "23 G free / 76 %" figure noted earlier was `/` (the 97 G local root), a different mount —
  do not use it to size a run. A `save_top_k: 3` run needs ~120 GB, which fits today, but an
  `Errno 28` from others filling the last TB would land hours in at a checkpoint save. Check
  `df -h /fsx/home` before launching and consider `save_top_k: 1`.
- New tooling: `tau-voice-2/scripts/export_tool_schemas.py`,
  `nemo-voice-agent/scripts/measure_fc_token_budget.py`,
  `nemo-voice-agent/scripts/repair_tau2_fc_prompt.py`.
- **Train on `tau2_canonical/shards`, not `tau2_fixed/shards`**, and with
  `max_fc_total_tokens: 12000`. The old pairing (`tau2_fixed` + 8,000) trains on truncated 4–5-tool
  prompts; the new pairing at 8,000 trains on nothing at all.

## Appendix — pointers

- Model / checkpoint internals: `CODE_WALKTHROUGH.md` §7 (model family), §8 (checkpoint
  composition).
- Launch recipe: `FINETUNING_11B.md`.
- Data format contract and its traps: memory `duplex-stt-training-data-contract`.
- Collection procedure: `tau-voice-2/user_docs/training_data_generation.md`.
- Existing 3-cut sample set: `/fsx/home/kai.li/data/voicechat/tau2_fixed/`.
