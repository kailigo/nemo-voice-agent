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
`step(t, audio_so_far)` method, and drive it from the adapter instead of a `for` loop.
**This is a loop inversion, not a rewrite.**

*Correction (2026-08-17, on GPU):* the last sentence of that paragraph originally read "the KV
cache is already incremental." **It is not, and it cannot be made so from here** — see 0b-bis.

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

### 0b-bis. There is no KV cache, and the prompt must not be stepped

Two findings from building and running the driver (`streaming_fc_session.py`, validated on an H200
via `scripts/run_on_gpu.sh` and `scripts/check_streaming_driver.py --checks cache,stream`).

**No incremental decode exists for this checkpoint.** `_init_inference` (`:3719`) sets
`use_cache=False` for Nemotron and that is *correct*, not a stub. Three independent blockers, in
increasing order of severity:

1. `DynamicCache` is the wrong type. Nemotron-Nano-9B-v2 is a **hybrid Mamba2/attention** model
   (`modeling_nemotron_h.py`), so half the layers need conv/SSM state, not K/V. Forcing it on dies
   with `AttributeError: 'DynamicCache' object has no attribute 'conv_kernel_size'`.
2. The model's own `HybridMambaAttentionDynamicCache` (`:155`) does not work either: it never
   assigns `self.conv_kernel_size` (only the mixer does, `:294`) though `:461`/`:546` read it off
   the cache, and `update_conv_state(cache_init=True)` dereferences `.device` on a *list*.
3. **The decisive one:** incremental decode is unreachable through this call path. The Mamba
   mixer's single-step branch is gated on `cache_position[0] > 0` (`:375`), `NemotronHModel.forward`
   defaults `cache_position` to `arange(seq_len)` (`:1359`), and `DuplexSTTModel.forward` (`:466`)
   **has no `cache_position` parameter at all**. A one-position step therefore always arrives as
   `cache_position=[0]` and every layer takes the *prefill* branch. This would not raise; it would
   be **silently wrong** — the worst available failure mode.

Consequence: every step re-runs the full prefix, so inference is **O(T²)**. Check 1 in
`check_streaming_driver.py` exists to keep this reproducible — it PASSES when forcing the cache
*fails*, so the conclusion cannot quietly rot into folklore. The fix, if throughput demands it, is
to thread `cache_position` as an additive optional kwarg plus a repaired cache subclass; that is a
real change to the model's forward signature and is deliberately out of Phase 0 scope.

**Measured cost, in the real harness (`tau2 run`, retail, one H200, 2026-08-18): 199 ticks in
939.8 s for 40.0 s of audio = 4.72 s/tick against a 200 ms budget, i.e. 23.5× slower than
realtime.** Per 0c this is *not* a correctness problem — the harness derives sim time from
`audio_sent_duration_ms` and only ever sleeps to pad a *fast* tick, so a slow forward just runs
slow. It is purely an eval-throughput cost, and it is a large one: **~1.4 GPU-hours per 200 s
conversation, so a 300-conversation baseline is ~430 GPU-hours, ~54 h across 8 GPUs.** That is
affordable once for arm A only if it is planned for; it is far too slow to iterate against.

Three things the per-tick series says, and they settle where the cost goes:

*The cost is 100 % LM frames.* Tick times alternate exactly 3.46 s / 5.20 s, tracking the
2,3,2,3 residual-accumulator pattern. Fitting `t = a + b·frames` gives **b = 1.74 s/frame, a =
−0.02 s**: the fixed per-tick overhead — TTS, resampling, the tick machinery, the user simulator —
is *zero* to measurement precision. Every optimisation has to come out of the forward pass.

*Per-frame cost is linear in prefix length, and the prompt is the prefix.* Same driver, same GPU,
same window, with a 37-token prompt instead of the τ2 tool schema: **77 ms/frame** (120 ticks in
23.1 s = 0.96× realtime). Prefix ratio 25×, time ratio 22.5× — linear, as a full prefix re-run
must be. This is why the earlier standalone figure recorded here (879 ms/tick, "4.40× realtime")
was misleading and has been replaced: it was not measured with the τ2 tool-schema prompt loaded, so
it described a prefix an order of magnitude shorter than any real episode.

*Growth within an episode is the small term.* Ticks 2–50 mean 4.49 s, ticks 151–200 mean 4.85 s —
only **+8 %** across 500 frames. The O(T²) audio term barely registers because 4,575 prompt tokens
dwarf 2,500 frames even at the end of a 200 s conversation. Which is the point: the blowup is the
model re-reading the tool schemas on every single frame, and those bytes are identical on every
step and across every conversation in a domain. **A prompt-prefix cache alone is worth ~20×** —
enough on its own to bring this to roughly realtime — and it is by a wide margin the highest-value
optimisation available, well before full incremental decode. The Mamba layers are what make it
non-trivial: reusing prompt state requires passing an initial recurrent state into the suffix
kernel, which is the same missing plumbing as above.

**Do not step the LLM over the prompt.** The naive driver replayed `_step_inference` once per
prompt position — ~4.3k discarded full-prefix forwards before the first audio frame. In the
no-cache regime every step is stateless, so the whole loop is replaceable by one vectorized write
(`_prefill_prompt_embeds`): with no cache the only surviving effect of a prompt-position step is
the fusion write at `:3969`; all `gen_*` writes sit behind `if not is_prompt_position.all()`
(`:4014`), fusion is elementwise over positions, and RNNT is gated on `_run_rnnt_in_loop`, which is
never set. Measured: `start()` went from minutes to **0.0 s**. This equivalence is valid *only*
because there is no cache — restoring one invalidates it, and the docstring says so.

### 0b-ter. Validation status of the driver, and the bug it caught

Driver: `nemo/collections/speechlm2/models/streaming_fc_session.py`. Checks:
`scripts/check_streaming_driver.py` (GPU, via `scripts/run_on_gpu.sh <jobid> <gpu> …`) and
`scripts/test_parse_call.py` (CPU, no model — runs in a second).

| Seam | Status |
|---|---|
| Cache cannot be forced on | **PASS**, reproduced in 1 s (check 1) |
| Two-clock arithmetic `t_lm = prompt + audio + fc` | **exact** at 150 / 750 / 2,495 frames |
| Residual accumulator (200 ms → 2,3,2,3 frames) | **exact**, `residual_samples: 0` every run |
| Full 199.6 s conversation | **PASS** — 2,495 frames, no drift, no horizon exhaustion |
| Throughput | 4.72 s/tick = **23.5× realtime** in-harness (see 0b-bis) |
| FC detect → parse → inject | **PASS** via forced call (check 4) |
| Model *spontaneously* emitting `<SOTC>` | **untestable pre-SFT** (check 3, INCONCLUSIVE by design) |
| Injected tool result is *read* by the model | **PASS**, unexpectedly (see below) |
| Live τ2 orchestrator (`tau2 run`, retail) | **PASS**, `EXIT=0` — see 0c-bis |
| Causal-window vs offline agreement | **not answerable pre-SFT** (see below) |

**Check 4 exists because check 3 cannot pass yet.** Check 3 waits for the checkpoint to emit
`<SOTC>`; `stt_extracted_lora` is the *input* to our SFT and has never seen τ2 data, so it never
will — which would have left the entire FC round-trip unvalidated until after training. Check 4
instead forces a known `<SOTC> <TOOLCALL>[…]</TOOLCALL> <EOTC>` sequence onto the function channel
(one token per audio frame, matching the training layout) and drives the real detection state
machine, parser, mid-tick stall, wire formatting, and injection.

It immediately earned itself. `_TOOLCALL_BLOCK` was `<TOOLCALL>\s*(\[.*\])\s*$` — **anchored to
end-of-string, while the trained format ends with `</TOOLCALL>`**. The regex never matched, the
blob kept its tags, `json.loads` failed, and *every* tool call decoded to `name=""` with empty
arguments. Nothing raises on that path: the provider would have forwarded empty-named calls into
τ2, so arm C would have scored ~0 tool success and read as "the model picks tools badly" rather
than as a one-line parser bug. Fixed, plus a tolerant fallback (missing either tag, bare object,
nested arrays in arguments), 12 CPU cases in `scripts/test_parse_call.py`, and a distinct
`UNPARSEABLE` warning in the provider so this class of failure can never again masquerade as poor
model behaviour.

Lesson worth keeping: **a check whose pass condition depends on model quality validates nothing
until the model is good.** Force the input instead.

**The injected result is genuinely read, not merely stepped over.** Check 4 forces exactly one
call, then leaves the function channel entirely model-driven. At tick 31 the forced
`find_user_id_by_name_zip` was detected and `{"user_id": "yusuf_rossi_9620"}` injected; at tick 85
the *model* emitted its own well-formed `<SOTC> <TOOLCALL>[…]</TOOLCALL> <EOTC>` block —
`get_order_details({"order_id": "yusuf_rossi_9620"})`, carrying the exact value from the injected
response. Two things follow. (1) `_drain_injection` really does place response tokens where the LM
attends to them; the FC round-trip is validated semantically, not just mechanically. (2) The
earlier claim that a pre-SFT checkpoint "never emits `<SOTC>`" needs qualifying: it never does so
*unprompted*, but one in-context exemplar is enough for it to imitate the wire format correctly.
Semantically the follow-up is wrong (a `user_id` passed as `order_id`) — which is precisely the
tool-use competence SFT is meant to supply, and a useful sign that the format is the easy part.
This also means check 4 must assert on the *first* call only; requiring exactly one call made it
fail for succeeding too well.

**The `online_window_size` question cannot be answered on this checkpoint.** `--compare-offline`
reports 875 ms/tick and a similarity of 0.0273 raw, but that number is an artifact, not a
measurement (see below).

**Throughput is a cost, not a blocker — but it is a bigger cost than first recorded.** At the
measured 23.5× realtime (0b-bis), ~430 GPU-hours across the 8 GPUs of one node is **~54 h wall**.
`ml.p5en.48xlarge` grants **4 days** per job (the partition allows 7; only the unused `interactive`
partition is capped at 4 h), so a full 300-conversation baseline still fits inside a single ordinary
allocation — but it now consumes over half of one, rather than an overnight slice. Two consequences
worth acting on: run arm A **8-way in parallel from the start** (one episode per GPU, each loads its
own 9B — `--max-concurrency 1` per process, 8 processes), and treat the prompt-prefix cache as
promoted from "nice to have" to the thing that decides whether arms B and C are affordable at the
same scale. The *debug* loop is ~80 min per 200 s conversation, so debug against short
`--max-steps-seconds`, never a full episode. The scarce resource remains obtaining an allocation at
all (25 of 27 p5en nodes were `alloc` on 2026-08-18), so when one is held, use it.

Restating the artifact point precisely: the similarity number is not a
measurement: `offline_inference` emits audio-timestamp tokens (`<$0.72$>`, `<|1.52|>`) that the
driver strips, and `difflib`'s ratio is then dominated by the length mismatch (the check now also
reports a normalized figure). More fundamentally, both outputs are off-task refusal boilerplate, so
comparing them measures nothing about whether a 5.6 s causal window suffices for *turns referencing
earlier context* — the actual question in 0c. Answering it needs a τ2-tuned checkpoint; deferred
rather than guessed at, and 0c's "still to confirm" stands.

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

### 0c-bis. The integration is validated end to end (2026-08-18)

`DiscreteTimeNeMoAdapter` had never run inside a live τ2 orchestrator; standalone checks cannot
cover that seam. `scripts/tau2_smoke_nemo.sh` now does, and a retail episode completes with
`EXIT=0`: 101 ticks, `NeMo duplex session ready: prompt=4575 tokens, 16 tools, modality=audio`,
non-empty agent transcript, `termination_reason=max_steps`, reward persisted, and the post-episode
hallucination reviewer run against Bedrock. **Reward 0.0 is the correct result here** —
`stt_extracted_lora` emits off-task refusal boilerplate and makes no tool calls — so it measures the
harness, not the model.

Four integration bugs it caught, none of which raise in the standalone checks:

1. **`import pyaudio` at module scope** in `voice/utils/audio_io.py` made the whole voice package
   unimportable on a headless node, and `registry.py` catches that ImportError and logs it at
   DEBUG — so `voice_streaming_user_simulator` was *silently absent from the registry*. Made lazy
   inside `play_audio`, its only caller.
2. **`UtteranceTranscript` has no `text` field** (it is `transcript_received`, appended via
   `add_transcript`). Crashed at tick 9 into a 4-attempt retry loop, each retry reloading the 9B.
3. **`add_audio` was never called**, so `get_transcript_for_audio` would have returned `""`
   forever *without raising* — a blank agent transcript for an entire baseline. Now `_record_audio`
   is called at both `agent_audio_chunks.append` sites, with the *telephony* byte count, because
   that is what `get_proportional_transcript` measures.
4. **Three eval-path models are hardcoded `gpt-4.1` with no working CLI flag.** The post-episode
   hallucination reviewer is the dangerous one: it runs by default, `--review-model` does not reach
   it (`hallucination_reviewer.py:197` uses the config constant directly), and it is called outside
   any try — so on a box with no OpenAI key it kills the process *after* the GPU time is spent. All
   three now read env vars; `tau2_smoke_nemo.sh` points them at Bedrock. Two related traps: the
   interruption/backchannel decision model fails *silently* (both call sites swallow it and return
   `decision=False`, giving a materially different simulated user), and for voice runs
   `results.json["simulations"]` is **always `[]` by design** — the records are the per-simulation
   files plus `simulation_index`, so an empty list is not evidence a run was lost.

Two external blockers remain, neither of them code. The **ElevenLabs key is free-tier** (10,000
chars/month, 6,228 left as of 2026-08-18, no extension possible, resets 2026-09-11) — enough for
smoke tests, nowhere near a 300-conversation baseline. And
**all seven official τ-bench voice ids 404** on our key (they are private voices in Sierra's
account); `scripts/tau2_stock_voices.env` substitutes stock library voices, which costs external
leaderboard comparability and makes `regular`-complexity results meaningless, so pass
`--speech-complexity control`.

### 0d. Produce the arm-A baseline

Run A on a fixed task subset of all 3 test domains. Record reward rate, all four voice metrics,
and tool-call parse rate. **Acceptance criterion for Phase 0: a baseline table exists.**

This also tells us which failures dominate, which should re-prioritise everything below.

**Run it in three stages, not one pass.** The task lists are 278 tasks (retail 114, airline 50,
telecom 114). The figures below are measured from a real full-length episode (stage 1, 2026-08-18),
not extrapolated from short caps.

| stage | scope | GPU cost | wall (8 GPU) | TTS chars | what it buys |
|---|---|---|---|---|---|
| 1 | 1 task, 200 s cap | **0.8 GPU-h (done)** | **49 min (done)** | ~400 | FC-budget headroom, memory ceiling, throughput at real prefix length |
| 2 | 24 tasks (8/domain, spread) | 30–122 GPU-h | 4–15 h | ~45k | the failure taxonomy, voice metrics, tool-call parse rate |
| 3 | remaining 254 | 320–1,300 GPU-h | 40–162 h | ~480k | the statistical baseline for the C-vs-A claim |

The ranges are not uncertainty about throughput — they are `--hallucination-retries` (see below).
Scripts: `scripts/tau2_stage1_full_episode.sh <jobid> <gpu>` and
`scripts/tau2_select_subset.py` (emits the stage-2 `--task-ids`).

**`--hallucination-retries` is the single most expensive decision in Phase 0, and it defaults to 3.**
`batch.py::run_unit` re-runs the *entire episode* whenever the post-hoc reviewer finds fabricated
user content, up to 3 times, keeping the last attempt (`hallucination_retries_used`). So a task
costs up to **4 episodes**, and that multiplier — not the tick budget — is what decides whether
stage 3 fits in an allocation: 40 h at 1 attempt, 162 h (6.7 days) at 4. Stage 1's first attempt
was flagged and re-run, so for arm A assume the multiplier bites: a pre-SFT agent that says almost
nothing (211 characters of text across 340 ticks) leaves the user simulator nothing to react to, so
it invents context — which is exactly what the reviewer is built to catch. Each retry does inject
reviewer feedback and a new seed
(`seed + n·1000`), so retries are not pure repetition — but budget for 4 and set the flag
deliberately.

**`--max-steps-seconds` is *not* the binding cost, which corrects an earlier reading of this plan.**
The pre-SFT checkpoint does not run to the cap: it terminates on `TOO_MANY_ERRORS` first.
`orchestrator.py:327` counts *environment-rejected tool calls*, `max_errors` is 10, and the
measured episode hit 10 and stopped at **tick 691 of 1000** (138 s of a 200 s cap) after 4,557 s of
wall clock. So a pre-SFT episode is ~1.3 GPU-h and the cap is an upper bound rather than the price.
Expect this to invert once a fine-tuned checkpoint stops making bad calls — arm C episodes will run
longer per attempt and should be re-timed, not assumed.

**Throughput over a full episode is 6.6 s/tick, not 4.72.** 4,557 s / 691 ticks. The 4.72 in 0b-bis
was measured over the first 199 ticks; the difference is the linear-in-prefix growth that section
documents, so both are right and the long-run figure is the one to budget with (33× realtime).

**Memory: ~79 GB mid-episode, not the 119 GB in §1c.** That 119,235 MiB is a *training* figure and
overstates inference by ~40 GB. Measured on a 143,771 MiB H200: 79,481 MiB at tick 566 with the GPU
pinned at 100 % util (confirming the compute-bound prefill), rising to a stable ~115 GB after a
second in-process model load on the hallucination retry — so loads do not fully release, and 3+
loads in one process is where OOM risk actually lives. One episode per GPU remains the ceiling
(2 × 79 GB > 143 GB), but there is more headroom than §1c implied.

Two further notes on running it:

- **Hold the model across tasks.** Startup is ~4.5 min; at 35 tasks per worker, reloading per
  episode throws away ~2.6 h of wall time. Note the retry path already reloads in-process, which
  is where the memory growth above comes from.
- **Do not build the subset with `--num-tasks`.** It is `tasks[:N]` and the lists are grouped by
  scenario, so a prefix is badly biased — `--num-tasks 30` covers 20 of retail's 87 scenarios and
  **1 of telecom's 3**. Telecom is the trap: only 3 distinct `reason_for_call` values across 114
  tasks, because what varies there is the fault state and repair sequence, not what the user says.
  `tau2_select_subset.py` keys on `evaluation_criteria` for telecom and on the scenario elsewhere.

**Stage 1 is DONE (2026-08-18) — see 0d-bis for what it measured and the one bug it found.**

**The ElevenLabs quota is not the stage-1 blocker it looked like.** The completed 340-tick episode
spent **172 characters** — three user utterances — because a user simulator with nothing to react to
says almost nothing. Quota after stage 1: **3,772 of 10,000 used, resets 2026-09-11 07:03 UTC,
`can_extend_character_limit: False`.** The quota is a *user-simulator* cost and is unrelated to the
agent's voice (see 0d-bis). It binds at stage 3 rather than stage 2: a conversation that actually
progresses is 10–20 user turns ≈ 1–2k chars, so 254 episodes is ~50k–500k chars, 5–50× the monthly
allowance. In dollars that is ~$5–50 at ElevenLabs' $50–100/M, so **the constraint is the free tier,
not the money** — a month of a paid plan covers all of Phase 0. Also decide the trial count up
front: 1 trial gives a noisy `pass^1`, and 4 trials multiplies every number in the table by four.

### 0d-bis. Stage 1 result (2026-08-18): PASS, and the failure mode is now known

`scripts/tau2_stage1_full_episode.sh 1278 0 --hallucination-retries 0` → *Successfully completed
all simulations*, `EXIT=0`, result persisted to `data/simulations/stage1_full_episode/`.
340 ticks, 2,939.8 s, `termination_reason=too_many_errors`, reward 0.0, `pass^1 = 0.000`.
Reward 0.0 is the expected and correct answer for `stt_extracted_lora`; the point of stage 1 was
everything else.

**The arm-A failure mode, exactly.** The agent made **10 tool calls and all 10 failed**:

| calls | tool | error |
|---|---|---|
| 7 | `find_user_id_by_name` | `Tool not found` — **the tool does not exist** |
| 1 | `find_user_id_by_zip` | `Tool not found` — **the tool does not exist** |
| 2 | `find_user_id_by_name_zip` | `User not found` — real tool, wrong arguments |

So the checkpoint **invents tool names**, and its inventions are plausible truncations of the real
one (`find_user_id_by_name_zip` → `find_user_id_by_name`). 8 of 10 calls were unroutable. This is a
better-specified target for SFT than "scores 0": the *wire format* is already right — all 10 parsed
cleanly, confirming the `_TOOLCALL_BLOCK` fix in 0b-ter — and what is missing is the tool inventory
and argument binding. It is also *why* episodes end early: `max_errors` is 10 (§0d).

**Cost is 2/3 GPU and 1/3 API, which changes how to parallelise.** Decomposing the 2,642 s of tick
time by whether the tick invoked the user simulator:

- 198 ticks with no user generation: **4.86 s** mean — this is the GPU cost, and it grows only
  4.27 → 5.90 s (+38 %) from the first quarter to the last, consistent with the linear-in-prefix
  model in 0b-bis.
- 142 ticks (42 %) that did generate: **11.83 s** mean. Only 1.81 s of that is the user-simulator
  LLM (257 s total, 10 % of wall clock); the rest is the turn-taking decision/interrupt models and
  TTS.

*Do not read the raw 7.77 s/tick mean as prefix growth.* Per-tick wall clock rises 4.41 → 12.91 s
across the episode, but that is composition, not slowdown: generate-ticks cluster late, because as
the agent becomes unresponsive the simulator is asked to speak more often. The GPU curve underneath
is the gentle one. **~36 % of arm-A wall clock is API latency that holds no GPU**, so 8 concurrent
episodes per node is if anything conservative.

**Memory, FC budget, and horizon: all clear.** 74–79 GB of 143,771 MiB at steady state (the §1c
figure of 119,235 MiB is a *training* number and overstates inference by ~40 GB), GPU pinned at
100 % util, `fc budget=12000` never approached, and `T=20325 / audio horizon=3750 frames` never
exhausted. The one memory hazard is the retry path — see 0d.

**Episode length is highly variable, so budget on the mean and not on one sample.** Two full
episodes of the same task ended at 340 and 691 ticks (0.8 and 1.3 GPU-h). Length is set by how fast
the agent burns 10 tool errors, which is sampling-dependent. At 0.8–1.3 GPU-h, 278 tasks is
**222–361 GPU-h, or 28–45 h on 8 GPUs** with retries off — better than the 51 h projected from the
cap, because these episodes self-terminate.

**One harness bug found and fixed** (tau-voice-2 `82187c5`): an empty user-simulator completion was
fatal. `_generate_full_duplex_voice_message` passed it to `synthesize_voice`, which raises
`ValueError("Message must have text content.")`, and nothing in the simulation loop catches it — so
the episode died and `run_with_retry` then spent all 3 retries reproducing it, reloading an 11B
model each time. It first fired at tick 144 of a 1,000-tick run. Silence is a legal duplex state and
the context that provokes it is literally a `[Both parties silent for N seconds]` annotation, so it
is now treated as "stay quiet this tick", the same as the `wait` action. **The completed stage-1
episode hit that path 139 times in 340 ticks** — this was not a rare edge case; it was
unconditionally blocking, and it is blocking for any near-mute agent, which every arm is until the
model learns to talk on task.

**`SilenceTTS` does not invalidate reward — nothing listens to the agent's audio.** This corrects an
earlier claim in this plan that stage 2 needed a real voice before its numbers meant anything. The
agent→user channel is *text*, not audio: every persisted `agent_chunk` has `is_audio: false` and a
text `content`, and `discrete_time_adapter._execute_tick` uses the synthesised PCM for exactly two
purposes — `_record_audio` → `get_proportional_transcript`, which paces the text across ticks in
proportion to audio played, and the `both.wav` artifact. There is no ASR anywhere on the agent side;
the agent transcript comes straight off the LM's text channel. So the agent's voice *quality* enters
no metric and only its *duration* matters, which `SilenceTTS` already models at 150 wpm.

What `SilenceTTS` therefore actually costs is narrow: the agent channel of `both.wav` is silent (so
nothing is human-listenable), and pacing is flat-rate rather than real prosody, which perturbs
turn-taking and interruption *timing* only. If a real agent voice is wanted for either reason,
**`NeMoTTS` is the right choice — local, free, no quota** — and an ElevenLabs agent voice would buy
nothing measurable. (It would be ~15 lines if ever needed: `TextToSpeech` wants PCM16 at
`NEMO_SAMPLE_RATE` = 16 kHz, and `voice/synthesis/synthesize.py::synthesize_voice` already returns
exactly that, retries included. Note `--nemo-tts` is named in `nemo/config.py`'s docstring but **no
such CLI flag exists**; set `cfg.tts` or add a preset.)

**What stage 1 deliberately does not answer.** Reward against a *competent* agent. The conversation
collapsed into mutual silence (hence the 139 silence ticks) because the pre-SFT checkpoint emitted
211 characters of text in 340 ticks and then errored out — an agent-quality result, not a TTS
artifact. Stage 2 can proceed on `SilenceTTS`; its reward number is as meaningful as stage 3's.

---

### 0d-ter. PARKED 2026-08-19 on the ElevenLabs quota — and arm A's failure is three failures, not one

Status: **stage 2 is stopped, not failed.** Resume when an ElevenLabs key with real quota exists.
Everything else on the arm-A path works; the section below is what a resumed run should already know.

**The blocker is characters, and it is measured, not inferred.** `GET /v1/user/subscription` returns
`character_count: 9696` of `character_limit: 10000`, `tier: free`, reset **2026-09-11**. So 304
characters remained, which is ~4 user turns. The 7 live episodes sat at ticks 213–398 of a 1,000-tick
cap and needed ~6–8 more turns *each* (~45 turns, ~3,300 chars), so they could not finish and the 16
unstarted ones would each have burned a 4.5-min model load to die at their first utterance. Stage 2's
steps were cancelled (`scancel 1303.<n>`); the allocation was left held.

**Do not confuse the two ElevenLabs limits.** They have different fixes and we hit only one:

| limit | free tier | hit? | signature |
|---|---|---|---|
| concurrent requests | 2 | **yes, twice — survived** | HTTP 429, `rate_limit_error` / `concurrent_limit_exceeded` |
| characters / month | 10,000 | not yet; 304 left | HTTP 429, `character_limit` |

The concurrency 429 is *transient and self-healing*: `runner/progress.py::run_with_retry` re-ran the
whole unit (`attempt 1/4`) and both affected episodes continued for another 28 min. **A mid-episode
fault is therefore not evidence of a dead episode** — a watchdog that greps episode logs for
`emergency cleanup` will requeue live work onto fresh GPUs. Only the fan-out's post-exit verdict in
`progress.log` is authoritative. Mitigations already committed: env-overridable retry constants
(`config.py`), 8 attempts / 30 s ceiling in `tau2_smoke_nemo.sh`, and `STAGGER_SECONDS` in
`tau2_stage2_subset.sh`. Staggering alone will not save an 8-way fan-out against a 2-request cap.

**Character economics, measured from `user_labels.txt` rather than extrapolated.** User turns scale
linearly with ticks — 1 turn per ~103 ticks, holding at 100/101/99/113 across four independent
episodes — and the labelled utterances are 43/142/90/57 chars, so **~74 chars/turn, ~740 per full
200 s episode**. Therefore 24 episodes ≈ **18k chars** and stage 3's 254 ≈ **188k**, *per arm*. The
free tier cannot fund one 24-episode diagnostic pass. Note the cost is coupled to whether the fix
below works: episodes that die at 10 errors cost ~190 chars, ones that run the full cap cost ~740.

**Arm A scores 0.000 for three separate reasons. 0d-bis named only the smallest of them.** The
per-episode `artifacts/*/audio/*_labels.txt` files are the evidence — they carry utterance text with
timestamps, and tool calls with timestamps, for every persisted episode:

| # | failure | evidence | fixable by prompt? |
|---|---|---|---|
| a | ASR errors on proper nouns and spelled-out digits | user says *"Mei Ahmed … M, E, I … A, H, M, E, D … seven eight seven zero five"*; model calls `find_user_id_by_name_zip {"first_name":"May","last_name":"Amelia","zip":"78870"}` | **no — this is what SFT is for** |
| b | fabricates a missing required argument instead of asking or switching tools | user: *"I don't remember it. Can't you look it up another way?"* → model calls `get_order_details {"order_id":"W0000000"}` | partly |
| c | **no error recovery: verbatim repetition until `max_errors`** | 10 identical calls, 3.2 s apart, *after the user's audio has ended* | **yes, directly** |

(c) is the one that sets the score. Every episode dies at ~60 s of a 200 s cap, so reward is
structurally 0.000 independent of task difficulty:

```
retail 21   last user words 27.8s → calls at 33.0 36.2 39.4 42.6 45.8 49.0 52.2 55.4 58.6 61.8
            get_order_details ×10, all {"order_id":"W0000000"}, all "Error: Order not found"
retail 92   find_user_id_by_name_zip ×2 → find_user_id_by_zip → find_user_id_by_name ×7
            args mutate May → Amelia → Maya; tool name decays into ones that do not exist
```

**This corrects 0d-bis: "the checkpoint invents tool names" is a symptom, not the disease.** In
retail 92 the model's *first* call is `find_user_id_by_name_zip` — the correct tool, which exists —
and the invented variants appear only after the error, as the repetition loop degenerates. Tool
*selection* is substantially right on the first attempt, consistent with the model card's 82.5 % on
Full-Duplex-Bench v3. The 27 % invented-name rate is the loop's output, so citing it as the target
for SFT aims at the wrong thing.

**The fix under test: the eval harness was omitting NVIDIA's tool-use protocol.** `provider.py` sent
only `policy + <AVAILABLE_TOOLS>`. The released checkpoint was trained with
`augment_fc_system_prompt: true` and NVIDIA's own inference entrypoint
(`examples/speechlm2/offline_voicechat_fc_infer.py`) supplies a protocol scaffold whose rules map 1:1
onto (b) and (c) — *"never guess … ask the user"*, and *"if a tool call fails … do not retry the tool
call for the same request"*. Now sent, verbatim, via `_FC_PROTOCOL_RULES` +
`DEFAULT_FC_SYSTEM_PROMPT_TEMPLATE`, gated by `NeMoDuplexConfig.fc_prompt_protocol` (default on) with
a `nemo-base-bare-prompt` preset as the control. Confirmed reaching the model: prompt **4,838 tokens
vs 4,575**. See the correction appended to 1c-bis — that section's reasoning is right for *our* cuts
and backwards for the *released* checkpoint.

**A second defect found while validating: telecom's policy contains 18 non-ASCII characters** (emoji
📶📵📡✈️📱🔽🔒🔋 and superscripts ¹²³⁴) and the model card requires ASCII-only system prompts — so 8
of 24 episodes were malformed. `_to_ascii()` now NFKD-folds and warns with a codepoint histogram.

**Partial paired evidence, and it is only partial.** Two protocol-run episodes passed the tick depth
at which their bare-prompt controls died, with no `too_many_errors` anywhere in the run:

| episode | control | with protocol |
|---|---|---|
| retail 78 | 288 → `too_many_errors` | **380, running** |
| retail 92 | 319 → `too_many_errors` | **398, running** |

No episode persisted before the quota ran out, so **there is no post-fix reward number and no
post-fix tool-call table.** Do not cite one.

**The experiment to run first on resume — it needs zero characters.** `both.wav` is **stereo 8 kHz,
one channel per speaker**, so every persisted episode contains the real, already-paid-for user audio,
and the label files give the control's exact calls. Replaying that channel into the model with the
bare vs protocol prompt, executing calls against the real environment
(`registry.get_env_constructor("retail")()` → `make_tool_call`, which reproduces `Error: Order not
found` exactly), is a paired deterministic A/B where the *only* variable is the prompt.
`scripts/check_streaming_driver.py` already has the pieces: `build_session`, `drive()` (200 ms ticks,
`<SOTC>` detection), and `push_tool_result`. Four episodes × 2 prompts = 8 runs ≈ one 8-GPU wave,
~15 min each. Metric: consecutive identical calls, and whether the model *speaks* after the first
error instead of retrying. This settles (b) and (c) but cannot settle (a).

**ElevenLabs is used for TTS only — so it is replaceable.** Verified, not assumed: two call sites,
both text-to-speech (`elevenlabs_utils.py:90` `text_to_speech.convert` for user utterances, and
`synthesis/audio_effects/speech_generator.py` for `[cough]`/`[sneeze]`/disfluency inserts, which also
bill). **No STT** — transcription is Deepgram `nova-3` *if enabled*, and it is not: `batch.py:497`
and `build.py:235` both pass `transcription_config=None`, which is also why `user_transcript` is
empty in every persisted tick and the text survives only in `*_labels.txt`. `setup_voices.py` does
call `text_to_voice.design`/`voices.create`, but we never run it (the official tau-bench voice IDs
404 on our key; stock IDs come from `tau2_stock_voices.env`). Nothing needs an ElevenLabs *voice*,
*model*, or *transcription* — only PCM_S16LE mono @ 16 kHz bytes for a string.

So a local TTS drops in behind one seam: `synthesis/synthesize.py:22` dispatches on the provider
string and `data_model/voice.py:327` raises on anything but `"elevenlabs"`. `nemo.collections.tts` is
already installed in the voicechat env (import it with `LD_LIBRARY_PATH=$ENV_PREFIX/lib`, per the
env recipe, or `_sqlite3` fails on `CXXABI_1.3.15`). The cost is voice identity: `tasks_voice.json`
assigns a per-task `voice_id` across 114 retail tasks, so a single-speaker local model removes
speaker variation as a difficulty axis — a mild bias in arm A's favour, to be **stated in the
results, not hidden**. Decision deferred: the user is obtaining an API key.

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

**1c. Raise `max_fc_total_tokens` to 12,000 and turn on `fc_log`. DONE and GPU-validated.**
Measured on the repaired shards with augmentation on: **8,389 / 8,137 / 8,254** — all three over
8,000, so the old value would now discard 100 % of the corrected data. 1b's saving does not close
the gap because 1a costs ~+2,000. 12,000 leaves 3.6–3.9k headroom on retail and is the smallest
round value that also clears telecom (system prompt 6,903 alone). `fc_log: true` is now set —
it defaults to `False`, which is exactly how 8,000 silently discarding everything would have gone
unnoticed. **GPU smoke test (8 steps, one H200, log in `data/voicechat/fc_check2/`): passes.**
Zero cuts dropped at 12,000 — the budget holds where 8,000 discarded 100 % of the repaired data.
The function channel carries the trained layout (`<SPECIAL_20>` then
`<TOOLCALL>[{"name": "find_user_id_by_name_zip"…`), with 6 response spans injected out to position
~10,056, matching the ~10.9k predicted for a canonical retail cut (4,478 prompt + 2,495 audio
frames + 3,909 inserted FC). Cost: **119,235 MiB of 143,771 MiB at `batch_size: 1`** — so batch
size 1 is a hard requirement at this sequence length, not a conservative choice.

**1c-bis. `augment_fc_system_prompt: true` is a no-op on our data — do not "fix" it at inference.**
Worth stating because the flag reads as if the model trained on the ~150-token `<TOOLCALL>`
instruction scaffold. It did not. `collate_system_prompt` (`s2s_dataset.py:2148`) resolves the
prompt from `cut.custom["system_prompt"]` on its **first** branch and only augments on the
following `elif`; our cuts populate `custom` (verified byte-identical to `supervisions[0].text`),
so the wrap never fires. Note `_get_fc_cut_total_prompt_tokens` (`:1209`) *does* augment
when computing the drop decision, so the 12,000 budget above was measured with ~150 tokens of
extra headroom; that direction is conservative and harmless.

> **CORRECTION (2026-08-19) — this section's scope was wrong, and it cost ~10 GPU-hours.** It
> previously concluded "the eval driver must send the **raw tools-only** prompt … adding the scaffold
> would itself be the train/test mismatch." That holds only for a checkpoint finetuned on **our**
> cuts. It is backwards for the **released** `NVIDIA-NemotronLabs-VoiceChat-11B`, which NVIDIA trained
> *with* `augment_fc_system_prompt: true` on *their* data and which their own inference entrypoint
> (`examples/speechlm2/offline_voicechat_fc_infer.py`) serves *with* the protocol scaffold. Sending it
> the bare prompt is itself the mismatch — see 0d-ter for the three failure modes that follow. The
> rule is per-checkpoint: **bare prompt for our SFT'd models, protocol scaffold for the released
> one.** The `[bos] + text_to_ids(prompt) + [eos]` wrapping (`:2220`) is unaffected and still required
> in both cases.

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
| 10 | **User-simulator TTS quota — MATERIALIZED 2026-08-19, stage 2 parked.** ElevenLabs is the only TTS path and the free tier is 10,000 chars/month; a 24-episode diagnostic needs ~18k and stage 3 ~188k *per arm*. Nothing else on the eval path is blocked. | **high** | paid key (in progress), or a local TTS behind `synthesize.py:22`; see 0d-ter |

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
