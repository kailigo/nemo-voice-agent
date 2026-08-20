# Reproducing the Full-Duplex-Bench v3 numbers on the model card

The [NVIDIA-NemotronLabs-VoiceChat-11B card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
reports three FDB-v3 numbers:

| metric | card |
| --- | --- |
| Tool Selection | **82.5 %** |
| Argument accuracy | **44.2 %** |
| Pass@1 | **33 %** |

This document is the audit trail for reproducing them with the released checkpoint: what the
benchmark actually requires, which parts of the published pipeline we run unchanged, which we
replaced and why, and what the numbers came out to.

Benchmark checkout: `/fsx/home/kai.li/code/Full-Duplex-Bench/v3` (unmodified — nothing in
this reproduction edits it).

## 1. What the benchmark actually needs

The released pipeline looks like it needs a LiveKit Cloud account and an OpenAI key. Read
end to end, it decomposes:

* **`livekit_inference.py` + `lk_agent_tool.py`** publish a WAV into a LiveKit room where an
  agent wrapping a *hosted* realtime API (GPT Realtime, Gemini, Grok, Ultravox) listens, and
  write `result_{provider}.json` into each example folder.
* **The three evaluators never touch LiveKit.** They glob `result_{provider}.json` and score
  JSON. `evaluate_tool_calls.py:640-652`.

So the transport is not load-bearing for any published metric: LiveKit is replaceable by any
client that can stream the WAV and log the tool calls.

### Which of the two inference paths to measure

This mattered more than anything else in the setup, and I got it wrong first.

* The **eager research path** in this repo (`duplex_stt_model.py:3718-3722`) hard-disables the
  LLM cache for Nemotron — it logs *"Using no-cache mode for Nemotron (full history each
  step)"* — so it recomputes the whole conversation every frame. Measured: **12.2x slower than
  realtime** on an H200. NVIDIA's own `offline_voicechat_fc_infer.py` runs this path too,
  which is why it looks canonical.
* The **served path** is the NIM container (`voicechat_realtime_instructions/`, CUDA + Triton +
  vLLM, one GPU, ~66 GB), which the model card's "Interactive streaming deployment" section
  points at. vLLM logs `GPU KV cache size: 1,105,920 tokens` at startup.

Measured on the container, 2026-08-19, one H200 (`scripts/voicechat_realtime_latency.py`):

| | NVIDIA's `what_is_your_name.wav` | FDB-v3 `ecommerce_01`, 45.8 s |
| --- | --- | --- |
| wall / input audio | 1.002 | 1.001 |
| output audio / arrival span | 1.01 | 1.02 |
| chunks dropped | 0 | 0 |
| turn-taking latency (ASR end-of-speech → agent speech onset) | 794 ms | 476 ms, 639 ms |
| inferences per second of audio | 6.27 | 6.25 |

The card quotes 448 ms smooth turn-taking latency; ours is measured from the server's own ASR
end-of-speech marker, which lags the acoustic end, so these are if anything pessimistic.

The two paths also *behave* differently, so this is not only a cost question. On `ecommerce_01`
the research path hallucinated `order_id: "LHR"`; the container returned
`track_order({"order_id": "A-BC123"})` for a user who spells "a B C one two three" out loud —
right tool, near-miss argument, which is the shape of the published 82.5 % / 44.2 %.

**Everything below is therefore measured through the container, not the research path.** Note
that the container is built from the *released* checkpoint: our locally remapped
`stt_extracted_lora` is missing all 635 `tts_model.*` tensors and cannot speak. (Weight
provenance is verified separately by `scripts/verify_checkpoint_identity.py`: 982 tensors match
after name normalisation and 11/11 sampled are bit-identical, so the shared trunk *is* the
released weights, renamed.)

Credential requirements, per metric:

| metric | needs a key? |
| --- | --- |
| Tool Selection (the 82.5 %) | **no.** Mean F1 of multiset recall/precision over tool *names*. Pure set arithmetic. |
| Argument accuracy | yes — a gpt-4o judge for semantic matching ("August 20" == "2026-08-20", ±5 % numeric). Without `--use-llm` it silently falls back to exact string match, which is a *different metric*, not a neutral one. |
| Pass@1 | yes — same judge. Strict binary: recall == precision == 1 **and** every argument matches. |
| Latency / response quality | judge, plus a Parakeet pass over the input audio. |

## 2. Coverage of the released data

* `benchmark_data_v2.json`: 100 scenario entries, 100 distinct ids.
* `fdb_v3_data_released/`: 100 example folders, **79 distinct scenario ids** — 21 scenarios
  have two speaker renditions, and 21 scenarios have no audio at all.

100 result files is therefore full coverage, and matches the README's "process all 100 audio
samples". The evaluators drop absent scenarios from the denominator silently, so a dead shard
produces a confident-looking score over a subset; `fdb_v3_evaluate.py::coverage` prints the
count before scoring for exactly that reason.

## 3. What we run

```
scripts/fdb_v3_tools.py                # the 12 tools + agent instructions, parsed out of lk_agent_tool.py
scripts/voicechat_realtime_latency.py  # realtime client: latency, realtime headroom, tool calls
scripts/fdb_v3_asr_input.py            # Parakeet over input.wav -> user_speech_end_rel (latency only)
scripts/fdb_v3_evaluate.py             # the benchmark's own evaluators, Bedrock judge patched in
scripts/fdb_v3_nemo_infer.py           # research-path replay; kept only as the A/B against the container
scripts/fdb_v3_fanout.sh               # 8 shards for the research-path replay
```

Serving, per `voicechat_realtime_instructions/` (docker on these nodes has no `nvidia` runtime,
so the container runs under `srun` + pyxis/enroot instead of `docker run`; the image pulls
anonymously from `nvcr.io`, no NGC key):

```bash
enroot import -o /fsx/home/kai.li/data/containers/nemotron-labs-voicechat.sqsh \
  docker://nvcr.io#nim/nvidia/nemotron-labs-voicechat:latest        # 11.3 GB -> 19.5 GB sqsh
# then, inside an allocation: NEMO_CHECKPOINT_PATH=/checkpoint /s2s/deploy_s2s_model.sh   (~12 min, 31 GB)
#                             NIM_HTTP_API_PORT=9000 /s2s/run_s2s_server.sh               (~6 min to ready)
curl http://<node>:9000/v1/realtime/health
```

The per-example driver against that server is the piece still being written; the client above
already streams one example end to end with tools registered.

**Faithful to the published setup:**

* Tool schemas, tool descriptions, parameter descriptions and the agent instructions are
  parsed out of `lk_agent_tool.py` with `ast` rather than transcribed, so our prompt cannot
  drift from the one the published table was produced with. If NTU restructures that file,
  extraction raises instead of quietly diverging.
* Tool execution goes through the benchmark's own `MockAPIRegistry` at the `instant` latency
  profile — the reference default (`lk_agent_tool.py:52`; the released `run_agent.sh` never
  passes `--latency`, and the per-scenario `latency_profile` metadata field is read by
  nothing in the released pipeline).
* Tool results are returned as `json.dumps(result)`, byte-for-byte what `AssistantFnc` hands
  its model. (`--tool-response-style sentence` follows the model card's TTS-friendly
  recommendation instead; it is *not* what the other providers in the table received, so it
  is off by default.)
* A call lands in `actual_tool_calls` only if it would have executed under LiveKit — known
  name, all required arguments present. LiveKit's function tool rejects anything else before
  `log_tool_call`, so those calls are invisible to the published F1; counting ours would
  penalise our precision for failures the reference silently drops. Rejects are kept in
  `rejected_tool_calls`, which is the best diagnostic in the result file.
* Audio plus trailing silence. `livekit_inference.py:270-282` appends 1.5 s; the WAVs already
  contain the response gap (median ~47 s of file for ~10 s of speech). The container needs more
  than LiveKit did — it is full duplex and only emits while input flows, so `deploy.md`
  recommends ~20 s of trailing silence or the final reply is truncated mid-sentence.

**Deviations, each stated in every result file's `notes` block:**

1. **No LiveKit.** The WAV is streamed to the container's `/v1/realtime` WebSocket at 24 kHz
   PCM16 in 80 ms chunks, paced at true realtime, which is what LiveKit would have done to a
   hosted realtime API. Tool schemas are registered flat (`{name, description, parameters}`)
   via `session.update`, so the *server* renders the function-calling prompt — our earlier
   hand-rolled tool block is no longer in the loop.
2. **The prompt is NOT the benchmark's `VoiceAgent` instructions alone — the server appends
   NVIDIA's tool-restraint text.** This was discovered on 2026-08-20, after the run below, and
   it invalidates the "instructions only" claim that stood here before.

   `audio_server.py:53` gates prompt construction on `USE_JINJA_TEMPLATE_PROMPT`, which
   **defaults to `"0"`**. Only the enabled branch uses `/s2s/prompt_template.jinja`. The default
   branch (`:1179-1198`) does:

   ```python
   if instructions: prompt += instructions
   ...
   if prompt: prompt += "\n\n"
   prompt += TOOLS_TEMPLATE.format(tools_content=clean_tools_json)
   ```

   and `TOOLS_TEMPLATE` (`audio_server.py:68`) opens with a newline followed by NVIDIA's
   decision-process text. The server log confirms it: our instructions, then exactly `\n\n\n`,
   then "When you receive a request, follow this decision process:", on 104 sessions — and zero
   occurrences of the branch's "Preparing prompt using jinja template" log line.

   So every session in the run below received two directly contradictory prompts concatenated:

   | the benchmark's instructions say | the server appended |
   | --- | --- |
   | "Execute the tool unconditionally!" | "DO NOT use any tools when not needed, under no circumstance" |
   | "DO NOT ASK CLARIFYING QUESTIONS" | "If a required argument is missing, ask the user; never guess" |

   `USE_JINJA_TEMPLATE_PROMPT=1` gives the clean instructions-only prompt that the other
   providers in the published table received; the jinja template carries no decision-process
   text. That is the faithful arm and it is what `nemo_rt_jinja` measures.

   Corollary: `--system-message nvidia+benchmark` is *not* the control arm it looks like. The
   restraint text is already present by default, so that flag duplicates it rather than adding
   it.
3. **The judge is Claude Sonnet 4.5 via Bedrock, not gpt-4o.** We have no OpenAI key;
   Bedrock authenticates off the instance IAM role. The metric definition is unchanged and
   the prompts are the benchmark's own, but a stricter or looser judge moves argument
   accuracy and Pass@1 directly. Tool Selection is unaffected — it never calls the judge.
   The substitution is printed at the top of every run.

The judge is patched in at the one seam each evaluator uses to get a client
(`_get_openai_client`, or the module-level `OpenAI` in `analyze_tool_latency.py`), so no
proxy server and no edit to the benchmark is needed. `_patch` raises if neither seam is
present rather than leaving the judge pointed at OpenAI.

## 4. Results

Probe, `ecommerce_01`, 2026-08-19, through the container with all 12 tools registered:

```
tool call @16.87s  track_order({"order_id": "A-BC123"})
agent: "I can track that order of your for you. Your order has been received and is now in transit."
user speech end 10.79s -> agent speech onset 11.27s   latency 476 ms
user speech end 18.47s -> agent speech onset 19.11s   latency 639 ms
wall/audio 1.001   output_realtime_ratio 1.02   chunks_dropped 0
```

Right tool; the argument is a near-miss on formatting — the user spells "a B C one two three"
out loud (`acting_notes`: "Say the order ID clearly, one character at a time"), expected
`order_id: "ABC123"`, got `"A-BC123"`. That is the shape of the published result: high tool
selection, argument accuracy roughly half of it. For contrast, the research path on the same
example emitted `order_id: "LHR"` — an airport code, from a different domain's tool.

Cost: realtime. 78.6 min of benchmark audio is ~79 min on one GPU, ~12 min wall across the 8
in an allocation with one server per GPU. (The research-path replay would have been ~16
GPU-hours; that arm is now only worth running as a deliberate A/B.)

### Full 100-example run, 2026-08-19

`scripts/fdb_v3_realtime_infer.py --provider nemo_rt` over all 100 audio folders on one H200,
then `scripts/fdb_v3_evaluate.py --provider nemo_rt` with the Bedrock judge (274 judge calls, 0
failures).

Coverage: **99 completed, 1 `inference_error`**, 2 silent. The failure is
`ecommerce_20_66c4f3cb14cbfc4db836bd4e`, and it is **deterministic, not a race** — three
attempts (`scripts/fdb_v3_retry_failures.sh`, 2 rounds) all died the same way, so the earlier
"race" reading was wrong for this example. The mechanism is the released Triton backend's
degenerate-tool-call recovery: the model opens a tool call and never emits the end-of-tool-call
token → `Fast extract: exceeded 512 steps without eotc_id` → the vLLM request meanwhile hits
EOS → `append_request: request '…' not found` → HTTP 500 → WebSocket 1011. It is left in place
and scored as a no-response rather than retried away.

| metric | ours (n=98) | ours (n=100) | card | judge? |
| --- | --- | --- | --- | --- |
| Tool Selection | **73.1 %** | 71.7 % | 82.5 % | **no** |
| Argument accuracy | **51.7 %** | 50.7 % | 44.2 % | yes |
| Pass@1 | — | **35.0 %** | 33.0 % | yes |

Turn-taking 98/100. `n=98` drops the two samples that produced no output; `n=100` scores them 0.
The card does not say which convention it quotes, so both are reported.

**The two misses point in opposite directions and cancel in Pass@1** — which is why Pass@1 lands
on target and is the weakest of the three as evidence. The split falls exactly along whether the
judge is involved:

* **Argument accuracy overshoots by 7.5 pts, and it is judge-mediated.** Sonnet 4.5 being more
  permissive than gpt-4o on semantic argument matching is the leading explanation; a stricter
  judge moves this number directly and we cannot settle it without an OpenAI key. Beating the
  card here is not a result to claim.
* **Tool Selection undershoots by 9.4 pts, and it uses no judge at all** —
  `evaluate_tool_calls.py::evaluate_tool_selection` is multiset F1 over tool *names*, pure set
  arithmetic. So this gap is real and cannot be attributed to the substitution. **This is the
  one number that does not reproduce.**

Where Tool Selection loses, per `logs/fdb_v3/nemo_rt_evaluation_report.json`:

| cut | value |
| --- | --- |
| mean recall / mean precision | 0.789 / 0.862 |
| scenarios losing precision only / recall only | 19 / 20 |
| total calls emitted vs expected | 184 vs 150 (**1.23x — the model over-calls**) |
| call-count shape | 52 right, 27 over, 21 under |
| by difficulty (easy / medium / hard) | 0.758 / 0.770 / 0.609 |
| by expected-call count (1 / 2 / 3) | 0.752 / 0.608 / 0.698 |

The loss is **broad, not one failure mode**: 19 scenarios lose only precision and 20 lose only
recall, so there is no single fix. Most frequent spurious calls are `update_identity_doc` (10),
`get_exchange_rate` (9), `search_products` (9) — plausible-but-unasked actions, the signature of
an agent being too eager rather than confused about the tool set.

**Leading hypothesis for the gap, and the next arm to run:** deviation 2 above — we send the
benchmark's `VoiceAgent` instructions *alone*, and NVIDIA plausibly produced the card's numbers
with their own `DEFAULT_SYSTEM_MESSAGE` prepended, which contains explicit restraint about when
to call tools. `--system-message nvidia+benchmark` runs that arm; it is a full re-run, ~80 min on
one GPU. Until it is run, the honest statement is: **Pass@1 reproduces, argument accuracy is not
comparable because the judge differs, and Tool Selection is 9.4 points short for reasons internal
to our setup.**

Not yet measured: the latency section. `analyze_tool_latency.py` reports `total_samples: 0`
without `user_speech_end_rel`, which needs `scripts/fdb_v3_asr_input.py` (Parakeet over
`input.wav`) to run first.
