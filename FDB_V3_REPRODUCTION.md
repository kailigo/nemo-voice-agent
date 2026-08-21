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

**Where it landed:** Pass@1 reproduces (35.0 % vs 33 %), argument accuracy is not comparable because
the judge differs, and Tool Selection is 10.8 points short. Eleven candidate deployment discrepancies
were measured and eliminated — see [the discrepancy audit](#the-discrepancy-audit-2026-08-21--eleven-suspects-eliminated-and-where-the-11-points-are).
The gap is half spurious tool firing, half recall shortfall, and all of it in the 30 `hard`
scenarios. The public harness contains no NVIDIA provider, so the integration behind 82.5 % cannot be
diffed against ours. A final path-vs-path test on those 30 scenarios identified the mechanism: the two
inference paths sit at different points on an eagerness/caution trade-off, and Tool Selection — a
function-*name* metric — rewards eagerness, so the same episodes reverse order once arguments are
scored. That makes the card's own 82.5 % / 44.2 % pair coherent against our 71.7 % / 50.7 %, and the
numbers are **accepted as measured**.

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

The card quotes 448 ms smooth turn-taking latency. Ours is measured from the server's own ASR
end-of-speech marker, and **that makes it optimistic, not pessimistic** — corrected 2026-08-21, see
§5. The marker does not coincide with the instant the user stopped talking: measured across the full
run it sits a median 2.3 s *after* the acoustic end of the user's turn on the 63 examples the
benchmark's own latency metric keeps, and *before* the turn has finished on the other 35. A latency
measured from a late anchor is small by construction.

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
`ecommerce_20_66c4f3cb14cbfc4db836bd4e`, and it is **deterministic on this prompt branch, not a
race** — three attempts (`scripts/fdb_v3_retry_failures.sh`, 2 rounds) all died the same way, so
the earlier "race" reading was wrong for this example. It is not a fixed property of the example
either: the same audio completed cleanly on the jinja branch (§4), so the crash path is
prompt-dependent. The mechanism is the released Triton backend's
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

### The faithful-prompt arm, `nemo_rt_jinja`, 2026-08-21 — the prompt was not the cause

The hypothesis that stood here until 2026-08-21 was that the 9.4-pt Tool Selection gap came from the
prompt: deviation 2 showed all 100 sessions carried NVIDIA's restraint text appended after the
benchmark's contradictory "Execute the tool unconditionally!", which no other provider in the card's
table received. `USE_JINJA_TEMPLATE_PROMPT=1` removes it. That arm has now been run —
100 examples, provider `nemo_rt_jinja`, branch verified from the server log (`Preparing prompt using
jinja template` ×100, `Call a tool ONLY when the user` ×0).

**It did not close the gap. It moved every metric slightly the wrong way.** Comparing at n=100,
which is the only fair comparison because the jinja arm has no silent episodes:

| metric | `nemo_rt` (restraint text) | `nemo_rt_jinja` (clean) | card | judge? |
| --- | --- | --- | --- | --- |
| Tool Selection | 71.7 % | **70.8 %** | 82.5 % | no |
| Argument accuracy | 50.7 % | 48.2 % | 44.2 % | yes |
| Pass@1 | 35.0 % | 31.0 % | 33.0 % | yes |

The mechanism is worth more than the headline. Removing the restraint text **did** make the model
less eager — 184 calls → 162, i.e. 1.23x → **1.08x** of the 150 expected — and yet:

| | `nemo_rt` | `nemo_rt_jinja` |
| --- | --- | --- |
| mean recall | 0.778 | **0.765** |
| mean precision | 0.856 | 0.853 |
| call-count shape (right / over / under) | 52 / 27 / 21 | 49 / **29** / 22 |
| losing precision only / recall only / both | 19 / 21 / 8 | 23 / 22 / 8 |
| by difficulty (easy / medium / hard) | 0.758 / 0.770 / 0.609 | 0.732 / 0.770 / **0.609** |

**Precision did not improve despite 22 fewer calls, and recall fell.** So the calls the clean prompt
suppressed were disproportionately *correct* ones. Note the count of scenarios that over-call went
*up*, 27 → 29, while the total number of calls went down: fewer calls, spread worse. Hard scenarios
are identical to three decimal places (0.609) and medium is unchanged (0.770); the whole delta is on
easy (0.758 → 0.732).

So the restraint text was mildly *helping*, or at worst neutral — the third of the three outcomes
this arm was set up to distinguish, and the one that says something about the card rather than about
us. **The Tool Selection gap is real, judge-free, and internal to our setup. It is not a launch-flag
artifact.** `--system-message nvidia+benchmark` remains not worth GPU time: it duplicates text that
is already present by default and that we now know does not drive the gap.

The honest statement, updated: **Pass@1 reproduces on the default branch (35.0 % vs 33.0 %) and
falls below on the clean one (31.0 %); argument accuracy is not comparable in either arm because the
judge differs; and Tool Selection is 9.4–11.7 points short on both prompt branches, which rules out
the prompt as the explanation.** What remained untested at that point was the harness itself —
tool-call extraction, the `instant` latency profile, and the trailing-silence convention — plus the
possibility that the card's number was produced with a checkpoint or decode configuration the
released container does not reproduce. All of that is the subject of the audit below.

One incidental finding: `ecommerce_20_66c4f3cb14cbfc4db836bd4e`, described above as a
**deterministic** backend failure that survived three retries, **completed cleanly on the jinja
branch** with three sensible calls. So that crash is prompt-dependent, not a fixed property of the
released Triton backend. The jinja arm also has zero silent episodes, against two on the default
branch.

### The discrepancy audit, 2026-08-21 — eleven suspects eliminated, and where the 11 points are

The prompt arm ruled out one explanation and left the gap unattributed. Rather than accept it, every
difference we could find between the card's suggested setup and our deployment was checked. Each row
below is a measurement, not a reading of the docs.

| suspect | verdict | evidence |
| --- | --- | --- |
| Checkpoint provenance — Triton repo not built from the released weights | eliminated | triton-model-repo config vs released `voicechat-11b` config: **0** keys unique to either side, exactly **1** differing value (`model.stt.model.pretrained_asr`: `None` vs `''`). Against `stt_extracted_lora`, 339/340 keys mismatch — **config layout only, not weights**; those are bit-identical (see the research-path test below). It is the released checkpoint. |
| Prompt branch / restraint text | eliminated | both arms run at n=100, 71.7 % and 70.8 % (§4 above). |
| Trailing-silence convention (our 20 s vs FDB's 1.5 s) | eliminated | only **2 of 184** calls fall outside FDB's own capture window (`livekit_inference.py`: `target_samples = duration_sec × 24 kHz`, +1500 ms). Truncating to their window: 71.7 → 71.9 %, 70.8 → 70.8 %. |
| LiveKit VAD / turn-detection layer we don't have | eliminated | `lk_agent_tool.py:413` is `AgentSession(llm=model, tools=tools)` — no `vad`, no `turn_detection`. There is no layer to be missing. |
| Tool-result latency profile | eliminated | the per-scenario `latency_profile` field (normal 78 / fast 12 / slow 10) is **dead data — no code reads it**. Only the process-global `--latency` exists, default `instant` at `lk_agent_tool.py:52`, which is what we ran. |
| Input sample rate | eliminated | we send 24 kHz, which is exactly `api-reference.md:131` ("Client input sample rate 24 kHz"; the server resamples to 16 kHz internally). |
| ASR quality upstream of tool choice | eliminated | container-vs-Parakeet WER mean 0.21, p50 0.17. Tool Selection by WER quartile: 69.0 / 64.3 / 78.4 / 75.2 % — flat and non-monotone. Transcription is not what picks the tool. |
| Tool schemas the model sees | eliminated | AST diff of `lk_agent_tool.py` against `fdb_v3_tools.build_tools()`: **12/12 tools identical** — names, descriptions, per-parameter descriptions from the `Args:` docstrings, and required-argument sets. Zero mismatches. |
| Agent instructions | eliminated | `VoiceAgent.instructions` vs `extract_instructions()`: 1213 chars, **byte-identical**. |
| Tool-result payloads fed back | eliminated | both sides call FDB's own `MockAPIRegistry` and return `json.dumps(result)`; we deliver it as `function_call_output`. Same bytes. |
| Strict reject accounting penalising us (calls dropped for missing required args never enter `actual_tool_calls`) | eliminated, and the distinction is **empty** | **0** rejected calls and **0** unparseable-argument calls across both arms. Every call the model emitted was a valid tool name with all required arguments. Scoring leniently changes nothing: 71.7 → 71.7 %. |
| Decode configuration | eliminated | container defaults are `LLM_TEMPERATURE=0.0`, `LLM_TOP_P=1.0`, `LLM_REPETITION_PENALTY=1.0` (`model.py:193-195`), and `infer/utils.py:186` short-circuits to `greedy_tokens` under exactly those values. Decoding is deterministic greedy — the gap is not sampling noise and there is no decode win available. |
| Metric convention (the card scoring the same behaviour a different way) | eliminated | our multiset F1 reproduces the official evaluator exactly at 71.7 %. Alternatives, all at n=100: set F1 76.0, recall-multiset 77.8, precision 69.6, recall-set 78.3, set-coverage 72.0, exact-match 52.0, micro-pooled F1 69.5. Per-scenario instead of per-recording: first speaker 73.1, best-of-speakers 74.8. **82.5 % is above every one of them.** |

**The structural finding, which limits how far any audit can go: the public FDB-v3 harness has no
NVIDIA/Nemotron provider at all.** `lk_agent_tool.py::get_realtime_model()` supports grok,
gpt_realtime, azure_openai, gemini2_5, gemini3_1 and ultravox — nothing else. The only `nvidia`
string anywhere in v3 is the Parakeet ASR used for *scoring*. The README credits NTU with the
benchmark and NVIDIA with "a collaborative research discussion and advisory role", so the integration
that produced 82.5 % was never released. Our client is a reimplementation against the same tools,
instructions, mock APIs and scorer; it cannot be diffed against theirs, because theirs is not public.

#### Where the 11 points actually are

With the plumbing eliminated, the shape of the gap is the answer. Split by how many calls a scenario
expects:

| expected calls | n | expected | actual | matched | recall | precision | mean F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 66 | 66 | 66 | 52 | 78.8 % | 78.8 % | 75.2 % |
| 2 | 18 | 36 | **55** | 27 | 75.0 % | **49.1 %** | 60.8 % |
| 3 | 16 | 48 | **63** | 37 | 77.1 % | **58.7 %** | 69.8 % |
| all | 100 | 150 | 184 | 116 | 77.3 % | 63.0 % | 71.7 % |

**Recall is flat** (78.8 / 75.0 / 77.1 %) — the model is not failing to chain calls, which was the
obvious hypothesis and is wrong. Precision collapses instead: 184 calls against 150 expected, and the
excess is concentrated exactly where chains are (2-call scenarios emit 1.53x the expected count). The
arithmetic closes off one direction entirely: to reach F1 = 0.825 at our precision, recall would have
to be **1.19** — impossible; at our recall, precision would have to be 0.884 against our 0.630.

Classifying all 184 calls into mutually exclusive buckets (`scripts/fdb_v3_gap_anatomy.py`): 116
matched an expected slot, **36 called a function the scenario never expects** (24 of them
**cross-domain**), 20 were exact duplicates of an earlier call (same function *and* arguments, in only
9 examples), 12 over-counted an expected function with new arguments. Duplicates are counted first, so
a spurious call re-issued twice lands in the duplicate bucket; counting every never-expected call
regardless of duplication gives 43, of which 31 are cross-domain. The
spurious calls are not garbage — their arguments are scenario-grounded, which is what makes them
diagnostic:

```
housing_18  t=21.4/29.4/37.0s  get_exchange_rate {"amount": 700,  "USD"->"EUR"}   x3
housing_24  t=30.2/31.3/34.8s  get_exchange_rate {"amount": 1500, "USD"->"EUR"}   ("update my max price to 1500")
ecommerce_23 t=15.8s           get_exchange_rate {"amount": 100,  "USD"->"EUR"}
```

No user in any of these scenarios mentions currency. `housing_24` says *"update my max price to 1500
because I'm on a tighter budget"* and the model fired a currency conversion on the number it heard.
`ecommerce_23` is titled **"Unsupported Tool (Refund) + Valid Requests"** — the user asks to "process
a refund", no refund tool exists, and the model substituted the nearest money-shaped tool instead of
declining. These are hallucinations, not mislabelled ground truth.

Pricing them with an oracle that deletes every spurious and duplicate call while keeping recall:

| | `nemo_rt` | `nemo_rt_jinja` |
| --- | --- | --- |
| as measured | 71.7 % | 70.8 % |
| − exact duplicates | 73.2 % | 71.7 % |
| − calls to never-expected functions | 75.4 % | 75.1 % |
| − both (the oracle) | **77.3 %** | **76.0 %** |
| oracle, by difficulty (easy / medium / hard) | 77.8 / **83.1** / 70.2 % | 75.0 / 83.5 / 68.6 % |

So the 10.8-point gap is **two roughly equal halves**: ~5.6 points of spurious firing and duplicates,
and ~5.2 points of residual recall shortfall that no precision fix touches. Both live in `hard`
(n=30): 60.9 % as measured, still only 70.2 % under the oracle, while `medium` reaches 83.1 % — i.e.
**on easy and medium scenarios we are at card level once over-firing is removed, and the entire
residual is the 30 hard scenarios**, which are the disfluency-heavy multi-tool chains
(`PAUSE`, `FILLER`, "Filler-Heavy Multi-Tool Chain", "Hesitant Three-Tool Chain").

**Conclusion.** Eleven candidate deployment discrepancies were checked and eliminated with
measurements; none of them accounts for the gap, and the remaining behaviour is a property of the
served model on hard disfluent chains — over-firing tools it was never asked for, and missing about a
quarter of the ones it was. The one difference that cannot be closed is not in our control: the
integration behind the card's 82.5 % is not in the public repo, so no further diff against it is
possible from released artifacts. Anything more would have to come from NVIDIA — specifically whether
their FDB-v3 row was produced through this container at all, or through the research path with a
different tool-call surface.

### The research-path test, 2026-08-21 — the gap is eagerness, and Tool Selection rewards it

The audit's last open item — container or research path? — is testable without NVIDIA, because both
paths run the same weights. `scripts/fdb_v3_nemo_infer.py` was given a `--difficulty` filter and run
over the 30 `hard` scenarios that hold the entire residual, as provider `nemo_research_hard`.

**Same weights, first.** The research path loads `stt_extracted_lora`, which the table above records
as mismatching the released config on 339/340 keys — but that is a *config-layout* difference, not a
weights difference, and reading it as the latter would have made this comparison meaningless.
`scripts/verify_checkpoint_identity.py`: 9/9 sampled tensors **bit-identical** to `voicechat-11b`, 0
shape or dtype differences, the 2 ours-only tensors are copies of released ones, and the 650
released-only tensors are all `tts_model.*` (635) + `rnnt_decoder.*` (9) + `rnnt_joint.*` (6) — the
speech-synthesis and RNNT heads the research path does not use. The comparison is path vs path.

| arm, 30 `hard` scenarios | expected | actual | matched | recall | precision | Tool Selection | statuses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nemo_research_hard` | 72 | 86 | 51 | 70.8 % | 59.3 % | **72.7 %** | 30 completed |
| `nemo_rt` | 72 | 94 | 51 | 70.8 % | 54.3 % | 60.9 % | 29 completed, 1 `inference_error` |
| `nemo_rt_jinja` | 72 | 75 | 48 | 66.7 % | 64.0 % | 60.9 % | 30 completed |

The research path wins the hard bucket by 11.8 points — and the mechanism is not tool *choice*.
Matched calls are identical (51 vs 51), spurious-function counts are identical (14 vs 14). What
differs is **silence: 5 of 30 container episodes emit no tool call at all, against 1 for the research
path.** That is the whole deficit.

Two explanations were tested and both are refuted:

* **Context pressure against the 6144-token window** — refuted, and backwards. The zero-call
  episodes are *shorter* than average: audio p50 19.3 s vs 20.8 s, max 24.2 s vs 50.2 s. The
  episodes that overflow the window are the ones that answer.
* **Barge-in / committing to speech before the request finishes** — refuted for 4 of the 5. In
  `finance_18` (first agent onset 20.9 s, user audio ends 19.4 s), `finance_23` (18.6 / 17.9),
  `housing_19` (14.5 / 13.5) and `housing_21` (23.0 / 19.3) the agent begins speaking *after* the
  user has finished, with nothing left to interrupt. Only `travel_23` fits the pattern (a 2.8 s
  greeting ahead of a later request).

What the container does instead is **ask for the missing argument** — "I just need to know how much
you would like to convert", "I just need your starting address", "Just let me know the city you are
in" — which is exactly what the benchmark's own instructions forbid in capitals ("DO NOT ASK
CLARIFYING QUESTIONS… DO NOT reply with a question or conversational filler instead of calling the
tool"). It is a real deficiency against this benchmark. It is not a broken serving path.

**And the research path's advantage is largely an artifact, because Tool Selection scores function
names only.** Rescoring the same 30 episodes with arguments included reverses the order
(`scripts/fdb_v3_name_vs_args.py`, greedy one-to-one matching so a tool fired five times cannot
earn five credits):

| arm, 30 `hard` | name-only F1 | name + args (lenient) | name + args (strict) |
| --- | --- | --- | --- |
| `nemo_research_hard` | **71.6 %** | 22.5 % | 19.3 % |
| `nemo_rt` | 56.4 % | **25.4 %** | 21.8 % |
| `nemo_rt_jinja` | 55.5 % | **28.8 %** | **22.7 %** |

"lenient" requires the actual call to match every argument the expected call specifies and forgives
extras; "strict" is full dict equality; both normalise case, leading articles and punctuation, and
both drop the 13 of 72 expected calls whose arguments contain a `$RESULT_n` placeholder, since those
can never match. Name-only differs slightly from the table above because of that denominator change.

The four episodes where the container fired nothing show what the name-only metric is paying for:

```
finance_23   expected  modify_autopay {"bill_type":"mortgage","source_account":"savings"}
             research  modify_autopay {"bill_type":"credit_card", ...}    <- wrong bill, full credit
finance_18   expected  get_exchange_rate + get_card_benefits {"card_type":"premium"}
             research  get_exchange_rate {1000, USD->EUR} correct; get_card_benefits {"platinum"}
housing_21   expected  search_apartments {"bedrooms":3} + calculate_commute {...}
             research  search_apartments {"city":"Tokyo","bedrooms":3,"max_price":1000}
                       + search_flights x8  (Tokyo/LHR/JFK/ORD, repeated)
housing_19   expected  calculate_commute {"origin":"my house","dest":"the gym","mode":"driving"}
             research  essentially correct                               <- one clean win of four
```

`housing_21` is the clearest case: a runaway hallucination that emits eight spurious flight searches
in an apartment-search episode still outscores a model that asked one clarifying question, because it
matched `search_apartments` once.

**Conclusion, and why this closes the thread.** The container is not misconfigured and the research
path is not better; the two paths sit at different points on an eagerness/caution trade-off, and
FDB-v3's headline metric is scored on the axis that rewards eagerness. This finally makes the card's
own two rows coherent: **82.5 % selection against 44.2 % argument accuracy** is the signature of a
system that fires tool names readily and gets their arguments wrong, and **our 71.7 % / 50.7 %** is
the signature of a more cautious one. Both rows are consistent with the same weights behind different
orchestration, which is the most that released artifacts can establish. Two of the card's three
metrics reproduce, the third is characterised down to its mechanism, and the numbers are accepted as
measured. The all-100 numbers below supersede the `71.7 % / 50.7 %` pair quoted here, which was the
hard-30 slice scored with our own argument convention.

### The all-100 close, 2026-08-21 — we bracket the card rather than match it

The research arm was finished over the remaining 70 easy+medium scenarios and merged into one
provider with `scripts/fdb_v3_merge_arm.py` (the evaluators glob `*/result_<provider>.json`, so the
arm *is* the filename; merging rewrites the `provider` field to agree, refuses to clobber, and checks
final coverage, because a partial arm scores *higher* while measuring less). Running the benchmark's
own `evaluate_tool_calls.py` on all three arms at 100 examples:

| arm, all 100 | Tool Selection | turn-taken | Argument acc, exact-match |
| --- | --- | --- | --- |
| `nemo_research` | **78.3 %** | 99/100 | 22.2 % |
| `nemo_rt` | 73.1 % | 98/100 | 36.7 % |
| `nemo_rt_jinja` | 70.8 % | 100/100 | 32.8 % |
| card | **82.5 %** | — | **44.2 %** |

Tool Selection lands **4.2 points short**, and the residual is fully accounted for: 40 never-expected
calls (27 of them *cross-domain*) plus 7 exact duplicates. Deleting those while leaving recall
untouched scores **82.1 %**, within 0.4 of the card. It is a precision deficit, not a recall deficit.

**The argument row was not comparable, and that mattered.** The card's 44.2 % came from `--use-llm`,
whose judge is told to forgive `$RESULT` references, date formats, aliases, ±5 % numerics and
underscore-vs-space; `exact_match_args` forgives none of it. So 22.2 % / 36.7 % / 32.8 % are a strict
lower bound on the same behaviour, not a deficit. `scripts/fdb_v3_judge_args.py` supplies the
comparable number — 209 distinct verdicts (345 before dedup), two judge families, per-verdict audit
logs under `logs/fdb_v3_judge/`:

| arm | exact-match | judge: Sonnet 4.5 | judge: gpt-oss-120b |
| --- | --- | --- | --- |
| `nemo_research` | 22.2 % | 30.8 % | 26.1 % |
| `nemo_rt` | 36.7 % | **51.7 %** | **47.3 %** |
| `nemo_rt_jinja` | 32.8 % | 48.2 % | 43.7 %–44.7 % |
| card | — | **44.2 %** | **44.2 %** |

**On both judges the container path meets or beats the card on arguments (47.3–51.7 % vs 44.2 %)
while sitting 9.4 points below it on names, and the research path is closer on names but far below on
arguments.** The card's operating point lies *between* our two arms, on the same eagerness/caution
axis §4 identified. That is a stronger and more useful result than a match would have been: it says
the released weights can be driven to either side of the card's numbers by orchestration alone.

Three caveats, all recorded rather than smoothed over:

* **The card's judge is unreproducible here.** It is `gpt-4o`; there is no OpenAI key on this cluster
  and Bedrock's catalogue has no gpt-4o. These are our judges applying the card's rubric verbatim
  (`assert_rubric_unchanged()` fails loudly if the upstream rubric drifts from the copy in the
  script), not the card's number.
* **The judges disagree by up to 4.7 points**, so quote a range and name the judge. gpt-oss is
  systematically stricter (rejections −6/−5/−7 against Sonnet's −3/−0/−2), and the audit shows why:
  both reject calls carrying *extra* fabricated arguments — a `passenger_name: "user_1234"` on an
  `add_to_cart`, a hallucinated `max_price: 1000` on `search_apartments` — which `exact_match_args`
  forgives by construction, since it only iterates the expected keys. The rubric is silent on extras,
  so that strictness is judge-invented; it is defensible, and it is not the card's convention.
* **One verdict was unobtainable** — Bedrock returns an empty completion for a passport-number
  comparison, evidently a content filter. It is reported as an interval (both substitutions scored)
  rather than dropped. This matters because `llm_judge_argument` wraps its call *and* its parse in a
  bare `except Exception: return exact_match_args(...)`: under plain `--use-llm` a filtered response,
  a throttle or a truncated body silently becomes an exact-match verdict and the run reports ~22 %
  looking like it judged. The script counts judged / unparseable / api-failed separately and
  suppresses the number outright above a 5 % miss rate.

## 5. Latency, measured 2026-08-21

`scripts/fdb_v3_asr_input.py` (Parakeet over `input.wav`, 100/100, 0 failures) supplied the missing
`user_speech_end_rel`, so `analyze_tool_latency.py` runs for the first time. Re-scoring changed no
headline metric — nothing in §4 reads that field.

`logs/fdb_v3/nemo_rt_latency_report.json` and `…_nemo_rt_jinja_latency_report.json`. Note the N
column: it is the count *surviving* the interruption filter, so it differs per arm and the two
columns are not the same sample.

| | `nemo_rt` N | mean | median | `nemo_rt_jinja` N | mean | median |
| --- | --- | --- | --- | --- | --- | --- |
| First response | 63 | 4.05 s ± 3.22 | 3.04 s | 69 | 3.97 s ± 4.16 | 2.80 s |
| Tool call | 51 | 1.46 s ± 1.93 | 1.12 s | 55 | 1.77 s ± 3.47 | 1.20 s |
| Task completion | 63 | 4.82 s ± 5.45 | 3.20 s | 69 | 4.18 s ± 4.39 | 2.80 s |
| Filler usage | 3/63 (5 %) | | | 5/69 (7 %) | | |
| dropped as interruptions | 35/98 (36 %) | | | 31/100 (31 %) | | |

**Neither this 4.05 s nor the 0.64 s quoted in §1 is comparable to the card's 448 ms**, and the
reason is the anchor, not the model:

* **FDB anchors on the acoustic end of the user's first turn** — Parakeet word timestamps split on a
  >2.0 s gap, the benchmark's own rule (`run_tool_benchmark.py:358`).
* **Our `response_latency_s` anchors on the server's ASR end-of-speech marker.** Aggregated over all
  100 result files: **135 turns on 98 examples, mean 0.730 s, median 0.64 s, floor 0.48 s, p90
  0.80 s.** First turn only: n=98, median 0.64 s.

Reconciling per example (`marker = agent_first_word − response_latency_s[0]`, `delta = marker −
acoustic end of turn 1`):

| set | N | delta median | reading |
| --- | --- | --- | --- |
| FDB keeps (FR ≥ 0) | 63 | **+2.32 s** | server called the turn over *after* the audio did |
| FDB drops (FR < 0) | 35 | **−11.28 s** | server called it over *mid-turn*; the agent started talking |

**The 35 dropped examples are the finding.** All 35 have a clean server-anchored first-turn latency
(0.48–0.96 s, median 0.64, every one positive), so nothing looks wrong server-side. What happened is
that end-of-speech fired at an *intra-turn* pause. `ecommerce_01_65e8cf8f4c7424fa062e54a3` — the
user's turn 1 is one continuous 14 s utterance, 1.20 → 15.28 s, no internal gap above 0.56 s:

```
 8.88-10.16  "Could you track it for me?"    <- server calls end-of-speech (~10.32 s)
10.80        agent's first word                 server-anchored latency 0.48 s
10.48-15.28  "The order ID is A B C one two three."   <- same user turn, still going
```

FDB scores FR = 10.80 − 15.28 = **−4.48 s** and drops it as an interruption. Both measurements are
correct; they disagree about where a turn ends.

So the honest latency claim is: **once the server decides the user has stopped, it replies in a
median 0.64 s — but on 35 of 98 examples (36 %) it decides that mid-turn and talks over the user.**

**Open hypothesis.** That 36 % barge-in rate and the 1.23x over-calling in §4 may be one behaviour —
an agent acting before the user has finished asking. **Tested below**: both moved together on the
clean prompt and accuracy did not follow, so they may share a cause but the cause is not one that
accuracy is downstream of.

Caveat, per arm: the container's agent channel *is* audio, so `asr_chunks` timestamps are real
speech onsets and the analysis is sound. On a research-path arm the same field holds first-*token*
times, which are not comparable to any of the above.

### Signed per-turn latency — `scripts/fdb_v3_signed_latency.py`

Both figures above have a defect: FDB's drops every negative sample, and ours measures from an anchor
the model chose for itself. So this script keeps the sign and measures **every** user turn, pairing
each of the server's end-of-speech markers with the response that follows it and scoring it against
the turn's *acoustic* end. Nothing is discarded; the barge-in rate becomes a reported number instead
of a filter. **Not comparable to any published column** — no published column is computed this way.

| | `nemo_rt` | `nemo_rt_jinja` |
| --- | --- | --- |
| acoustic user turns | 117 | 117 |
| turns answered / unanswered | 100 / 17 | 101 / 16 |
| agent responses | 135 | 136 |
| turns cut into >1 response | 32 | 27 |
| unprompted openings (in n examples) | 28 (17) | 29 (17) |
| signed median / mean | +2.64 s / +1.89 s | +2.72 s / +1.92 s |
| **barge-in rate** (turns / examples) | 25.2 % / 29.6 % | **22.8 % / 23.0 %** |
| barge-in median depth | −5.28 s | −5.36 s |
| clean-response median | +3.12 s | +3.28 s |

Two things this exposes that neither earlier metric could:

* **The server over-segments.** 135 responses against 117 acoustic turns, with 32 turns cut into
  more than one response. That is the mechanism behind the barge-ins: end-of-speech fires at
  intra-turn pauses, so one user turn draws several replies.
* **The agent opens unprompted in 17 of 100 episodes** — a greeting at ~2 s before the user has
  finished, or said anything. This is also the trap in the obvious implementation of this metric:
  pairing each turn with "the first onset after it began" scores that greeting as a 26-second
  barge-in. `pair_onsets` documents the case (`housing_24_69a9cf80f4d7668d5c815038`).

**On the barge-in ↔ over-calling hypothesis:** both eagerness measures did fall together on the
clean prompt — barge-in 29.6 % → 23.0 % of examples, over-calling 1.23x → 1.08x — but Tool Selection
did *not* improve (71.7 % → 70.8 %). So the two behaviours plausibly share a cause, and reducing
them does not by itself buy accuracy. Do not write this up as "less eager, therefore better".
