# Reproducing the Full-Duplex-Bench v3 numbers on the model card

The [NVIDIA-NemotronLabs-VoiceChat-11B card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
reports three FDB-v3 numbers:

| metric | card |
| --- | --- |
| Tool Selection | **82.5 %** |
| Argument accuracy | **42.2 %** |
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

So the transport is not load-bearing for any published metric, and LiveKit is replaceable by
a local replay. Which is necessary anyway: our model is 11B of weights in-process, and with
no KV cache (Nemotron-Nano is hybrid Mamba2, so it re-reads full history each step) inference
runs at ~12x slower than realtime — it cannot hold a realtime WebRTC session open.

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
scripts/fdb_v3_tools.py        # the 12 tools + agent instructions, parsed out of lk_agent_tool.py
scripts/fdb_v3_nemo_infer.py   # local replay -> result_{provider}.json
scripts/fdb_v3_fanout.sh       # 8 shards, one persistent worker per GPU
scripts/fdb_v3_asr_input.py    # Parakeet over input.wav -> user_speech_end_rel (latency only)
scripts/fdb_v3_evaluate.py     # the benchmark's own evaluators, Bedrock judge patched in
```

```bash
scripts/fdb_v3_fanout.sh 1303 --provider nemo          # ~2 h wall on 8 GPUs
python scripts/fdb_v3_asr_input.py --provider nemo     # after, for the latency section
python scripts/fdb_v3_evaluate.py --provider nemo
```

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
* Audio plus 1.5 s of trailing silence, matching `livekit_inference.py:270-282`. The WAVs
  already contain the response gap (median ~47 s of file for ~10 s of speech).

**Deviations, each stated in every result file's `notes` block:**

1. **No LiveKit.** Local replay at 16 kHz into `DuplexSTTModel`, 80 ms frames.
2. **`transcript` is the model's text channel, not ASR of synthesised speech.** This
   checkpoint returns `tokens_audio: None` — the agent channel is text. So
   `audio_agent_speech_start` is *first-token* time, not first-speech time. Tool selection
   and argument accuracy do not depend on this; latency and response quality do, and are not
   comparable to the published table.
3. **Audio-clock timestamps**, not wall clock. Wall clock would measure our GPU (12x slower
   than realtime), not the model's turn-taking.
4. **The judge is Claude Sonnet 4.5 via Bedrock, not gpt-4o.** We have no OpenAI key;
   Bedrock authenticates off the instance IAM role. The metric definition is unchanged and
   the prompts are the benchmark's own, but a stricter or looser judge moves argument
   accuracy and Pass@1 directly. Tool Selection is unaffected — it never calls the judge.
   The substitution is printed at the top of every run.

The judge is patched in at the one seam each evaluator uses to get a client
(`_get_openai_client`, or the module-level `OpenAI` in `analyze_tool_latency.py`), so no
proxy server and no edit to the benchmark is needed. `_patch` raises if neither seam is
present rather than leaving the judge pointed at OpenAI.

## 4. Results

Probe (1 example, `ecommerce_01`, 2026-08-19) — the harness end to end before committing GPU
hours to it:

```
status=completed rtf=12.18 expected=['track_order'] got=['track_order'] rejected=0
transcript: "Your order LHR has been received and is now out for delivery."
```

Tool selection correct; the argument is a hallucination. The user spells "A-B-C-1-2-3" out
loud (`acting_notes`: "Say the order ID clearly, one character at a time"), expected
`order_id: "ABC123"`, the model emitted `order_id: "LHR"` — an airport code, from a different
domain's tool. That single example matches the *shape* of the published result: high tool
selection, argument accuracy roughly half of it.

Cost, measured: 12.18x realtime, 577 s for a 47 s example, 97 GB on an H200 at 84 %
utilisation (so one worker per GPU, not two). 78.6 min of benchmark audio ≈ 16 GPU-hours ≈
2 h wall on 8 GPUs.

Full 100-example run: **in progress** (launched 2026-08-19 20:39 UTC on allocation 1303).
Numbers go here when it lands — nothing is quoted before then.
