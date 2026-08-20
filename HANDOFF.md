# Handoff — Nemotron VoiceChat 11B: FDB-v3 reproduction + τ-voice SFT

Written 2026-08-20. Read this top to bottom before touching anything; §1 and §7 are the parts that
will cost you real time if you skip them.

Owner: Kai Li. Two repos, one conda env, a contended Slurm cluster.

---

## 0. The two tasks, in one paragraph each

**Task 1 — reproduce the model card (the near-term deliverable).** `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`
publishes Full-Duplex-Bench v3 numbers: Tool Selection 82.5 %, Argument accuracy 44.2 %, Pass@1
33.0 %. We reproduce them from `/fsx/home/kai.li/code/Full-Duplex-Bench/v3` **through the NIM
container** (the served path), not the eager research path. Status: one full 100-example run
landed, but on the *wrong prompt branch* — see §3. Pass@1 and argument accuracy came out fine;
Tool Selection is 9.4 pts short and that gap is not yet attributable. The faithful arm has not
been run.

**Task 2 — SFT the 11B for τ-voice (the longer plan).** `TAU_VOICE_SFT_PLAN.md` is the live plan
document; it is long, it is measurement-heavy, and **it is the source of truth, not this file**.
Three arms: A = released checkpoint baseline, B = train on the 3 Sierra domains, C = train on 11
new domains and test transfer to the 3 held-out ones. Phase 0 (eval harness) is nearly done.
Arm B is blocked on a training-side token budget; arm A is parked on an ElevenLabs TTS quota.

---

## 1. Critical rules — non-negotiable

These are Kai's standing instructions. `CLAUDE.md` in the repo root is authoritative; this is a
summary, not a replacement.

### 1a. Never release a Slurm allocation

**Do not give up an allocated node unless Kai explicitly asks.** The cluster is contended — on
2026-08-20, 26 of 27 GPU nodes were `alloc` with others already queued on `(Resources)` and
`(Priority)`. A released node does not come back; it costs hours to days.

This covers every way an allocation can end:

* no `scancel` on the **job**
* don't exit or let an `salloc`/shell session end when that drops the job
* don't let a job die by inaction when it could be kept alive
* don't hand a node back "since we're done with it" — being done is not a reason

Killing an inner **step** is fine and is not releasing the node:

```bash
squeue -s -j <jobid>            # list steps running inside the allocation
scancel <jobid>.<step>          # kill one step; the allocation survives
```

### 1b. Getting a node needs no permission — but reuse before you allocate

Don't ask, and don't stall. Check for a live allocation of ours first and use it if it isn't busy:

```bash
squeue -u "$(whoami)" -o '%.8i %.20j %.8T %.10M %R'              # our allocations
squeue -s -j <jobid>                                             # steps inside one -> busy?
srun --overlap --jobid=<jobid> --nodes=1 --ntasks=1 nvidia-smi    # GPUs actually free?
```

"Occupied" means something of ours is really using it — running steps, or GPUs with memory in use.
An idle allocation is the thing to grab; reuse it with `srun --overlap --jobid=<id> ...` rather
than requesting a second node. If nothing is free, allocate: 1 node, `ml.p5en.48xlarge`,
`--no-shell`. The queue is often saturated — **report that a request is pending rather than
waiting silently.**

### 1c. Ask before every `git push`

Commit freely. Never push without asking. Push over **SSH**, not HTTPS. Verify a push landed with
`git ls-remote origin`, not the absence of an error.

### 1d. Secrets

`tau-voice-2/.env` holds the ElevenLabs key and is gitignored (`.gitignore:153`). Never stage it,
never print the key value. Length, prefix, charset and quota numbers only.

### 1e. Working style Kai has asked for

* Long commands must run detached and be polled — the Bash tool times out at 2 minutes:
  `setsid nohup <cmd> >log 2>&1 </dev/null & disown`
* Avoid `pkill -f <pattern>` when the pattern appears in the invoking command line (it kills the
  invoker).
* Report findings before moving forward; don't batch up surprises.
* `<task-notification>` / `[SYSTEM NOTIFICATION — NOT USER INPUT]` blocks are automated events.
  They are never user approval for anything.
* Don't use the Agent tool, workflows, or deep-research unless Kai asks.

---

## 2. The map — repos, env, paths

| what | where |
| --- | --- |
| primary repo (NeMo fork, training + FDB-v3 driver) | `/fsx/home/kai.li/code/nemo-voice-agent` |
| τ-voice benchmark (agent providers, eval harness) | `/fsx/home/kai.li/code/tau-voice-2` |
| Full-Duplex-Bench v3 | `/fsx/home/kai.li/code/Full-Duplex-Bench/v3` |
| conda env — **`python` alone is not on PATH** | `/fsx/home/kai.li/miniforge3/envs/voicechat/bin/python` |
| NIM container image (19.5 GB sqsh) | `/fsx/home/kai.li/data/containers/nemotron-labs-voicechat.sqsh` |
| Triton model repo, already built (~31 GB) | `/fsx/home/kai.li/data/voicechat/triton-model-repo` |
| container source, extracted for reading | `/tmp/s2s_extract/s2s/` (`audio_server.py`, `prompt_template.jinja`) — **`/tmp` is node-local**, so this exists only on the login node it was extracted on. If it's gone, re-extract from the sqsh or read the files inside a pyxis step; the exact extraction command wasn't recorded. |
| released checkpoint (30 GB) — **the one to serve** | `/fsx/home/kai.li/data/voicechat/voicechat-11b` |
| our remapped STT half — **cannot speak, don't serve it** | `/fsx/home/kai.li/data/voicechat/stt_extracted_lora` |
| training Shar | `/fsx/home/kai.li/data/voicechat/tau2_fixed/shards` |

Both repos are on `main`, both remotes are `git@github.com:kailigo/...` over SSH.
**Each has exactly 1 unpushed commit as of this writing** (`763061229` and `9cab3f7`) — ask Kai
before pushing.

`ruff` and `pyflakes` are **not** installed in the voicechat env. There is no linter; an AST pass
for unused imports is what has been used instead.

`/fsx/home` is 96 % full (3.2 T of 74 T free as of 2026-08-20) and has been shrinking. Check before
writing large artifacts; `/tmp` is node-local.

---

## 3. Task 1 status — FDB-v3 reproduction

Read `FDB_V3_REPRODUCTION.md` in full. The short version:

**What landed (2026-08-19):** 100-example run via `scripts/fdb_v3_realtime_infer.py --provider
nemo_rt`, scored by `scripts/fdb_v3_evaluate.py`. 99 completed, 1 deterministic
`inference_error`. Judge is Claude Sonnet 4.5 via Bedrock (no OpenAI key available), patched in at
each evaluator's client seam.

| metric | ours (n=98) | card | judge-mediated? |
| --- | --- | --- | --- |
| Tool Selection | 73.1 % | 82.5 % | **no** — pure multiset F1 over names |
| Argument accuracy | 51.7 % | 44.2 % | yes |
| Pass@1 (n=100) | 35.0 % | 33.0 % | yes |

**Why it needs redoing.** The container was launched on the **default** prompt branch
(`USE_JINJA_TEMPLATE_PROMPT` defaults to `"0"`), which appends 2,158 chars of NVIDIA
tool-restraint text to our instructions. So every session received two directly contradictory
policies concatenated — the benchmark says *"Execute the tool unconditionally!"*, the appended
text says *"DO NOT use any tools when not needed, under no circumstance"*. The model over-called
at 1.23x expected. That number is not attributable until the clean arm runs.

**The next run, and it is ready to go the moment a GPU lands:**

```bash
scripts/fdb_v3_serve.sh <jobid> jinja          # jinja is already the default arg
# confirm the branch from the server log -- the client CANNOT see it:
grep -c 'Preparing prompt using jinja template' <log>   # must be > 0
grep -c 'Call a tool ONLY when the user'        <log>   # must be 0
scripts/fdb_v3_realtime_infer.py --provider nemo_rt_jinja   # distinct name: do not overwrite nemo_rt
scripts/fdb_v3_evaluate.py --provider nemo_rt_jinja
```

Cost: realtime. 78.6 min of benchmark audio ≈ 80 min on one GPU, ~12 min across 8 (one server per
GPU; `scripts/fdb_v3_fanout.sh`). **One GPU per node is the ceiling** — `audio_server.py:48`
hardcodes `TRITON_URL` to `localhost:8000` and pyxis shares the host network namespace, so two
servers on one node collide.

**Not measured yet:** the latency section. `analyze_tool_latency.py` reports `total_samples: 0`
without `user_speech_end_rel`, which needs `scripts/fdb_v3_asr_input.py` (Parakeet over
`input.wav`) run first. Needs a GPU.

---

## 4. Task 2 status — τ-voice SFT

`TAU_VOICE_SFT_PLAN.md` is the plan. Current blockers, in severity order:

1. **Arm A is parked on the ElevenLabs quota** (§0d-ter). Free tier is 10,000 chars/month; 304
   chars remained on 2026-08-19, resetting 2026-09-11. A 24-episode diagnostic needs ~18k chars
   and stage 3 needs ~188k *per arm*. ElevenLabs is used for **TTS only** — verified, two call
   sites, no STT — so a local TTS drops in behind one seam (`synthesis/synthesize.py:22` and
   `data_model/voice.py:327`). `nemo.collections.tts` is already installed. Cost is voice
   identity: a single-speaker local model removes speaker variation as a difficulty axis, a mild
   bias in arm A's favour that must be **stated in results, not hidden**. Kai was obtaining a paid
   key; check with him before building the local path.
2. **Arm B is blocked on `max_fc_total_tokens: 8000`** (§2b). With correct tool schemas, 100 % of
   retail/airline/telecom cuts are **dropped, not truncated**, and silently. Fix is the lossless
   `content` un-double-encoding (677 tokens/cut) **plus** raising the budget to ~12,000. The raise
   needs a GPU to validate memory.
3. **Arm A scores 0.000 for three reasons, not one** (§0d-ter): (a) ASR errors on proper nouns and
   spelled digits — this is what SFT is for; (b) fabricating missing required arguments; (c) **no
   error recovery — verbatim repetition until `max_errors`**, which is what actually sets the
   score. A fix is under test (NVIDIA's FC protocol scaffold, now sent verbatim), with partial
   paired evidence only.

**The zero-cost experiment to run first on resume** (§0d-ter, last paragraph): `both.wav` is
stereo 8 kHz, one channel per speaker, so every persisted episode already contains paid-for user
audio, and `artifacts/*/audio/*_labels.txt` gives the control's exact calls. Replay that channel
with bare-vs-protocol prompt against the real environment. `scripts/check_streaming_driver.py`
has the pieces (`build_session`, `drive()`, `push_tool_result`). 4 episodes × 2 prompts ≈ one
8-GPU wave. Settles (b) and (c); cannot settle (a). **Needs zero ElevenLabs characters.**

---

## 5. What was just built, and its state

`nemo_rt` — a second τ-voice audio-native provider, for the NIM container. Committed as
`tau-voice-2` `9cab3f7`. Documented in `TAU_VOICE_SFT_PLAN.md` §0e.

```
tau-voice-2/src/tau2/voice/audio_native/nemo_rt/
    provider.py               NemotronRealtimeProvider, NemotronVADConfig, _preflight_prompt_length
    discrete_time_adapter.py  DiscreteTimeNemotronRTAdapter
    events.py                 SessionEndEvent, InputAudioTranscriptionDeltaEvent, parse_nemo_rt_event
    __init__.py
```

Four wiring edits: `config.py` (3 registry entries), `voice/audio_native/adapter.py` (factory
branch), `agent/discrete_time_audio_native_agent.py` (Literal + VAD branch), and a 2-line hook in
`openai/provider.py` — `receive_events()` now dispatches through a `_parse_event` class attribute
so a subclass can add event types without reimplementing the receive loop and its error handling.

**Two providers, deliberately.** `nemo` = the in-process research path, registered `cascaded`
(its agent channel is text, so it needs a TTS). `nemo_rt` = the container, registered
`audio_native` (it ships the speech decoder). The provider string also selects the system-prompt
text: `_build_system_prompt` uses `CASCADED_MODEL_INSTRUCTION` for cascaded, else
`AUDIO_NATIVE_VOICE_INSTRUCTION`.

**Verification state: offline only. It has never talked to a live container.** What *is* verified:
factory returns the right adapter and model; `bytes_per_tick=1600`, `_chunk_size=640`, converter
16000→24000; the exact `session.update` payload against a `FakeWS`; input-rate rejection at
construction; `truncate_item` sends 0 messages; `close_session` returns the `session.end` stats;
`parse_nemo_rt_event` on all the new and delegated types; `_preflight_prompt_length` byte-exact
against the container's real templates on all 15 loadable domains, both branches.

**First live step** is a smoke episode once a container is up. Kai was offered, and has not yet
answered, the alternative of a fake WebSocket server speaking the real protocol so the
integration can be exercised on CPU before GPU time is spent. That offer is still open and is
probably the best use of a pending queue.

---

## 6. Next steps, in the order they unblock

| # | step | needs | notes |
| --- | --- | --- | --- |
| 1 | Fake-container smoke test of `nemo_rt` (connect → configure → ticks → tool call → barge-in → close) | CPU | offered to Kai, awaiting a yes; de-risks step 3 |
| 2 | `nemo_rt_jinja` FDB-v3 arm + evaluate | 1 GPU, ~80 min | §3; the actual reproduction deliverable |
| 3 | First live `nemo_rt` τ-voice episode | container | validates step 1's assumptions |
| 4 | `fdb_v3_asr_input.py` + latency section | GPU (Parakeet) | closes the last unmeasured FDB-v3 metric |
| 5 | Zero-character bare-vs-protocol A/B | 8 GPUs, ~15 min each | §4; settles arm A failures (b) and (c) |
| 6 | Arm B token budget: lossless fix + raise to 12k, re-validate memory | GPU | unblocks arm B |
| 7 | Local-TTS decision | Kai | gates all of arm A |

Also queued, low priority: drop the two dead heads (~4.7 GB fp32) on the next checkpoint extract.

Jobs `1349` and `1350` were `PENDING (Priority)` at handoff — 1 node each, `ml.p5en.48xlarge`.

---

## 7. Traps — every one of these has already bitten, or was caught one step short of biting

**Container protocol** (authority: `/tmp/s2s_extract/s2s/audio_server.py` — read it, don't guess):

* **`USE_JINJA_TEMPLATE_PROMPT` defaults to `"0"`. Always launch with `=1`.** Two independent
  reasons: the restraint-text contradiction (§3), and **τ-voice telecom renders to 32,622 chars
  against the 32,000-char `MAX_INSTRUCTIONS_LENGTH`**, at which point the server `_emit_error`s
  and returns **before dispatching prompt or tools and before sending `session.updated`** — so the
  handshake looks healthy and the model runs promptless and tool-less. telecom-workflow clears the
  default branch by only 532 chars. The limit is measured on the **fully rendered** prompt, not on
  what you sent.
* **An out-of-range input rate silently discards the entire session config.**
  `handle_session_update` validates the rate and `return`s at `:1527` *before* reading `tools` or
  `instructions`. Valid range 16000–48000. Symptom: an agent that chats pleasantly and never calls
  anything, plus `tools_not_set` on every tool result.
* **The prompt is only dispatched before the first audio chunk** (`:1574`, gated on
  `sequence_started`). A later `session.update` is ACKed with `session.updated` and dropped.
* **Only four client message types are accepted** (`:1770-1830`): `input_audio_buffer.append`,
  `session.update`, `session.close`, `conversation.item.create` (`function_call_output` only).
  Everything else — including `conversation.item.truncate` and `response.create` — hits
  `logger.warning("Ignored message type")`.
* **`speech_started` / `speech_stopped` have `audio_start_ms` / `audio_end_ms` hardcoded to `0`**
  (`:905`, `:920`, labelled "Pipecat/OpenAI Realtime compatibility"). Feeding that into
  `TickResult.truncate_agent_audio` yields `max_bytes = 0`, so **every barge-in replaces the whole
  in-flight utterance with silence** — measured: 1600 bytes returned, **0 audible** — and reports
  nothing, because `discarded = received - len(played)` is `1600 - 1600` after silence-padding.
  `nemo_rt` therefore does *not* call `truncate_agent_audio`; `nemo` should, since there the offset
  is real. The plan's §0a bullet says otherwise and §0e corrects it.
* **`response.done`'s usage block is hardcoded zeros** (`:1015`). Don't record a `UsageRecord`;
  zero reads as "measured: free".
* **Output is always 24 kHz PCM16 in 80 ms frames.** The negotiation branch at `:1550` compares
  formats (both `"pcm16"`), not rates, so it is unreachable. The echoed `session.updated` audio
  config is cosmetic — `_format_to_structured` hardcodes 24000 both ways.
* **Client-side WebSocket keepalive must be off** (`ping_interval=None`). The server is
  uvicorn and does answer pings, but the client's default 20 s pong timeout fires while the server
  is inside a Triton call.
* **`chunks_dropped` is not a tail trim.** Each queued item is 160 ms
  (`triton_chunk_size × NUM_CHUNKS_PER_INFERENCE` = 1280×2 @16 kHz); on `QueueFull` the server
  **evicts the oldest** (`:698-701`), punching a hole mid-stream. Capacity ~16 s. The server's own
  "1.6 seconds queue buffer" comment at `:100` is stale by 10×.
* **Both prompt branches strip only `type` and `ack_messages`** and pass every other key into the
  prompt verbatim. So pydantic-generated `title` keys *do* reach the model here, unlike with
  hosted providers — leave them, so every provider gets an identical tool block.
* **`response.function_call_arguments.done` always sends `arguments` as a JSON string**
  (`json.dumps` at `:1459`). Decode it. Watch for double-encoding: the prompt template advertises
  `{"name": ..., "arguments": "tool_args1"}`, so a literal-minded model produces a string inside a
  string, which decodes to `str` not `dict`.
* **One server per node.** `TRITON_URL` is hardcoded to `localhost:8000` (`:48`) and pyxis shares
  the host network namespace.
* **Append 10–20 s of trailing silence** to any input stream, or the reply is truncated — the
  model only emits while input flows.

**Infrastructure / process:**

* **Bare `torchrun` on the login node always dies — there is no GPU there.** Check `nvidia-smi`
  first; use `scripts/run_on_gpu.sh` inside an allocation.
* **Concurrent episodes need distinct `MASTER_PORT`** (fixed in `068db5f81`); one process per
  episode. **`exit 0` lies** — only the fan-out's post-exit verdict in `progress.log` is
  authoritative.
* **A mid-episode ElevenLabs 429 is not a dead episode.** `runner/progress.py::run_with_retry`
  re-runs the whole unit and episodes have continued for 28 min afterwards. A watchdog that greps
  for `emergency cleanup` will requeue live work onto fresh GPUs.
* **Two ElevenLabs limits, different fixes:** concurrent requests (2 on free, transient,
  self-healing) vs characters/month (10,000, terminal). Distinguish by the error body.
* **`nemo.collections.tts` needs `LD_LIBRARY_PATH=$ENV_PREFIX/lib`** or `_sqlite3` fails on
  `CXXABI_1.3.15`.
* **Inside the container, unset `PYTHONPATH LD_LIBRARY_PATH CONDA_PREFIX`** or our conda env leaks
  in.
* **The research path has no KV cache** and cannot get one from where we are
  (`duplex_stt_model.py:3718-3722` hard-sets `cache=None`). The "12.2× slower than realtime"
  figure is that path's, **not the model's** — the container is realtime at wall/audio 1.001.
  Don't attribute it to the model.
* **τ-voice audio is 8 kHz telephony** on a **200 ms tick grid**; alignment is exact. Everything
  `TickResult` accounts for is in telephony bytes — convert *before* counting, or 24 kHz bytes
  overstate durations 6×.

---

## 8. Cheat sheet

```bash
PY=/fsx/home/kai.li/miniforge3/envs/voicechat/bin/python

# Slurm
squeue -u "$(whoami)" -o '%.8i %.20j %.8T %.10M %R'
squeue -s -j <jobid>
srun --overlap --jobid=<jobid> --nodes=1 --ntasks=1 nvidia-smi
scancel <jobid>.<step>        # a STEP. never the bare jobid.

# Serve the container (jinja is the default and the only correct choice)
scripts/fdb_v3_serve.sh <jobid> jinja [port]
curl http://<node>:9000/v1/realtime/health

# FDB-v3
scripts/fdb_v3_realtime_infer.py --provider nemo_rt_jinja
scripts/fdb_v3_evaluate.py      --provider nemo_rt_jinja
scripts/fdb_v3_fanout.sh                      # 8 GPUs, one server each
scripts/fdb_v3_retry_failures.sh

# τ-voice
scripts/tau2_smoke_nemo.sh
scripts/tau2_stage2_subset.sh                 # STAGGER_SECONDS to soften TTS concurrency
scripts/tau2_quick_report.py                  # untracked; quick reward/error table
scripts/tau2_watch_run.sh                     # untracked; poll a live fan-out
scripts/check_streaming_driver.py             # research-path driver; build_session/drive/push_tool_result

# Measurement helpers
scripts/measure_fc_token_budget.py --shards ... --tool_schemas ...
tau-voice-2/scripts/export_tool_schemas.py --output data/tool_schemas.json

# Long jobs (Bash tool times out at 2 min)
setsid nohup <cmd> >/tmp/x.log 2>&1 </dev/null & disown
```

Untracked working scripts at handoff: `scripts/tau2_quick_report.py`,
`scripts/tau2_watch_run.sh`. Modified and uncommitted: `scripts/tau2_smoke_nemo.sh`,
`scripts/tau2_stage2_subset.sh` (retry-constant and stagger changes from the 0d-ter work).

---

## 9. Things that are NOT true, and must not be claimed

* **There is no post-fix τ-voice reward number.** No episode persisted before the ElevenLabs quota
  ran out. The FC-protocol fix has *partial paired tick-depth evidence only*. Do not cite a reward.
* **`nemo_rt` has not run against a live container.** All its verification is offline/CPU.
* **The FDB-v3 Tool Selection gap is not yet attributable.** It was measured on a prompt carrying
  two contradictory tool-use policies. Don't call it a model property until `nemo_rt_jinja` runs.
* **Argument accuracy beating the card is not a result.** The judge is Sonnet 4.5, not gpt-4o, and
  that metric is judge-mediated. Tool Selection is the only one of the three that uses no judge.
* **"The checkpoint invents tool names" is a symptom, not the disease.** Tool *selection* is
  substantially right on the first attempt; the invented names appear only as the
  repeat-until-`max_errors` loop degenerates. The 27 % invented-name rate is the loop's output.
* **`--system-message nvidia+benchmark` is not a control arm.** The restraint text is already
  present on the default branch, so the flag duplicates it.
* **`stt_extracted_lora` cannot speak** — it is missing all 635 `tts_model.*` tensors. Build the
  Triton repo from the *released* checkpoint.

---

## 10. Open decisions for Kai

1. Push the two unpushed commits? (`763061229`, `9cab3f7`)
2. Local TTS behind `synthesize.py:22`, or wait for a paid ElevenLabs key? Gates all of arm A.
3. Build the fake-container WebSocket server to smoke-test `nemo_rt` on CPU while the queue is
   pending?
4. From §9 of the plan, still open: whether τ-voice reports one headline score or separate voice
   metrics; whether the 11 new domains' policies are comparable in depth to the Sierra three;
   whether telecom's 2,285 tasks are the real eval set; whether arm B holds out tasks or trials.

---

## 11. Reading order for a new agent

1. `CLAUDE.md` — the rules, verbatim.
2. This file, §1 and §7.
3. `FDB_V3_REPRODUCTION.md` §3 (deviations) and §4 (results).
4. `TAU_VOICE_SFT_PLAN.md` §2b, §0d-ter, §0e — the three sections carrying live blockers.
5. `/tmp/s2s_extract/s2s/audio_server.py` — before writing any container client code. Every claim
   in §7's first block is a line reference into it; verify rather than trust.
6. `tau-voice-2/src/tau2/voice/audio_native/nemo_rt/provider.py` — its module docstring is the
   design rationale for the whole package.
