# Overnight working plan — night of 2026-08-20

Companion to `HANDOFF.md`. That file is context; this one is the shift schedule. Written against
the allocation that exists right now:

```
JOBID 1374   RUNNING 10:01   1 node   ip-10-1-30-86   step 1374.0 = the allocation's own bash
8x NVIDIA H200, 143771 MiB each, ALL IDLE (0 MiB used, 0 % util)
```

**The allocation is the scarce resource, not the night.** Rule 1a in `HANDOFF.md` applies without
exception: no `scancel 1374`, no letting it drop, no handing it back at dawn because the work
finished. Kill inner steps freely (`scancel 1374.<step>`).

---

## 0. What this night is for

Three things are true at once and they set the whole plan:

1. **The headline deliverable is one 80-minute GPU job that has never been run.** The FDB-v3
   numbers we have came from a container on the wrong prompt branch. `nemo_rt_jinja` is the
   faithful arm. It needs 1 GPU and no supervision.
2. **The container can only use one GPU on this node.** `audio_server.py:48` hardcodes
   `TRITON_URL` to `localhost:8000` and pyxis shares the host network namespace, so two servers
   collide. `fdb_v3_serve.sh` pins `CUDA_VISIBLE_DEVICES=0`. So GPU 0 is the container's for the
   night and **GPUs 1–7 are free for research-path work** — that is 7 idle H200s if we don't plan
   for them.
3. **Anything that runs the τ-voice user simulator is blocked, not slow.** ElevenLabs is at 304
   characters of a 10,000/month free tier, resetting 2026-09-11. ~4 user turns total. So every
   task tonight must be **zero-character**: replay pre-recorded audio, never synthesise. This is
   the single constraint that shapes track B and rules out the obvious "just run a τ-voice
   episode" smoke test.

Two tracks run in parallel. Track A owns GPU 0 and the container. Track B owns GPUs 1–7 and the
research path. They share nothing and neither blocks the other.

---

## 1. Resource map

| GPU | owner | all night |
| --- | --- | --- |
| 0 | NIM container (`fdb_v3_serve.sh`, jinja branch) | A1 → A2 → A4 → A5 |
| 1 | Parakeet ASR pass, then a replay shard | B1 → B3 |
| 2–7 | replay shards | B3 |
| all 8 | arm-B memory validation — **only if the container is down** | B4, stretch |

Ports: container on **9000**. Research-path workers need distinct `MASTER_PORT` (base 29700 +
shard) or they die `EADDRINUSE`, which reads like a networking fault. `fdb_v3_fanout.sh` already
does this; any hand-rolled fan-out must too.

Environment, every time:

```bash
PY=/fsx/home/kai.li/miniforge3/envs/voicechat/bin/python
export LD_LIBRARY_PATH=/fsx/home/kai.li/miniforge3/envs/voicechat/lib:${LD_LIBRARY_PATH:-}
JOB=1374
NODE=ip-10-1-30-86
```

Disk: `/fsx/home` is at 96 %, 3.2 T free. A 100-example FDB run writes one agent wav per example;
the existing `nemo_rt` run is the size precedent, so budget the same again. If space gets tight,
`fdb_v3_realtime_infer.py --no-save-audio` exists — but the audio is what makes a bad number
diagnosable, so drop it only under pressure.

---

## 2. Track A — the container, GPU 0

### A1. Serve on the jinja branch, and prove the branch (~10 min) — **GATE**

```bash
mkdir -p logs/fdb_v3
setsid nohup scripts/fdb_v3_serve.sh $JOB jinja 9000 \
  >logs/fdb_v3/serve_jinja.log 2>&1 </dev/null & disown
# ~6 min to ready: Triton load, vLLM engine ~140 s, warmup
curl -s http://$NODE:9000/v1/realtime/health
```

**Do not proceed on the health check alone.** The client cannot see which prompt branch the
server took, and the whole point of tonight's arm is the branch:

```bash
grep -c 'Preparing prompt using jinja template' logs/fdb_v3/serve_jinja.log   # must be > 0
grep -c 'Call a tool ONLY when the user'        logs/fdb_v3/serve_jinja.log   # must be 0
```

The first grep only fires once a session has sent a prompt, so it may read 0 until A2 starts —
check it again after the first example, and **abort A2 if it is still 0 or if the second grep is
non-zero.** A run on the wrong branch is worse than no run: it looks like data.

*Failure playbook.* If the server never reaches ready: read the tail of the log for the Triton
model-repo path (`/data/models` must contain `nemotron-voicechat`), confirm the mount, confirm
`PYTHONPATH LD_LIBRARY_PATH CONDA_PREFIX` are unset inside (the script does this). Two attempts,
then stop, write down the failure, and give the night to track B. Do not burn hours here.

### A2. The `nemo_rt_jinja` FDB-v3 arm (~85 min, unattended) — **the deliverable**

```bash
setsid nohup $PY -u scripts/fdb_v3_realtime_infer.py \
  --server ws://$NODE:9000 \
  --provider nemo_rt_jinja \
  --server-prompt-mode jinja \
  >logs/fdb_v3/infer_nemo_rt_jinja.log 2>&1 </dev/null & disown
```

`--provider nemo_rt_jinja` is load-bearing: results are written as
`result_<provider>.json` inside each of the 100 example dirs, so reusing `nemo_rt` would
**overwrite the default-branch run we still need for the comparison.** `--server-prompt-mode
jinja` stamps the branch into every result file so a future reader cannot mistake the arm.

Runs at realtime — 78.6 min of benchmark audio, single session, sequential. Poll:

```bash
ls /fsx/home/kai.li/code/Full-Duplex-Bench/v3/fdb_v3_data_released/*/result_nemo_rt_jinja.json | wc -l
```

Expect ~1 result/50 s. **Coverage is the verdict, not the exit code** — the evaluators drop absent
scenarios from the denominator, so a shard that dies silently inflates the score. One known
deterministic failure is expected: `ecommerce_20_66c4f3cb14cbfc4db836bd4e` dies the same way on
every attempt (Triton's degenerate-tool-call recovery: `Fast extract: exceeded 512 steps without
eotc_id` → `append_request: request not found` → HTTP 500 → WS 1011). Leave it; score it as a
no-response. If *other* examples fail, `scripts/fdb_v3_retry_failures.sh` re-runs just those.

### A3. Score it, and say what it means (~15 min)

```bash
$PY scripts/fdb_v3_evaluate.py --provider nemo_rt_jinja 2>&1 | tee logs/fdb_v3/eval_nemo_rt_jinja.log
```

Judge is Claude Sonnet 4.5 via Bedrock (274 calls, 0 failures last time). Then build the
three-way table — card / `nemo_rt` (default branch) / `nemo_rt_jinja` — and answer the one
question the night exists to answer:

> Tool Selection was 73.1 % against the card's 82.5 %, measured on a prompt carrying two
> contradictory tool-use policies, with the model over-calling at 1.23x expected. Does removing
> the contradiction close the gap?

Three outcomes, all publishable, and write down which one happened **before** reaching for a next
hypothesis:

* **Closes it** → the 9.4-pt gap was our launch flag. The reproduction is done.
* **Doesn't move** → the gap is real and internal to our setup; over-calling is not
  prompt-induced. Next hypothesis needed, and the per-cut breakdown in
  `logs/fdb_v3/nemo_rt_jinja_evaluation_report.json` (precision-only vs recall-only losses, calls
  emitted vs expected) is where to look.
* **Gets worse** → the restraint text was *helping*, i.e. the card's numbers may rely on
  restraint the benchmark prompt doesn't supply. That is a finding about the card, and the most
  interesting of the three.

Argument accuracy is judge-mediated (Sonnet, not gpt-4o) — **do not claim beating the card on it.**
Tool Selection is the only judge-free metric of the three.

### A4. First **live** `nemo_rt` test — with zero ElevenLabs characters (~1 h to write, ~10 min to run)

The `nemo_rt` provider committed today (`tau-voice-2 9cab3f7`) has never spoken to a real
container. Everything about it is offline-verified. This is the highest-value de-risking available
tonight, and it must not go through the τ-voice user simulator (that needs TTS, and there are 304
characters left).

So: drive the adapter directly with **pre-recorded** audio. Write
`tau-voice-2/scripts/nemo_rt_live_check.py` that:

1. loads a real domain's prompt and tools — start with `mock` (3,186 chars rendered, enormous
   headroom) then `retail` (20,383);
2. `create_adapter("nemo_rt", tick_duration_ms=200, send_audio_instant=False)`, `connect(...)`;
3. feeds 8 kHz μ-law telephony ticks from an existing wav — the user channel of
   `data/simulations/stage2_subset/retail__92/.../audio/both.wav` (stereo, one channel per
   speaker) is real τ-voice user audio and already paid for;
4. asserts, per tick: transcripts land under an `item_id`, `agent_audio_chunks` are non-empty and
   in **telephony** bytes, tool calls decode to `dict`, and `session.end` stats come back with
   `chunks_dropped == 0`.

Write this while A2 is running — it is pure CPU. What it is actually checking, i.e. the four
things most likely to be wrong in code that has never met its server:

| claim to falsify | how it fails if wrong |
| --- | --- |
| `session.update` is accepted as sent | server logs a validation error; no `session.updated` |
| the prompt reaches the model before the sequence starts | the model answers generically and calls nothing |
| `arguments` decodes to a `dict` | our `logger.error` for decode-to-non-dict fires |
| audio accounting is in telephony bytes | durations off by exactly 6x |

If `chunks_dropped > 0`, that is a real finding, not noise: each dropped item is 160 ms and the
server **evicts the oldest**, punching a hole mid-stream rather than trimming the tail.

### A5. Concurrency probe — only after A2 has finished (~20 min, optional)

Tonight's arm is sequential and realtime-bound, which is why it costs 80 min. If one server can
hold 2+ concurrent sessions without degrading `wall/audio`, every future FDB run gets cheaper:

```bash
# two shards against the SAME server, into a throwaway provider name
$PY -u scripts/fdb_v3_realtime_infer.py --server ws://$NODE:9000 \
    --provider concurrency_probe --shard 0 --num-shards 2 --limit 4 &
$PY -u scripts/fdb_v3_realtime_infer.py --server ws://$NODE:9000 \
    --provider concurrency_probe --shard 1 --num-shards 2 --limit 4 &
```

Metric: `wall/audio` and `chunks_dropped` per example versus the 1.001 / 0 baseline. **Run this
only after A2 is complete** — a contended server during A2 would silently degrade the arm. Use a
throwaway `--provider` name so the probe's result files can be deleted.

---

## 3. Track B — research path, GPUs 1–7

Independent of track A. Start B1 immediately; it has the earliest-surfacing risk of anything
tonight.

### B1. Parakeet ASR pass over the input audio (~20 min) — **do this first, it may fail on download**

`analyze_tool_latency.py` reports `total_samples: 0` without `user_speech_end_rel`, so the latency
section of every FDB-v3 report is empty. `fdb_v3_asr_input.py` fills it in one pass on one GPU,
over results that already exist.

```bash
srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 bash -c "
  export CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/fsx/home/kai.li/miniforge3/envs/voicechat/lib
  $PY scripts/fdb_v3_asr_input.py --provider nemo_rt --dry-run"
```

**Known risk, unverified: `nvidia/parakeet-tdt-0.6b-v2` is NOT in `~/.cache/huggingface`.** The
first run will try to download it. Compute nodes have reached `nvcr.io`, so outbound probably
works, but it has not been confirmed for HuggingFace from a node. If the download fails, fetch it
from the login node into the shared HF cache and re-run — do not spend the night on it.

Then the real pass for both arms (the second one after A2 lands):

```bash
$PY scripts/fdb_v3_asr_input.py --provider nemo_rt
$PY scripts/fdb_v3_asr_input.py --provider nemo_rt_jinja
```

**Caveat to carry into the report, unfixable by better ASR:** our agent-side timestamp is *first
text token*, not first audio sample, because the research path's agent channel is text. So this
unlocks "time to first token on the audio clock", **not** the card's "time to first speech".
Directionally useful, not comparable. For the container arm the agent channel *is* audio, so
`nemo_rt_jinja` latency is the comparable one — say which is which in the report.

### B2. Re-score with latency (~10 min)

Re-run `fdb_v3_evaluate.py` for both providers once B1 has written `user_speech_end_rel`, and
report the latency section for the first time.

### B3. The zero-character bare-vs-protocol A/B (~2 h to write, ~50 min to run) — **the big one**

This settles two of the three reasons arm A scores 0.000, and it costs **no ElevenLabs
characters**. Verified available tonight — 4 real user-audio channels on disk:

| both.wav | run |
| --- | --- |
| `stage2_subset/retail__21/artifacts/task_21/sim_6b01a144-.../audio/both.wav` | control |
| `stage2_subset/retail__78/artifacts/task_78/sim_065afc64-.../audio/both.wav` | control |
| `stage2_subset/retail__92/artifacts/task_92/sim_00043517-.../audio/both.wav` | control |
| `stage2_protocol/retail__49/artifacts/task_49/sim_d6b9e8d7-.../audio/both.wav` | protocol |

Each is stereo 8 kHz, one channel per speaker, so the user channel is real paid-for audio, and the
sibling `audio/*_labels.txt` files give the control's exact tool calls with timestamps.

The design is a paired deterministic A/B where the **only** variable is the prompt: replay the
user channel into the model with the bare vs FC-protocol system prompt, executing tool calls
against the real environment (`registry.get_env_constructor("retail")()` → `make_tool_call`, which
reproduces `Error: Order not found` exactly).

4 audio channels x 2 prompts = 8 runs, one per GPU on 1–7 (one runs second). At the research
path's 12.2x realtime a ~200 s episode is ~40 min, so ~50 min wall.

**The script does not exist yet — this is the night's main writing task.**
`scripts/check_streaming_driver.py` has the three pieces (`build_session`, `drive()` with 200 ms
ticks and `<SOTC>` detection, `push_tool_result`) but it reads cuts from a Lhotse Shar, not a
`both.wav`. New code needed: wav-channel loading, prompt selection via
`NeMoDuplexConfig.fc_prompt_protocol` (default on) vs the `nemo-base-bare-prompt` preset, and the
live environment call loop.

Metrics — pick these before running, not after:

* **consecutive identical tool calls** (the disease: 10 identical calls 3.2 s apart, *after* the
  user's audio had ended)
* **does the model speak after the first error**, instead of retrying
* tool names emitted, and whether invented names appear at all

This settles failure (b) *fabricates a missing required argument* and (c) *no error recovery*. It
**cannot** settle (a) *ASR errors on proper nouns and spelled digits* — that is what SFT is for.
Do not let a good (b)/(c) result get written up as "arm A is fixed".

### B4. Arm-B token budget — stretch, and it conflicts with track A

§2b of the plan: apply the lossless `content` un-double-encoding (677 tokens/cut) and re-measure,
then validate that `max_fc_total_tokens: 12000` still fits in memory. The re-measure is CPU and
can happen any time:

```bash
$PY scripts/measure_fc_token_budget.py \
  --shards /fsx/home/kai.li/data/voicechat/tau2_fixed/shards \
  --tool_schemas /fsx/home/kai.li/code/tau-voice-2/data/tool_schemas.json
```

The **memory validation needs all 8 GPUs**, which means the container must come down first. Do not
take GPU 0 from track A for this. Only start it if track A is fully finished and there is >2 h
left, and note that bringing the container back costs ~6 min.

---

## 4. Timeline

Two tracks, wall-clock from start. CPU writing is deliberately scheduled against GPU waits.

| time | GPU 0 (track A) | GPUs 1–7 (track B) | CPU |
| --- | --- | --- | --- |
| 0:00 | A1 serve, ~6 min to ready | B1 Parakeet dry-run → real pass | — |
| 0:15 | **GATE**: branch greps | B1 running | — |
| 0:20 | A2 starts (~85 min) | B1 done / debugging download | write A4 live-check script |
| 1:30 | A2 running | idle | write B3 replay script |
| 1:45 | **A2 done** → A3 evaluate | B2 re-score `nemo_rt` w/ latency | — |
| 2:00 | A3 done → three-way table | B1 pass for `nemo_rt_jinja` | — |
| 2:15 | A4 live `nemo_rt` check | B3 launches, 8 runs on 7 GPUs | — |
| 3:05 | A5 concurrency probe | B3 running | — |
| 3:30 | container idle, held | **B3 done** → analyse | write up |
| 4:00+ | B4 stretch (needs GPU 0 released from the container, allocation stays) | | |

Under 4 hours of critical path. The night has room for all of it plus the two scripts, which is
why B4 is a stretch rather than a filler.

---

## 5. Hard rules for the night

1. **Never `scancel 1374`.** Never let it drop. Not at the end, not "since we're done".
   `scancel 1374.<step>` on inner steps is fine and is not releasing the node.
2. **Never `git push`.** There are 4 unpushed commits (3 in `nemo-voice-agent`, 1 in
   `tau-voice-2`); they stay unpushed until Kai says otherwise. Commit freely.
3. **Zero ElevenLabs characters.** 304 remain. Nothing tonight may call
   `text_to_speech.convert`. That rules out any τ-voice episode with a live user simulator —
   replay only. Never print the key; length/prefix/quota numbers only.
4. **Never reuse a `--provider` name.** Results are `result_<provider>.json` per example dir;
   a collision silently overwrites an arm we need for comparison.
5. **Coverage before score, every time.** `exit 0` lies and the evaluators drop absent scenarios
   from the denominator. Count result files on disk.
6. **Detach anything over 2 minutes** (`setsid nohup … </dev/null & disown`) and poll. Avoid
   `pkill -f <pattern>` where the pattern matches the invoking command line.
7. **Two attempts, then move on.** Any step that fails twice gets written down and the night moves
   to the next item. There is more queued work than night.
8. **Don't claim what wasn't measured.** See §9 of `HANDOFF.md`. In particular: no post-fix
   τ-voice reward number exists, and argument accuracy beating the card is not a result.

---

## 6. Morning report — have these ready

1. **The three-way FDB-v3 table** (card / `nemo_rt` default branch / `nemo_rt_jinja`) and one
   sentence on which of A3's three outcomes happened.
2. **Branch proof for the new arm** — the two grep counts, quoted.
3. **Coverage** for `nemo_rt_jinja`: n completed of 100, and whether the one known deterministic
   failure was the only one.
4. **Latency section**, for the first time, with the text-vs-audio agent-channel caveat stated per
   arm.
5. **Live `nemo_rt` verdict**: does the provider work against a real container, and which of A4's
   four falsifiable claims survived.
6. **B3's answer**: consecutive-identical-call counts, bare vs protocol, per episode — and an
   explicit note that it settles (b) and (c) only.
7. **What was skipped and why.**

---

## 6b. RESULTS — live log, 2026-08-21

Written as the work lands. **Timing note up front:** this plan was written ~07:20 on 2026-08-21, so
what follows is a morning run of a few hours, not a whole night. The timeline in §4 should be read
as relative offsets from 07:20, not as overnight hours.

Status board at 08:10:

| step | state | where |
| --- | --- | --- |
| A1 serve on jinja branch | **done, GATE PASSED** | `logs/fdb_v3/serve_jinja.log` |
| A2 `nemo_rt_jinja` arm | **done, 100/100, 0 failures** | `logs/fdb_v3/infer_nemo_rt_jinja.log` |
| A3 score the new arm | **done — hypothesis FALSIFIED** | `logs/fdb_v3/eval_nemo_rt_jinja.log` |
| B1b ASR pass, jinja arm | **done, 100/100** | `logs/fdb_v3/asr_nemo_rt_jinja.log` |
| signed per-turn latency | **done** | `scripts/fdb_v3_signed_latency.py` |
| A4 live `nemo_rt` check | **done — works on mock, blocked on retail** | `tau-voice-2/logs/nemo_rt_live_check_*.json` |
| A4b prompt-budget sweep | **done, 15 domains** | `tau-voice-2/logs/nemo_rt_prompt_budget.json` |
| A5 concurrency probe | not started | — |
| B1 Parakeet ASR pass | **done, 100/100, 0 failures** | `logs/fdb_v3/asr_nemo_rt.log` |
| B2 re-score with latency | **done** | `logs/fdb_v3/eval_nemo_rt_withlatency.log` |
| B3 bare-vs-protocol A/B | not started | — |
| B4 token budget | not started | — |

### A1 — gate passed

Container reached ready in ~9 min (two vLLM EngineCore loads, pids 3895513/3897517, no errors).
Health: `{"status":"ok", "mode":"triton", "triton_status":"ready",
"model_inference_stats":{"success_count":241,"fail_count":0}}`. Serving at `ws://ip-10-1-30-86:9000`.

The branch proof, which is the thing that actually gates the arm (counts as of 08:10, after 30
sessions):

```
grep -c 'Preparing prompt using jinja template'  logs/fdb_v3/serve_jinja.log  ->  31   (must be >0)  ✓
grep -c 'Call a tool ONLY when the user'         logs/fdb_v3/serve_jinja.log  ->   0   (must be  0)  ✓
```

So `USE_JINJA_TEMPLATE_PROMPT=1` took effect and **no restraint text reached the model** — the
opposite of the 100-session default-branch run in `FDB_V3_REPRODUCTION.md` §3 deviation 2. This is
the first faithful-prompt arm.

### A2 — done, 100/100, and slower than planned

Clean so far: every example `status=completed`, `rtf` 1.002–1.007, no server errors, no rejects.
Per-turn latency in the log is 0.48–0.8 s, matching the default-branch arm.

**Pace correction: ~1.7 min/example, so ~2 h 50 m total, not the planned 85 min.** The plan assumed
~50 s/example from audio duration alone; it ignored the ~20 s `--trailing-silence` plus connect and
teardown per example. Projected finish ~10:00. This pushes A3 to ~10:15 and A5 after that; A4 is
unaffected because it needs only a live server, not an idle one.

The model is over-calling on the jinja branch too — e.g. `ecommerce_13`: expected
`['add_to_cart','track_order']`, got `['add_to_cart','track_order','track_order',
'update_identity_doc']`. **Not a result yet** — n=30, and `update_identity_doc` was already the most
frequent spurious call on the default branch (10 occurrences). Worth noting only because it is the
early signal for A3's second outcome ("doesn't move"). Wait for the score.

Final: 100/100 `status=completed`, 0 errors, 0 silent episodes, `rtf` 1.00–1.01 throughout. Branch
proof re-checked at the end — jinja ×100, restraint text ×0. Wall clock ~2 h 50 m.

### A3 — the answer, and it is outcome 3: **the prompt was not the cause**

The night existed to test one hypothesis: *Tool Selection is 9.4 pts short of the card because our
prompt carried two contradictory tool-use policies.* Removing the contradiction did not close the
gap. It made every metric slightly worse.

| metric, n=100 | `nemo_rt` (default branch) | `nemo_rt_jinja` (faithful) | card |
| --- | --- | --- | --- |
| **Tool Selection** (judge-free) | 71.7 % | **70.8 %** | **82.5 %** |
| Argument accuracy (judge differs) | 50.7 % | 48.2 % | 44.2 % |
| Pass@1 | 35.0 % | **31.0 %** | 33.0 % |
| Response quality (judge) | 0.480 | 0.390 | — |
| turn-take rate | 98/100 | **100/100** | — |

So: **write down outcome 2-and-a-bit.** The gap is real, judge-free, and internal to our setup; it
is not a launch-flag artifact. It is not cleanly outcome 3 either — the restraint text was worth
~1 pt of Tool Selection and ~4 pts of Pass@1, which is not "the card relies on restraint the
benchmark doesn't supply", it is noise-adjacent on n=100. The honest claim is that **the prompt
branch is not where the missing 11 points live**, and the next hypothesis has to come from
somewhere else.

The mechanism, which is the useful part. Removing the restraint text did exactly what it should to
call volume, and that bought nothing:

| | `nemo_rt` | `nemo_rt_jinja` |
| --- | --- | --- |
| tool calls emitted vs expected | 184 (**1.23x**) | 162 (**1.08x**) |
| recall over tool names | 0.778 | **0.765** |
| precision over tool names | 0.856 | **0.853** |
| episodes right / over / under-calling | 52 / 27 / 21 | 49 / **29** / 22 |

Over-calling fell by 22 calls and precision *did not move* (0.856 → 0.853) while recall *fell*
(0.778 → 0.765). That is only possible if the calls the restraint text was suppressing were
disproportionately **correct** ones. The model is not over-calling because something told it to; it
is over-calling because it cannot tell which tool the turn needs, and a broader net catches more
right answers along with the wrong ones.

Where the delta sits, by cut: `hard` is **identical** (0.609 both arms), `medium` is **identical**
(0.770 both arms), and the entire Tool Selection difference is on `easy` (0.758 → 0.732). Pass@1
degrades most on multi-tool episodes (2 tools: 0.111 → 0.056; 3 tools: 0.0 → 0.0). Nothing in the
prompt change touched the hard cases at all.

Two incidental findings worth keeping:

* **`ecommerce_20_66c4f3cb14cbfc4db836bd4e` completed cleanly on the jinja branch.** It was
  documented as a *deterministic* Triton backend failure (`Fast extract: exceeded 512 steps without
  eotc_id` → `append_request: request not found` → HTTP 500 → WS 1011) surviving three retries. It
  produced three sensible calls this time. So that crash is **prompt-dependent**, not a fixed
  property of the example — the degenerate-tool-call path is reachable from the restraint prompt and
  not from the clean one. Correct §3 of `FDB_V3_REPRODUCTION.md` if it still calls it deterministic.
* Silent episodes went 2 → 0, so `*_all` and turn-taken metrics coincide on the jinja arm. That is
  the one unambiguous improvement.

### Signed per-turn latency — the metric Kai approved

`scripts/fdb_v3_signed_latency.py`, run over both arms. Purpose: FDB's own metric drops every
negative latency as an "interruption", which discards 35 of 98 examples on `nemo_rt` — and those are
exactly the fastest responses, so the surviving mean is biased slow by construction.

| | `nemo_rt` | `nemo_rt_jinja` |
| --- | --- | --- |
| acoustic user turns | 117 | 117 |
| turns answered / unanswered | 100 / 17 | 101 / 16 |
| **agent responses** | **135** | **136** |
| turns cut into >1 response | 32 | 27 |
| unprompted openings (examples) | 28 (17) | 29 (17) |
| signed median | **+2.64 s** | +2.72 s |
| **barge-in rate** (turns / examples) | 25.2 % / 29.6 % | **22.8 % / 23.0 %** |
| barge-in median depth | −5.28 s | −5.36 s |
| clean-response median | +3.12 s | +3.28 s |

Two structural findings that no published column shows:

1. **The server over-segments.** 135 responses against 117 acoustic turns; 32 turns get cut into
   more than one response. This is the mechanism behind the barge-ins — the end-of-speech detector
   fires at intra-turn pauses, and the model answers each fragment.
2. **17 of 100 episodes open unprompted**, with a greeting at ~2 s before the user has said
   anything. Any pairing rule keyed on "first agent onset after the turn started" scores that
   greeting as a 26-second barge-in; this script pairs through the server's own end-of-speech
   markers instead, which is why its numbers are trustworthy and the first draft's were not.

**Do not write this up as "less eager, therefore better."** Both eagerness measures fell together
(barge-in 29.6 → 23.0 % of examples, over-calling 1.23x → 1.08x) and Tool Selection did *not*
improve. They plausibly share a cause; reducing them does not buy accuracy.

### B1 + B2 — the latency section exists for the first time, and it needs two numbers

B1's flagged risk did not materialise: `nvidia/parakeet-tdt-0.6b-v2` downloaded from a compute node
without trouble, 100/100 transcribed, 0 failures. B2 then re-scored `nemo_rt` end to end (372 judge
calls, 0 failed). Headline metrics are unchanged from the 2026-08-19 run, as they must be — B1 only
adds `user_speech_end_rel`, which no headline metric reads:

```
tool_selection_acc 73.1 % (n=98) / 71.7 % (n=100)   card 82.5 %
argument_acc       51.7 % (n=98) / 50.7 % (n=100)   card 44.2 %
pass_rate (Pass@1) 35.0 % (n=100)                   card 33.0 %
```

The new part, `logs/fdb_v3/nemo_rt_latency_report.json`:

| | N | mean | median | min / max |
| --- | --- | --- | --- | --- |
| First response (FDB metric) | 63 | 4.05 s ± 3.22 | 3.04 s | 0.32 / 15.44 s |
| Tool call | 51 | 1.46 s ± 1.93 | 1.12 s | −1.76 / 11.12 s |
| Task completion | 63 | 4.82 s ± 5.45 | 3.20 s | 0.32 / 35.36 s |
| Filler usage | 3/63 (5 %) | | | |

**4.05 s is not comparable to the card's 448 ms, and neither is the 0.64 s we have been quoting.
They anchor to different instants, and the gap between the anchors is the actual finding.**

* **FDB's anchor** is the acoustic end of the user's *first* turn — Parakeet word timestamps split
  on a >2.0 s gap (`fdb_v3_asr_input.py::speech_end`, the benchmark's own rule from
  `run_tool_benchmark.py:358`). `first_response_latency = agent's first word − that instant`.
* **Our anchor** (`response_latency_s`, in every result file) is *the server's own ASR
  end-of-speech marker*. Aggregated over all 100 files: **135 turns on 98 examples, mean 0.730 s,
  median 0.64 s, sd 0.954, floor 0.48 s, p90 0.80 s, max 9.44 s.** First turn only: n=98, mean
  0.699 s, median 0.64 s.

Reconciling them per example (`marker = agent_first_word − response_latency_s[0]`, then
`delta = marker − acoustic end of turn 1`):

| set | N | delta median | delta mean | reading |
| --- | --- | --- | --- | --- |
| FDB keeps (FR ≥ 0) | 63 | **+2.32 s** | +3.31 s | server declared the turn over *later* than the audio did |
| FDB drops (FR < 0) | 35 | **−11.28 s** | −11.77 s | server declared it over *mid-turn* and the agent started talking |

So the server-anchored 0.64 s is **flattering, not conservative**: on the 63 kept examples the
server's end-of-speech marker sits a median 2.3 s *after* the user actually stopped, and latency
measured from a late anchor is small by construction.

**This corrects a claim in `FDB_V3_REPRODUCTION.md` §1**, which said the marker "lags the acoustic
end, so these are if anything pessimistic." The direction is backwards — a lagging anchor makes the
number optimistic — and the lag is not uniform: it is +2.3 s median on 63 examples and −11.3 s
median on the other 35.

**The 35 dropped examples are the more interesting half.** All 35 have a perfectly clean
server-anchored first-turn latency (0.48–0.96 s, median 0.64, every one positive), so nothing looks
wrong from the server's side. What happened is that the server's end-of-speech fired at an
*intra-turn* pause and the agent began speaking while the same user turn was still going. Worked
example, `ecommerce_01_65e8cf8f4c7424fa062e54a3` — the user's turn 1 is one continuous 14 s
utterance, 1.20 s → 15.28 s, no internal gap above 0.56 s:

```
 8.88-10.16  "Could you track it for me?"     <- server calls end-of-speech here (~10.32)
10.80        agent's first word                  server-anchored latency 0.48 s
10.48-15.28  "The order ID is A B C one two three."   <- user is still in the same turn
```

FDB scores that FR = 10.80 − 15.28 = **−4.48 s** and drops it as an interruption. Both measurements
are correct; they disagree about where a turn ends.

**Hypothesis this raises (not yet tested).** The agent begins speaking mid-turn on **35 of 98
examples (36 %)**, at pauses as short as ~0.3 s. That is the same eagerness that shows up in Tool
Selection as **184 calls emitted vs 150 expected (1.23x over-calling)**. Two independent
measurements of one behaviour would be a much stronger story than either alone — and if A2's jinja
arm moves the over-calling, re-running this latency analysis on `nemo_rt_jinja` tests whether the
barge-in rate moves with it. That is a cheap, concrete next experiment; **B1's second pass over
`nemo_rt_jinja` should therefore be treated as load-bearing, not as a nice-to-have.**

Caveat that stands regardless: for the container arm the agent channel *is* audio, so `asr_chunks`
timestamps are real speech onsets and this analysis is sound. The same numbers on a research-path
arm would be first-*token* times, which are not comparable.

### One stale field found in the old result files

Every `result_nemo_rt.json` carries `notes.prompt` = *"the tool block is rendered server-side from
/s2s/prompt_template.jinja"*. **That is wrong for the `nemo_rt` arm** — those 100 sessions ran the
default `TOOLS_TEMPLATE` branch, as deviation 2 establishes. The note was written before the branch
was discovered. `--server-prompt-mode` (added for A2) stamps the branch explicitly, so
`result_nemo_rt_jinja.json` will be self-describing; the `nemo_rt` files are not, and a future
reader trusting that note would mislabel the arm. Do not fix by editing the files in place —
provenance is better served by this note than by a rewrite.

### A4 — run live: the provider works, and it found a 6144-token wall

`tau-voice-2/scripts/nemo_rt_live_check.py` exists and parses. It drives
`DiscreteTimeNemotronRTAdapter` directly from the user channel of a recorded stereo `both.wav`
(auto-detected as the louder channel), converted PCM16 → μ-law with `audioop.lin2ulaw`, so it spends
**zero ElevenLabs characters**. It reports the four falsifiable claims from §2 A4 as explicit
PASS/FAIL lines plus `chunks_dropped`, and writes `logs/nemo_rt_live_check_<domain>.json`.

It was deliberately held until A2 finished — a second concurrent session would have silently
degraded the deliverable — then run `--domain mock` first, then `retail`.

**`mock`: 5 of 6 checks PASS.** 340 ticks, 68 s of audio in 70.4 s wall (1.036), `chunks_dropped=0`,
425 Triton inferences, arguments all decoded to dict, agent audio accounted in telephony bytes
(67.3 s against 70.4 s wall — the 6x-inflation failure mode did not happen), tick loop clean. The
provider works against a real container. The one FAIL is uninformative: the check requires a tool
call or a tool name in the transcript, and the audio is a *retail* episode played against the *mock*
domain's four tools, so there was nothing to call. The transcript is plainly responsive to the audio
("To process this return, I will need your full name, order number, and a photo of the damaged
items").

**`retail`: dead at tick 3, and the reason is a hard blocker for arm C.** WebSocket 1011 "Internal
server error", nothing else client-side. The cause is only in the container log: the backend
hardcodes `max_model_len=6144` (`checkpoint_utils/load_utils.py:424`, a literal, not an env var) and
retail's prompt plus its 16 tool schemas tokenize to **10978**, so vLLM refuses it in
`_send_system_prompt` before any audio is processed. `session.update` had been ACKed with "7469
chars of instructions accepted" — the prompt is not tokenized until the sequence starts, so the
config check tells you nothing.

Followed up with `tau-voice-2/scripts/nemo_rt_prompt_budget.py`, which measures the gate across all
15 loadable domains instead of estimating it (each rejection prints an exact token count, pinning
tokens-per-char at 0.481–0.582): **9 of 15 rejected outright**, including retail (10978) and airline
(10708), telecom over by 2.4x (14854). Then the sharper finding — **"fits" is not "runs"**: `events`,
`housing` and `media` clear the prompt gate and die mid-episode at 44.6 s, 16.2 s and 12.1 s with
`append_request: request not found`, because the same 6144 tokens must also hold the conversation.
Only `mock`, `banking` and `healthcare` completed a 68 s replay. Ordering by prompt+tools+generated
text is monotone (9,886 ✓ / 10,898 ✓ / 10,937 ✗ / 11,442 ✗ / 12,163 ✗), which is a shared budget
rather than a per-domain bug.

Written up with the three ways out in `TAU_VOICE_SFT_PLAN.md` §0e-bis, which also corrects §0e: its
32000-*character* gate is real but slack, and the binding limit is 3x tighter and in tokens. The
underlying model is `max_position_embeddings=131072`, so 6144 is a serving choice and the literal is
patchable in a writable container — but that deviates from the released config, so FDB-v3 numbers
must not be re-quoted from a patched server. **This is Kai's call, not mine (see §7).**

---

## 7. Left for Kai — do not decide these alone

1. Push the 4 unpushed commits?
2. Local TTS behind `synthesis/synthesize.py:22`, or wait for a paid ElevenLabs key? Gates all of
   arm A beyond replay experiments.
3. Raise `max_fc_total_tokens` to 12,000 in the training config (B4 measures whether it fits;
   changing the config is his call).
4. Whether to spend the last 304 ElevenLabs characters on anything at all.
