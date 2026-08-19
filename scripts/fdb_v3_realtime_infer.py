#!/usr/bin/env python3
"""Run FDB-v3 against the NemotronLabs Voicechat realtime container, one WebSocket session
per example, and write the `result_{provider}.json` files the benchmark's evaluators read.

Why not the in-repo replay
--------------------------
`fdb_v3_nemo_infer.py` drives `DuplexSTTModel` in-process. That path hard-disables the LLM
cache for Nemotron (`duplex_stt_model.py:3718-3722`), so it runs ~12x slower than realtime and
has no speech decoder in our remapped checkpoint. It is not the configuration the model card's
numbers come from: the card's "Interactive streaming deployment" section points at the NIM
container (Triton + vLLM), measured here at wall/audio 1.001 with ~0.5 s turn-taking latency.
That path is also *better behaved* -- on `ecommerce_01` the replay hallucinated
`order_id: "LHR"` where the container returns `track_order({"order_id": "A-BC123"})` -- so the
replay is kept only as a deliberate A/B, and this driver is the one that produces numbers.

Faithfulness notes
------------------
* The tool block is not ours. Tools go in flat (`{name, description, parameters}`) via
  `session.update` and the *server* renders the function-calling prompt from its own
  `/s2s/prompt_template.jinja`. Every prompt-format question this repo previously had to
  reason about is thereby answered by the server, not by us.
* Tool *execution* is still the benchmark's own `MockAPIRegistry` at the `instant` profile, and
  the executed/rejected distinction is `fdb_v3_nemo_infer.ToolExecutor`'s, unchanged: a call
  counts for F1 only if LiveKit's function tool would have run it.
* Audio is paced at true realtime, 24 kHz PCM16 in 80 ms chunks, as `/s2s/nemotron-voicechat-client.py`
  does. Faster would be throughput, not a conversation.
* `transcript` is the server's `response.output_audio_transcript`, i.e. the transcript of the
  speech it actually synthesised, and `agent_{provider}.wav` is that audio -- so unlike the
  replay this is comparable to the published latency and response-quality sections.
* Timestamps are on the *input* audio clock (seconds of user audio streamed when the event
  arrived), which is what the benchmark's metadata timestamps mean.

Usage:
    python scripts/fdb_v3_realtime_infer.py --server ws://ip-10-1-105-182:9000 --limit 1 -v
    python scripts/fdb_v3_realtime_infer.py --server ws://host:9000 --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path("/fsx/home/kai.li/code/tau-voice-2/src")))

from fdb_v3_nemo_infer import FDB_V3_DIR, ToolExecutor, discover  # noqa: E402

CLIENT_RATE = 24000  # the API's client-side rate in and out (api-reference.md)
CHUNK_MS = 80


def load_audio_24k(path: Path, trailing_silence_s: float):
    """Mono float32 at 24 kHz with trailing silence appended.

    The silence is not cosmetic: the model is full duplex and only emits while input flows, so
    a stream that stops at the user's last word truncates the reply mid-sentence (deploy.md
    recommends ~20 s). The released WAVs already carry the in-conversation gaps.
    """
    import numpy as np
    import soundfile as sf
    from math import gcd

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != CLIENT_RATE:
        from scipy.signal import resample_poly

        g = gcd(int(rate), CLIENT_RATE)
        audio = resample_poly(audio, CLIENT_RATE // g, int(rate) // g).astype("float32")
    pad = int(trailing_silence_s * CLIENT_RATE)
    return np.concatenate([audio, np.zeros(pad, dtype="float32")]) if pad else audio


def flat_tools(tools) -> List[Dict[str, Any]]:
    """tau2 ``Tool`` -> the realtime API's tool shape.

    The API takes the OpenAI *Realtime* function form (name/description/parameters at the top
    level), not the chat-completions ``{"type": "function", "function": {...}}`` wrapper.
    """
    out = []
    for tool in tools:
        schema = tool.openai_schema
        fn = schema.get("function", schema)
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return out


class Episode:
    """One example: stream its audio, service tool calls, collect what the evaluators need."""

    def __init__(self, executor: ToolExecutor, verbose: bool = False):
        self.executor = executor
        self.verbose = verbose
        self.t0 = 0.0
        self.sent_audio_s = 0.0  # input-audio clock: what the benchmark's timestamps mean
        self.audio_out = bytearray()
        self.delta_index: List[tuple] = []  # (input-clock t, first sample, n samples)
        self.transcript_chunks: List[Dict[str, Any]] = []
        self.user_transcript: List[str] = []
        self.errors: List[Dict[str, Any]] = []
        self.speech_stopped_at: List[float] = []
        self.stats: Dict[str, Any] = {}
        self.send_lag_max = 0.0
        self.dropped_calls = 0

    async def _send(self, ws, payload: Dict[str, Any]):
        payload.setdefault("event_id", str(uuid.uuid4()))
        await ws.send(json.dumps(payload))

    async def stream(self, ws, audio):
        step = int(CLIENT_RATE * CHUNK_MS / 1000)
        import numpy as np

        start = time.time()
        for i in range(0, len(audio), step):
            scheduled = start + (i / step) * (CHUNK_MS / 1000.0)
            now = time.time()
            if now < scheduled:
                await asyncio.sleep(scheduled - now)
            else:
                self.send_lag_max = max(self.send_lag_max, now - scheduled)
            chunk = np.clip(audio[i : i + step], -1.0, 1.0)
            await self._send(ws, {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode((chunk * 32767.0).astype("<i2").tobytes()).decode("ascii"),
            })
            self.sent_audio_s = (i + len(chunk)) / CLIENT_RATE

    async def receive(self, ws, done: asyncio.Event):
        try:
            await self._receive(ws, done)
        finally:
            # If the server hangs up without a `session.end` we must not sit out the whole
            # close timeout -- an empty `server_stats` in the result file is the tell.
            done.set()

    async def _receive(self, ws, done: asyncio.Event):
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("type", "?")
            t = self.sent_audio_s
            if kind == "response.output_audio.delta":
                blob = base64.b64decode(msg.get("delta", ""))
                self.delta_index.append((t, len(self.audio_out) // 2, len(blob) // 2))
                self.audio_out += blob
            elif kind == "response.output_audio_transcript.delta":
                delta = msg.get("delta", "")
                if delta:
                    self.transcript_chunks.append(
                        {"text": delta, "timestamp": [round(t, 3), round(t, 3)]}
                    )
            elif kind == "conversation.item.input_audio_transcription.delta":
                self.user_transcript.append(msg.get("delta", ""))
            elif kind == "input_audio_buffer.speech_stopped":
                self.speech_stopped_at.append(round(t, 3))
            elif kind == "response.function_call_arguments.done":
                await self._tool_call(ws, msg, t)
            elif kind == "error":
                self.errors.append(msg.get("error", {}))
            elif kind == "session.end":
                self.stats = msg.get("stats", {})
                done.set()
                return

    async def _tool_call(self, ws, msg, t: float):
        name = msg.get("name") or ""
        try:
            call_args = json.loads(msg.get("arguments") or "{}")
        except json.JSONDecodeError:
            call_args = {}
            self.dropped_calls += 1
        if not isinstance(call_args, dict):
            call_args = {}
        response, is_error = self.executor(name, call_args, t)
        await self._send(ws, {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": msg.get("call_id"),
                "output": response,
            },
        })
        if self.verbose:
            print(f"    [{t:6.2f}s] {name}({call_args}) -> "
                  f"{'REJECTED' if is_error else 'ok'}", flush=True)

    # -- derived quantities ---------------------------------------------------------
    def speech_onsets(self, frame_ms: float = 20.0, thresh: float = 0.02,
                      min_silence_s: float = 0.30) -> List[float]:
        """Input-clock times at which the agent *starts speaking*.

        The output channel streams silence too, so the first audio delta after end-of-speech
        is not an answer onset -- it lands within 0 ms of every `speech_stopped`. Energy on the
        received stream is, and each onset is dated by the delta that carried it.
        """
        import bisect

        import numpy as np

        if not self.delta_index:
            return []
        pcm = np.frombuffer(bytes(self.audio_out), dtype="<i2").astype("float32") / 32768.0
        step = int(CLIENT_RATE * frame_ms / 1000)
        n = len(pcm) // step
        if n == 0:
            return []
        rms = np.sqrt((pcm[: n * step].reshape(n, step) ** 2).mean(axis=1))
        gap = int(min_silence_s / (frame_ms / 1000.0))
        arrivals = [a for a, _, _ in self.delta_index]
        firsts = [s for _, s, _ in self.delta_index]

        onsets, silence = [], gap
        for k in range(n):
            if rms[k] > thresh:
                if silence >= gap:
                    i = min(max(bisect.bisect_right(firsts, k * step) - 1, 0), len(arrivals) - 1)
                    onsets.append(round(arrivals[i], 3))
                silence = 0
            else:
                silence += 1
        return onsets


async def run_example(args, example_dir: Path, metadata: dict, tools, registry) -> dict:
    import websockets

    audio = load_audio_24k(example_dir / "input.wav", args.trailing_silence)
    input_s = len(audio) / CLIENT_RATE
    executor = ToolExecutor(tools, registry, response_style=args.tool_response_style)
    ep = Episode(executor, verbose=args.verbose)

    status = "completed"
    wall_start = time.time()
    async with websockets.connect(args.url, max_size=None, ping_interval=None) as ws:
        ep.t0 = wall_start
        await ep._send(ws, {
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": CLIENT_RATE}},
                    "output": {"format": {"type": "audio/pcm", "rate": CLIENT_RATE}},
                },
                "instructions": args.instructions,
                "tools": flat_tools(tools),
            },
        })
        done = asyncio.Event()
        recv = asyncio.create_task(ep.receive(ws, done))
        await ep.stream(ws, audio)
        await ep._send(ws, {"type": "session.close"})
        try:
            await asyncio.wait_for(done.wait(), timeout=args.close_timeout)
        except asyncio.TimeoutError:
            status = "no_session_end"
        recv.cancel()
    wall = time.time() - wall_start

    if args.save_audio and ep.audio_out:
        import numpy as np
        import soundfile as sf

        sf.write(str(example_dir / f"agent_{args.provider}.wav"),
                 np.frombuffer(bytes(ep.audio_out), dtype="<i2"), CLIENT_RATE, subtype="PCM_16")

    onsets = ep.speech_onsets()
    out_audio_s = len(ep.audio_out) / 2 / CLIENT_RATE
    span = (ep.delta_index[-1][0] - ep.delta_index[0][0]) if len(ep.delta_index) > 1 else 0.0
    latencies = []
    for stop in ep.speech_stopped_at:
        onset = next((o for o in onsets if o > stop), None)
        if onset is not None:
            latencies.append(round(onset - stop, 3))
    if ep.errors and not ep.stats:
        status = "inference_error"

    transcript = "".join(c["text"] for c in ep.transcript_chunks).strip()
    return {
        # -- fields the benchmark's evaluators read ---------------------------------
        "example_id": metadata["id"],
        "pid": example_dir.name.rsplit("_", 1)[-1],
        "category": metadata.get("domain", "unknown"),
        "title": metadata.get("title", ""),
        "provider": args.provider,
        "evaluated_at": datetime.datetime.now().isoformat(),
        "status": status,
        "transcript": transcript,
        "asr_chunks": ep.transcript_chunks,
        "actual_tool_calls": executor.executed,
        # -- ours: diagnostics and honest labelling ---------------------------------
        "notes": {
            "harness": "nemo-voice-agent/scripts/fdb_v3_realtime_infer.py "
                       "(NIM container over WebSocket, no LiveKit)",
            "server": args.url,
            "transcript_source": "server response.output_audio_transcript, i.e. the transcript "
                                 "of the speech actually synthesised",
            "clock": "timestamps are input-audio-clock seconds (user audio streamed), and "
                     "audio is paced at realtime so they track wall clock",
            "prompt": "system prompt = the FDB-v3 VoiceAgent instructions; the tool block is "
                      "rendered server-side from /s2s/prompt_template.jinja",
            "tool_call_logging": "actual_tool_calls holds only calls that would execute under "
                                 "LiveKit; see rejected_tool_calls",
            "tool_response_style": args.tool_response_style,
            "trailing_silence_s": args.trailing_silence,
        },
        "rejected_tool_calls": executor.rejected,
        "unparseable_tool_calls": ep.dropped_calls,
        "input_duration_s": round(input_s, 3),
        "inference_wall_s": round(wall, 1),
        "realtime_factor": round(wall / input_s, 3) if input_s else None,
        "agent_audio_s": round(out_audio_s, 2),
        "output_realtime_ratio": round(out_audio_s / span, 3) if span else None,
        "send_lag_max_s": round(ep.send_lag_max, 3),
        "user_speech_end_s": ep.speech_stopped_at,
        "agent_speech_onsets_s": onsets,
        "response_latency_s": latencies,
        "audio_agent_speech_start": onsets[0] if onsets else None,
        "user_transcript": "".join(ep.user_transcript),
        "server_stats": ep.stats,
        "server_errors": ep.errors,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", required=True, help="host:port or ws://host:port[/v1/realtime]")
    p.add_argument("--data-dir", type=Path, default=FDB_V3_DIR / "fdb_v3_data_released")
    p.add_argument("--provider", default="nemo_rt",
                   help="Name in result_<provider>.json. Use a distinct name per variant so "
                        "two configurations never overwrite each other's results.")
    p.add_argument("--tool-response-style", choices=("json", "sentence"), default="json",
                   help="json (default) is what the benchmark's LiveKit agent returns.")
    p.add_argument("--system-message", choices=("benchmark", "nvidia+benchmark"),
                   default="benchmark",
                   help="benchmark (default) = the FDB-v3 VoiceAgent instructions alone, what "
                        "every provider in the published table got.")
    p.add_argument("--trailing-silence", type=float, default=20.0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-save-audio", dest="save_audio", action="store_false",
                   help="Skip agent_<provider>.wav (~3 MB per example; needed for any ASR of "
                        "the agent's speech, and the only artifact you can actually listen to).")
    p.add_argument("--close-timeout", type=float, default=60.0)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    url = args.server.rstrip("/")
    if not url.startswith(("ws://", "wss://")):
        url = f"ws://{url}"
    args.url = url if url.endswith("/v1/realtime") else f"{url}/v1/realtime"

    from fdb_v3_tools import (
        build_registry,
        build_tools,
        extract_instructions,
        nvidia_default_system_message,
    )

    tools = build_tools()
    registry = build_registry(latency_profile="instant")
    args.instructions = extract_instructions()
    if args.system_message == "nvidia+benchmark":
        args.instructions = f"{nvidia_default_system_message()}\n\n{args.instructions}"

    examples = discover(args.data_dir)
    if args.num_shards > 1:
        examples = examples[args.shard::args.num_shards]
    todo = [d for d in examples if args.force or not (d / f"result_{args.provider}.json").exists()]
    skipped = len(examples) - len(todo)  # count before --limit, or the log line lies
    if args.limit:
        todo = todo[: args.limit]

    print(f"server {args.url}", flush=True)
    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} of {len(examples)} examples to run "
          f"({skipped} already have result_{args.provider}.json"
          f"{f', {len(examples) - skipped - len(todo)} held back by --limit' if args.limit else ''})",
          flush=True)
    print(f"prompt: system_message={args.system_message} {len(args.instructions)} chars; "
          f"{len(tools)} tools registered via session.update", flush=True)

    failures = 0
    for i, example_dir in enumerate(todo, 1):
        metadata = json.loads((example_dir / "metadata.json").read_text())
        print(f"[{i}/{len(todo)}] {example_dir.name} "
              f"({metadata['difficulty']}, {metadata['num_expected_calls']} expected call(s))",
              flush=True)
        try:
            payload = asyncio.run(run_example(args, example_dir, metadata, tools, registry))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            failures += 1
            payload = {
                "example_id": metadata["id"],
                "pid": example_dir.name.rsplit("_", 1)[-1],
                "category": metadata.get("domain", "unknown"),
                "title": metadata.get("title", ""),
                "provider": args.provider,
                "evaluated_at": datetime.datetime.now().isoformat(),
                "status": "inference_error",
                "error": f"{type(exc).__name__}: {exc}",
                "transcript": "",
                "asr_chunks": [],
                "actual_tool_calls": [],
            }
        (example_dir / f"result_{args.provider}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        got = [c["function"] for c in payload.get("actual_tool_calls", [])]
        expected = [c["function"] for c in metadata["expected_tool_calls"]]
        print(f"    status={payload['status']} rtf={payload.get('realtime_factor')} "
              f"latency={payload.get('response_latency_s')} expected={expected} got={got} "
              f"rejected={len(payload.get('rejected_tool_calls', []))}", flush=True)
        if args.verbose:
            print(f"    args    : {[c['args'] for c in payload.get('actual_tool_calls', [])]}", flush=True)
            print(f"    agent   : {payload.get('transcript', '')[:300]}", flush=True)

    print(f"shard {args.shard}: done, {failures} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
