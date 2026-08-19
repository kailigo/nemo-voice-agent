#!/usr/bin/env python3
"""Measure response latency and realtime headroom against the NemotronLabs Voicechat server.

Why this exists
---------------
This repo contains two implementations of the same weights. The eager research path
(``nemo/collections/speechlm2/models/duplex_stt_model.py:3718``) hard-disables the LLM cache
for Nemotron -- it logs "Using no-cache mode for Nemotron (full history each step)" -- and so
recomputes the whole conversation every 80 ms frame, measured at 12.2x slower than realtime.
The *served* path is the NIM container (Triton + vLLM, see ``voicechat_realtime_instructions/``),
which keeps both halves of the hybrid state: paged KV for the attention layers and the Mamba2
SSM state. The model card's 448 ms turn-taking / 480 ms interruption latencies are only
definable there. So before any benchmark number is worth producing, measure the served path.

What it measures
----------------
Audio is paced at true realtime (80 ms per chunk, wall clock), exactly as
``/s2s/nemotron-voicechat-client.py`` does, because anything faster measures throughput
rather than latency.

* ``response_latency_s`` -- per turn, from the server's ``input_audio_buffer.speech_stopped``
  to the **onset of speech** in the returned audio. This is the quantity the card calls smooth
  turn-taking latency. Note that the first *audio delta* after end-of-speech is not that
  number: the model is full duplex and streams its output channel continuously, silence
  included, so deltas keep arriving whether or not it is talking (measured: a delta lands
  within 0 ms of every ``speech_stopped``). Onset is therefore detected by energy on the
  received stream and timestamped by the arrival of the delta that carried it.
* ``output_realtime_ratio`` -- seconds of synthesised audio received divided by the wall-clock
  span it arrived over. < 1 means the server cannot sustain speech in realtime, which is the
  failure mode that would make the model unusable regardless of first-token latency.
* ``send_lag_max_s`` -- how far behind schedule our sends fell. A server that cannot consume
  audio in realtime shows up here as TCP backpressure.
* ``session.end`` stats straight from the server: ``chunks_dropped`` and
  ``triton_inferences`` per second of audio (should be ~12.5/s for an 80 ms frame).

Nothing here is FDB-v3 specific; ``--tools`` exists only so a tool-calling turn can be timed
too (tool calls are acknowledged with a canned result, not executed).

Usage:
    python scripts/voicechat_realtime_latency.py --server ws://ip-10-1-105-182:9000 \
        --input-file /fsx/home/kai.li/data/voicechat/turn_taking.wav --out logs/fdb_v3/lat.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

CLIENT_RATE = 24000  # the API's client-side rate, input and output (api-reference.md)
CHUNK_MS = 80  # "Recommended chunk duration", and the model's frame size


def load_audio(path: Path, trailing_silence_s: float) -> "Any":
    """Return mono float32 at 24 kHz with trailing silence appended.

    The server resamples 24 kHz -> 16 kHz internally; we resample here rather than sending a
    native-rate file, because ``session.update`` only negotiates the rate, it does not convert.
    Trailing silence matters: the model is full duplex and only produces output while input
    flows, so a file that ends at the user's last word truncates the reply (deploy.md).
    """
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # mono
    if rate != CLIENT_RATE:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(rate), CLIENT_RATE)
        audio = resample_poly(audio, CLIENT_RATE // g, int(rate) // g).astype("float32")
    pad = int(trailing_silence_s * CLIENT_RATE)
    if pad:
        audio = np.concatenate([audio, np.zeros(pad, dtype="float32")])
    return audio


def pcm16(chunk) -> bytes:
    import numpy as np

    return (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class Session:
    def __init__(self, args):
        self.args = args
        self.events: List[Dict[str, Any]] = []
        self.audio_out = bytearray()
        # (arrival_t, first_sample_index, n_samples) per delta, so a sample position in the
        # concatenated output can be mapped back to when it reached us.
        self.delta_index: List[tuple] = []
        self.t0 = 0.0
        self.send_lag_max = 0.0
        self.sent_audio_s = 0.0
        self.stats: Dict[str, Any] = {}
        self.errors: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.user_transcript: List[str] = []
        self.agent_transcript: List[str] = []

    # --- wire helpers -------------------------------------------------------------
    def log(self, kind: str, **extra):
        self.events.append({"t": time.time() - self.t0, "type": kind, **extra})

    async def send(self, ws, payload: Dict[str, Any]):
        payload.setdefault("event_id", str(uuid.uuid4()))
        await ws.send(json.dumps(payload))

    # --- tasks --------------------------------------------------------------------
    async def stream_audio(self, ws, audio):
        """Send 80 ms chunks on a fixed wall-clock schedule; record how far behind we get."""
        step = int(CLIENT_RATE * CHUNK_MS / 1000)
        start = time.time()
        for i in range(0, len(audio), step):
            scheduled = start + (i / step) * (CHUNK_MS / 1000.0)
            now = time.time()
            if now < scheduled:
                await asyncio.sleep(scheduled - now)
            else:
                self.send_lag_max = max(self.send_lag_max, now - scheduled)
            chunk = audio[i : i + step]
            await self.send(ws, {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16(chunk)).decode("ascii"),
            })
            self.sent_audio_s = (i + len(chunk)) / CLIENT_RATE
        self.log("client.audio_done", audio_s=self.sent_audio_s)

    async def receive(self, ws, done: asyncio.Event):
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("type", "?")
            if kind == "response.output_audio.delta":
                blob = base64.b64decode(msg.get("delta", ""))
                self.delta_index.append(
                    (time.time() - self.t0, len(self.audio_out) // 2, len(blob) // 2)
                )
                self.audio_out += blob
                self.log(kind, bytes=len(blob), sent_audio_s=self.sent_audio_s)
            elif kind == "response.output_audio_transcript.delta":
                self.agent_transcript.append(msg.get("delta", ""))
                self.log(kind)
            elif kind == "conversation.item.input_audio_transcription.delta":
                self.user_transcript.append(msg.get("delta", ""))
                self.log(kind)
            elif kind == "response.function_call_arguments.done":
                self.tool_calls.append({
                    "t": time.time() - self.t0,
                    "name": msg.get("name"),
                    "arguments": msg.get("arguments"),
                    "call_id": msg.get("call_id"),
                })
                self.log(kind, name=msg.get("name"), sent_audio_s=self.sent_audio_s)
                # Canned result: the point is to time the resume, not to be a tool harness.
                await self.send(ws, {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": msg.get("call_id"),
                        "output": json.dumps({"status": "ok"}),
                    },
                })
            elif kind == "error":
                self.errors.append(msg.get("error", {}))
                self.log(kind, error=msg.get("error"))
            elif kind == "session.end":
                self.stats = msg.get("stats", {})
                self.log(kind, stats=self.stats)
                done.set()
                return
            else:
                self.log(kind, sent_audio_s=self.sent_audio_s)

    # --- metrics ------------------------------------------------------------------
    def speech_onsets(self, frame_ms: float = 20.0, thresh: float = 0.02,
                      min_silence_s: float = 0.30) -> List[Dict[str, float]]:
        """Arrival times of speech onsets in the agent's audio stream.

        The output channel carries silence as well as speech, so an onset is a frame above
        ``thresh`` RMS (fraction of full scale) preceded by at least ``min_silence_s`` of
        frames below it. Each onset is timestamped with the arrival time of the delta that
        carried that sample -- the audio clock and the wall clock agree only as long as the
        server keeps up, and the point of this script is not to assume that.
        """
        import numpy as np

        if not self.delta_index:
            return []
        pcm = np.frombuffer(bytes(self.audio_out), dtype="<i2").astype("float32") / 32768.0
        step = int(CLIENT_RATE * frame_ms / 1000)
        n = len(pcm) // step
        if n == 0:
            return []
        rms = np.sqrt((pcm[: n * step].reshape(n, step) ** 2).mean(axis=1))
        loud = rms > thresh
        gap_frames = int(min_silence_s / (frame_ms / 1000.0))

        starts = [t for t, _, _ in self.delta_index]
        first = [s for _, s, _ in self.delta_index]

        def arrival_of(sample: int) -> float:
            import bisect

            i = min(bisect.bisect_right(first, sample) - 1, len(starts) - 1)
            return starts[max(i, 0)]

        onsets, silence = [], gap_frames  # start "in silence" so a leading onset is caught
        for k in range(n):
            if loud[k]:
                if silence >= gap_frames:
                    onsets.append({
                        "audio_pos_s": round(k * frame_ms / 1000.0, 3),
                        "arrival_t": round(arrival_of(k * step), 3),
                        "rms": round(float(rms[k]), 4),
                    })
                silence = 0
            else:
                silence += 1
        return onsets

    def metrics(self, audio_s: float, wall_s: float) -> Dict[str, Any]:
        deltas = [e for e in self.events if e["type"] == "response.output_audio.delta"]
        stops = [e for e in self.events if e["type"] == "input_audio_buffer.speech_stopped"]
        starts = [e for e in self.events if e["type"] == "input_audio_buffer.speech_started"]
        onsets = self.speech_onsets()

        turns = []
        for stop in stops:
            onset = next((o for o in onsets if o["arrival_t"] > stop["t"]), None)
            delta = next((d for d in deltas if d["t"] > stop["t"]), None)
            start = next((s for s in reversed(starts) if s["t"] <= stop["t"]), None)
            turns.append({
                "speech_stopped_t": round(stop["t"], 3),
                "speech_onset_t": onset["arrival_t"] if onset else None,
                "response_latency_s": (
                    round(onset["arrival_t"] - stop["t"], 3) if onset else None
                ),
                # Kept for contrast: this is ~0 for a duplex model and is *not* the card's metric.
                "first_delta_after_stop_s": round(delta["t"] - stop["t"], 3) if delta else None,
                "latency_from_speech_start_s": (
                    round(onset["arrival_t"] - start["t"], 3) if onset and start else None
                ),
            })

        out_audio_s = len(self.audio_out) / 2 / CLIENT_RATE
        span = (deltas[-1]["t"] - deltas[0]["t"]) if len(deltas) > 1 else 0.0
        lat = [t["response_latency_s"] for t in turns if t["response_latency_s"] is not None]
        infer = self.stats.get("triton_inferences")
        recv_s = self.stats.get("audio_duration_received_s")
        return {
            "input_audio_s": round(audio_s, 2),
            "wall_s": round(wall_s, 2),
            "wall_over_audio": round(wall_s / audio_s, 3) if audio_s else None,
            "send_lag_max_s": round(self.send_lag_max, 3),
            "turns": turns,
            "response_latency_s_median": round(sorted(lat)[len(lat) // 2], 3) if lat else None,
            "output_audio_s": round(out_audio_s, 2),
            "output_arrival_span_s": round(span, 2),
            # < 1.0 means synthesis cannot keep up with speaking rate.
            "output_realtime_ratio": round(out_audio_s / span, 3) if span else None,
            "first_output_audio_t": round(deltas[0]["t"], 3) if deltas else None,
            "speech_onsets": self.speech_onsets(),
            "tool_calls": self.tool_calls,
            "server_stats": self.stats,
            "inferences_per_audio_s": (
                round(infer / recv_s, 2) if infer and recv_s else None
            ),
            "errors": self.errors,
            "user_transcript": "".join(self.user_transcript),
            "agent_transcript": "".join(self.agent_transcript),
        }


async def run(args) -> Dict[str, Any]:
    import websockets

    audio = load_audio(Path(args.input_file), args.trailing_silence)
    audio_s = len(audio) / CLIENT_RATE
    url = args.server.rstrip("/")
    if not url.startswith(("ws://", "wss://")):
        url = f"ws://{url}"
    if not url.endswith("/v1/realtime"):
        url = f"{url}/v1/realtime"

    tools = []
    if args.tools:
        raw = Path(args.tools).read_text() if Path(args.tools).exists() else args.tools
        tools = json.loads(raw)
    instructions = None
    if args.instructions:
        p = Path(args.instructions)
        instructions = p.read_text() if p.exists() else args.instructions

    sess = Session(args)
    print(f"connecting to {url}  ({audio_s:.1f}s of audio incl. {args.trailing_silence}s silence)")
    async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
        sess.t0 = time.time()
        await sess.send(ws, {
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": CLIENT_RATE}},
                    "output": {"format": {"type": "audio/pcm", "rate": CLIENT_RATE}},
                },
                "instructions": instructions,
                "tools": tools,
            },
        })
        done = asyncio.Event()
        recv = asyncio.create_task(sess.receive(ws, done))
        await sess.stream_audio(ws, audio)
        await sess.send(ws, {"type": "session.close"})
        try:
            await asyncio.wait_for(done.wait(), timeout=args.close_timeout)
        except asyncio.TimeoutError:
            print("warning: no session.end within the close timeout")
        recv.cancel()
    wall = time.time() - sess.t0

    result = sess.metrics(audio_s, wall)
    if args.audio_output and sess.audio_out:
        import numpy as np
        import soundfile as sf

        pcm = np.frombuffer(bytes(sess.audio_out), dtype="<i2")
        sf.write(args.audio_output, pcm, CLIENT_RATE, subtype="PCM_16")
        print(f"agent audio -> {args.audio_output}  ({len(pcm) / CLIENT_RATE:.1f}s)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"events": sess.events, **result}, indent=2))
        print(f"timings -> {args.out}")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", required=True, help="host:port, or ws://host:port[/v1/realtime]")
    p.add_argument("--input-file", required=True)
    p.add_argument("--trailing-silence", type=float, default=20.0,
                   help="Silence appended after the speech; the model only emits while input flows.")
    p.add_argument("--tools", default=None, help="JSON array inline or a path (optional).")
    p.add_argument("--instructions", default=None, help="System prompt inline or a path (optional).")
    p.add_argument("--audio-output", default=None)
    p.add_argument("--out", default=None, help="Where to write the full event log + metrics.")
    p.add_argument("--close-timeout", type=float, default=60.0)
    args = p.parse_args()

    result = asyncio.run(run(args))
    keys = ("input_audio_s", "wall_s", "wall_over_audio", "send_lag_max_s",
            "first_output_audio_t", "response_latency_s_median", "output_audio_s",
            "output_arrival_span_s", "output_realtime_ratio", "inferences_per_audio_s")
    print("\n--- latency ---")
    for k in keys:
        print(f"  {k:26s} {result.get(k)}")
    for turn in result["turns"]:
        print(f"  turn: user speech end {turn['speech_stopped_t']}s -> agent speech onset "
              f"{turn['speech_onset_t']}s  latency={turn['response_latency_s']}s "
              f"(first delta after stop: {turn['first_delta_after_stop_s']}s)")
    if result["tool_calls"]:
        for c in result["tool_calls"]:
            print(f"  tool call @{c['t']:.2f}s  {c['name']}({c['arguments']})")
    print(f"  server stats: {result['server_stats']}")
    if result["errors"]:
        print(f"  errors: {result['errors']}")
    print(f"\n  user  : {result['user_transcript'][:300]}")
    print(f"  agent : {result['agent_transcript'][:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
