#!/usr/bin/env python3
"""Run VoiceChat-11B on Full-Duplex-Bench v3 by replaying the benchmark audio locally.

Why this exists instead of a LiveKit provider
---------------------------------------------
FDB-v3's released pipeline is `livekit_inference.py` (a headless client that publishes a
WAV into a LiveKit Cloud room) plus `lk_agent_tool.py` (an agent that wraps a *hosted*
realtime API: GPT Realtime, Gemini, Grok, Ultravox, or a cascaded OpenAI stack). Our model
is 11B of weights on the GPU in this process. Routing its audio out to LiveKit Cloud and
back would add an account, a websocket, a 48 kHz -> 16 kHz -> 24 kHz resample chain and a
realtime pacing constraint that inference cannot meet anyway (no KV cache: Nemotron-Nano
is hybrid Mamba2, so decoding is slower than realtime).

None of that transport is load-bearing for the published metrics. The evaluation scripts
never touch LiveKit -- they read `result_{provider}.json` out of each example directory.
So this script produces exactly that file, and the benchmark's own evaluators run against
it unmodified. Nothing in Full-Duplex-Bench/ is edited by us.

The contract, read off the evaluators rather than guessed:
  evaluate_tool_calls.py:640-652  example_id, actual_tool_calls, transcript (+ whole dict)
  evaluate_tool_calls.py:355-356  turn_take_success = transcript or asr_chunks non-empty
  evaluate_tool_calls.py:320-326  latency = audio_agent_speech_start - user_speech_end_rel
  analyze_tool_latency.py:126-158 user_speech_end_rel, asr_chunks, actual_tool_calls with
                                  timestamp_start
  evaluate_pass_rate.py:530-532   same three fields as evaluate_tool_calls

What is faithful, and what is not
---------------------------------
Faithful:
  * Tool set, tool descriptions, parameter descriptions and agent instructions are parsed
    out of `lk_agent_tool.py` itself (see fdb_v3_tools.py) -- not transcribed.
  * The system prompt is rendered by NVIDIA's own `function_calling/template.jinja`, the
    one their offline FC entrypoint uses. Our `Tool.openai_schema` block is a different
    format -- wrapped in {"type":"function","function":{...}}, unsorted keys -- and NeMo's
    own example of a trained tool block (`s2s_dataset.py:1159`) is the flattened form the
    template produces, so the wrapped form matches neither training nor their inference.
    `--prompt-style tau2_provider` keeps the wrapped form for an A/B on what it costs.
  * Tool execution goes through the benchmark's own `MockAPIRegistry`, at the `instant`
    latency profile the released `run_agent.sh` uses.
  * Tool results are handed back as `json.dumps(result)`, byte-for-byte what
    `AssistantFnc` returns to its model.
  * A call is recorded in `actual_tool_calls` only if it would have *executed* under
    LiveKit -- known tool name, all required arguments present. LiveKit rejects anything
    else before `log_tool_call` runs, so those calls are invisible to the published F1.
    Ours would otherwise be penalised on precision for failures the reference silently
    drops. They are kept in `rejected_tool_calls` because they are the most interesting
    diagnostic we have.
  * The agent gets the audio plus 1.5 s of trailing silence, matching
    `livekit_inference.py:270-282`. The WAVs already contain the response gap: the median
    is 47 s of file for ~10 s of speech.

Not faithful, and stated in every report:
  * `transcript` is the model's own text channel, not Parakeet ASR of its speech. This
    checkpoint's `DuplexSTTModel._post_inference` returns `tokens_audio: None` -- the agent
    channel is text. So `audio_agent_speech_start` is really *first-token* time, and
    `asr_chunks` are text-channel tokens with the audio-clock time they were emitted at.
    Tool selection and argument accuracy do not depend on any of this; the latency and
    response-quality numbers do, and are not comparable to the published table.
  * Timestamps are on the audio clock (samples pushed / 16000), not wall clock. Wall clock
    is meaningless here: inference runs slower than realtime, so a wall-clock latency
    would measure our GPU, not the model's turn-taking.
  * `user_speech_end_rel` is not written here at all. It comes from Parakeet over the
    *input* audio, so it needs no 11B model and no GPU of its own: run
    `fdb_v3_asr_input.py` afterwards to add it. Until then the latency metrics report
    "unavailable" rather than a fabricated zero.

Usage (must run inside a GPU allocation -- the login node has no GPU):
    scripts/fdb_v3_fanout.sh <jobid>              # all 100 examples over 8 GPUs
    python scripts/fdb_v3_nemo_infer.py --limit 1 --verbose    # single example, debug
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

FDB_V3_DIR = Path("/fsx/home/kai.li/code/Full-Duplex-Bench/v3")
TAU2_SRC = Path("/fsx/home/kai.li/code/tau-voice-2/src")
HERE = Path(__file__).resolve().parent

SAMPLE_RATE = 16000  # the model's native input rate (provider.NEMO_SAMPLE_RATE)
FRAME_MS = 80  # one LM frame; the finest tick the session advances on
TAIL_SILENCE_MS = 1500  # livekit_inference.py:271


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


def _ffmpeg() -> str:
    """ffmpeg from the interpreter's own environment, falling back to PATH.

    The voicechat env ships its own ffmpeg 7 (the conda one needs a newer CXXABI than this
    box has -- see the env-recipe notes). We launch python by absolute path, so the env's
    bin/ is not on PATH and a bare "ffmpeg" is a FileNotFoundError.
    """
    local = Path(sys.executable).parent / "ffmpeg"
    return str(local) if local.exists() else "ffmpeg"


def load_pcm16_16k(path: Path) -> bytes:
    """Return mono 16-bit 16 kHz PCM for any of the benchmark's WAV variants.

    The released data is not uniform: 48 kHz and 16 kHz, mono and stereo, 16- and 32-bit.
    ffmpeg is already a stated prerequisite of the benchmark, and the reference client
    normalises the same way (`livekit_inference.py:64-83`, to 48 kHz for WebRTC).
    """
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == SAMPLE_RATE:
                return wf.readframes(wf.getnframes())
    except wave.Error:
        pass

    proc = subprocess.run(
        [_ffmpeg(), "-y", "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode()[-800:]}")
    return proc.stdout


# ---------------------------------------------------------------------------
# one example
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Executes a model tool call the way the benchmark's LiveKit agent would.

    Three outcomes, and the distinction between them is the whole reason this is a class
    and not two lines inline:

    * executed  -- known name, required arguments present. Logged, so it counts for F1.
    * rejected  -- unknown/empty name, or a missing required argument. LiveKit's function
      tool would have refused this before `log_tool_call`, so the published metric never
      sees it. We record it separately and hand the model an error string.
    * errored   -- the mock API itself raised. Should not happen (every mock function takes
      `**kwargs` and returns a literal) but is reported rather than swallowed.
    """

    def __init__(self, tools, registry, response_style: str = "json"):
        self._schemas = {}
        for tool in tools:
            fn = tool.openai_schema["function"]
            params = fn.get("parameters", {})
            self._schemas[fn["name"]] = set(params.get("required", []))
        self._registry = registry
        self._response_style = response_style
        self.executed: List[Dict[str, Any]] = []
        self.rejected: List[Dict[str, Any]] = []

    def __call__(self, name: str, args: Dict[str, Any], t_audio: float) -> tuple[str, bool]:
        """Return ``(response_text, is_error)`` for the function channel."""
        if name not in self._schemas:
            self.rejected.append({"function": name, "args": args, "timestamp_start": t_audio,
                                  "reason": "unknown tool name" if name else "unparseable tool call"})
            return f"Error: no tool named {name!r} is available.", True

        missing = sorted(self._schemas[name] - set(args or {}))
        if missing:
            self.rejected.append({"function": name, "args": args, "timestamp_start": t_audio,
                                  "reason": f"missing required argument(s): {missing}"})
            return f"Error: missing required argument(s) for {name}: {', '.join(missing)}.", True

        t0 = time.time()
        try:
            result = self._registry.call(name, **args)
        except Exception as exc:  # a mock API raising is a harness fault worth surfacing
            self.rejected.append({"function": name, "args": args, "timestamp_start": t_audio,
                                  "reason": f"mock API raised: {exc!r}"})
            return f"Error: {exc}", True
        exec_seconds = time.time() - t0

        self.executed.append({
            "function": name,
            "args": args,
            "timestamp_start": round(t_audio, 2),
            "timestamp_end": round(t_audio + exec_seconds, 2),
        })
        if self._response_style == "sentence":
            # What the model card asks for ("concise, TTS-friendly ASCII sentences"). Kept
            # behind a flag because it is NOT what the other providers in the published
            # table received.
            body = ", ".join(f"{k} is {v}" for k, v in result.items())
            return f"{name} returned: {body}.", False
        return json.dumps(result), False


def run_example(
    provider,
    example_dir: Path,
    metadata: dict,
    tools,
    registry,
    args,
) -> dict:
    """Drive one example end to end and return the `result_{provider}.json` payload."""
    pcm = load_pcm16_16k(example_dir / "input.wav")
    n_samples = len(pcm) // 2
    duration = n_samples / SAMPLE_RATE

    # The horizon is preallocated and inference is O(T^2) in it, so size it to this
    # example rather than paying for the 300 s default on a 40 s file.
    provider.config.max_audio_seconds = duration + TAIL_SILENCE_MS / 1000.0 + 2.0
    provider.config.max_fc_tokens = args.max_fc_tokens

    provider.connect(system_prompt=args.instructions, tools=tools, modality="audio")

    executor = ToolExecutor(tools, registry, response_style=args.tool_response_style)
    frame_bytes = SAMPLE_RATE * FRAME_MS // 1000 * 2
    tail_bytes = b"\x00" * (SAMPLE_RATE * TAIL_SILENCE_MS // 1000 * 2)
    stream = pcm + tail_bytes

    text = ""
    chunks: List[Dict[str, Any]] = []
    first_text_at: Optional[float] = None
    stalls = 0
    wall_start = time.time()
    # A heartbeat matters more here than in most drivers: an example is ~45 s of audio at
    # roughly an order of magnitude slower than realtime, and there is otherwise no output
    # between "session ready" and the final line -- so a wedged shard looks exactly like a
    # slow one for tens of minutes.
    next_report = args.progress_every_seconds

    for offset in range(0, len(stream), frame_bytes):
        if time.time() - wall_start > args.wall_clock_cap_seconds:
            return _payload(
                example_dir, metadata, args, status="wall_clock_timeout", text=text,
                chunks=chunks, first_text_at=first_text_at, executor=executor,
                duration=duration, pushed=offset / 2 / SAMPLE_RATE,
                wall=time.time() - wall_start, stalls=stalls, provider=provider,
            )

        t_audio = offset / 2 / SAMPLE_RATE
        if args.progress_every_seconds and t_audio >= next_report:
            elapsed = time.time() - wall_start
            print(f"    ... {t_audio:5.1f}s/{len(stream) / 2 / SAMPLE_RATE:.1f}s audio, "
                  f"{elapsed / 60:.1f} min wall ({elapsed / max(t_audio, 1e-9):.1f}x realtime), "
                  f"{len(text)} chars, {len(executor.executed)} call(s)", flush=True)
            next_report += args.progress_every_seconds

        delta, calls, stalled = provider.push_user_audio(stream[offset:offset + frame_bytes])

        if delta:
            if first_text_at is None and delta.strip():
                first_text_at = t_audio
            text += delta
            chunks.append({"text": delta, "timestamp": [round(t_audio, 3), round(t_audio, 3)]})
        if stalled:
            stalls += 1

        for call in calls:
            response, is_error = executor(call.name, call.arguments or {}, t_audio)
            provider.send_tool_result(call.id, response, is_error=is_error)
            if args.verbose:
                mark = "REJECTED" if is_error else "ok"
                print(f"    [{t_audio:6.2f}s] {call.name}({call.arguments}) -> {mark}", flush=True)

    return _payload(
        example_dir, metadata, args, status="completed", text=text, chunks=chunks,
        first_text_at=first_text_at, executor=executor, duration=duration,
        pushed=len(stream) / 2 / SAMPLE_RATE, wall=time.time() - wall_start,
        stalls=stalls, provider=provider,
    )


def _payload(example_dir, metadata, args, *, status, text, chunks, first_text_at,
             executor, duration, pushed, wall, stalls, provider) -> dict:
    speaker_id = example_dir.name.rsplit("_", 1)[-1]
    payload = {
        # -- fields the benchmark's evaluators read -------------------------------
        "example_id": metadata["id"],
        "pid": speaker_id,
        "category": metadata.get("domain", "unknown"),
        "title": metadata.get("title", ""),
        "provider": args.provider,
        "evaluated_at": datetime.datetime.now().isoformat(),
        "status": status,
        "transcript": text.strip(),
        "asr_chunks": chunks,
        "actual_tool_calls": executor.executed,
        # -- ours: diagnostics and honest labelling -------------------------------
        "notes": {
            "harness": "nemo-voice-agent/scripts/fdb_v3_nemo_infer.py (local replay, no LiveKit)",
            "transcript_source": "model text channel, NOT ASR of synthesised speech",
            "clock": "timestamps are audio-clock seconds (samples pushed / 16000)",
            "tool_call_logging": "actual_tool_calls holds only calls that would execute "
                                 "under LiveKit; see rejected_tool_calls",
            "tool_response_style": args.tool_response_style,
            "prompt_style": args.prompt_style,
            "system_message": args.system_message,
            "fc_prompt_protocol": bool(provider.config.fc_prompt_protocol),
        },
        "rejected_tool_calls": executor.rejected,
        "input_duration_s": round(duration, 3),
        "audio_pushed_s": round(pushed, 3),
        "inference_wall_s": round(wall, 1),
        "realtime_factor": round(wall / pushed, 2) if pushed else None,
        "tool_stall_ticks": stalls,
        "session_stats": provider.stats(),
    }
    if first_text_at is not None:
        payload["audio_agent_speech_start"] = round(first_text_at, 3)
    return payload


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def discover(root: Path) -> List[Path]:
    """Example directories, in the stable sorted order the shard split relies on."""
    import re

    pattern = re.compile(r"^(.+)_([0-9a-f]{24})$")
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and pattern.match(d.name) and (d / "input.wav").exists()
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=FDB_V3_DIR / "fdb_v3_data_released")
    p.add_argument("--provider", default="nemo",
                   help="Name in result_<provider>.json. Use a distinct name per variant "
                        "so two configurations never overwrite each other's results.")
    p.add_argument("--cascaded-config", default="nemo-base",
                   help="NeMoDuplexConfig preset (tau2 voice/audio_native/nemo/config.py). "
                        "nemo-base = released checkpoint with NVIDIA's tool protocol; "
                        "nemo-base-bare-prompt = the control with the protocol stripped.")
    p.add_argument("--tool-response-style", choices=("json", "sentence"), default="json",
                   help="json (default) is what the benchmark's LiveKit agent returns. "
                        "sentence follows the model card's TTS-friendly recommendation.")
    p.add_argument("--prompt-style", choices=("nvidia_template", "tau2_provider"),
                   default="nvidia_template",
                   help="nvidia_template (default) renders the system prompt through "
                        "NVIDIA's own function_calling/template.jinja -- flattened tools, "
                        "no 'type' key, sorted JSON keys. tau2_provider uses our "
                        "Tool.openai_schema block, which is what a tau2 run sends; kept "
                        "only to measure what the format difference is worth.")
    p.add_argument("--system-message", choices=("benchmark", "nvidia+benchmark"),
                   default="benchmark",
                   help="benchmark (default) = the FDB-v3 VoiceAgent instructions alone, "
                        "which is what every provider in the published table got. "
                        "nvidia+benchmark prepends NVIDIA's DEFAULT_SYSTEM_MESSAGE.")
    p.add_argument("--max-fc-tokens", type=int, default=2000,
                   help="LM positions reserved for function tokens. Adds to the "
                        "preallocated horizon, which inference is O(T^2) in, so the model "
                        "default of 12000 is a large and pointless cost for <=3 calls.")
    p.add_argument("--difficulty", action="append", default=None,
                   choices=("easy", "medium", "hard"),
                   help="Restrict to these difficulty buckets (repeatable). Filtered BEFORE "
                        "sharding so the split stays balanced over the subset. The whole "
                        "residual Tool Selection gap lives in `hard` "
                        "(FDB_V3_REPRODUCTION.md, the discrepancy audit), so `--difficulty "
                        "hard` is 30 examples that carry all of the signal.")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Stop after N examples (debugging).")
    p.add_argument("--force", action="store_true", help="Re-run examples that already have a result.")
    p.add_argument("--wall-clock-cap-seconds", type=float, default=3600.0)
    p.add_argument("--progress-every-seconds", type=float, default=10.0,
                   help="Heartbeat interval in audio seconds. 0 disables it.")
    p.add_argument("--verbose", action="store_true", help="Print each tool call as it happens.")
    args = p.parse_args()

    sys.path.insert(0, str(TAU2_SRC))
    sys.path.insert(0, str(HERE))

    from fdb_v3_tools import (
        build_registry,
        build_tools,
        extract_instructions,
        nvidia_default_system_message,
        render_system_prompt,
    )
    from tau2.voice.audio_native.nemo.config import get_nemo_config
    from tau2.voice.audio_native.nemo.provider import NeMoDuplexProvider

    tools = build_tools()
    registry = build_registry(latency_profile="instant")

    system_message = extract_instructions()
    if args.system_message == "nvidia+benchmark":
        system_message = f"{nvidia_default_system_message()}\n\n{system_message}"

    examples = discover(args.data_dir)
    if args.difficulty:
        wanted = set(args.difficulty)
        kept = [d for d in examples
                if json.loads((d / "metadata.json").read_text())["difficulty"] in wanted]
        print(f"difficulty filter {sorted(wanted)}: {len(kept)} of {len(examples)} examples",
              flush=True)
        examples = kept
    if args.num_shards > 1:
        examples = examples[args.shard::args.num_shards]
    todo = [d for d in examples if args.force or not (d / f"result_{args.provider}.json").exists()]
    if args.limit:
        todo = todo[: args.limit]

    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} of {len(examples)} examples to run "
          f"({len(examples) - len(todo)} already have result_{args.provider}.json)", flush=True)
    if not todo:
        return 0

    config = get_nemo_config(args.cascaded_config)
    if args.prompt_style == "nvidia_template":
        # Render once: the prompt is identical for every example, and the provider must not
        # append a tool block of its own on top of the one the template already wrote.
        args.instructions = render_system_prompt(system_message, tools)
        config.system_prompt_verbatim = True
    else:
        args.instructions = system_message
        config.system_prompt_verbatim = False
    print(f"prompt: style={args.prompt_style} system_message={args.system_message} "
          f"{len(args.instructions)} chars, ascii={args.instructions.isascii()}", flush=True)

    provider = NeMoDuplexProvider(config)

    failures = 0
    for i, example_dir in enumerate(todo, 1):
        metadata = json.loads((example_dir / "metadata.json").read_text())
        print(f"[{i}/{len(todo)}] {example_dir.name} "
              f"({metadata['difficulty']}, {metadata['num_expected_calls']} expected call(s))",
              flush=True)
        try:
            payload = run_example(provider, example_dir, metadata, tools, registry, args)
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
        finally:
            provider.disconnect()

        (example_dir / f"result_{args.provider}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        calls = [c["function"] for c in payload.get("actual_tool_calls", [])]
        expected = [c["function"] for c in metadata["expected_tool_calls"]]
        print(f"    status={payload['status']} rtf={payload.get('realtime_factor')} "
              f"expected={expected} got={calls} "
              f"rejected={len(payload.get('rejected_tool_calls', []))}", flush=True)
        if args.verbose:
            print(f"    transcript: {payload.get('transcript', '')[:400]}", flush=True)

    print(f"shard {args.shard}: done, {failures} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
