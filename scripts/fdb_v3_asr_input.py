#!/usr/bin/env python3
"""Add `user_speech_end_rel` to existing FDB-v3 result files, so latency can be scored.

`user_speech_end_rel` is the one latency input that has nothing to do with the agent: the
reference pipeline gets it by running Parakeet over the *input* audio and taking the end of
the user's first turn (`run_tool_benchmark.py:342-365`, first inter-word gap > 2 s, else the
last word's end). That makes it reproducible exactly, and separable -- which is why it lives
here instead of behind a flag in the inference driver:

  * The inference shards each hold an 11B model on a GPU. Loading a second 0.6B ASR model
    into all eight of them, to compute a number that does not depend on the agent, would be
    pure waste.
  * It can be re-run over results that already exist, in one pass on one GPU, in the time it
    takes the driver to do a couple of examples.

Without it, `evaluate_tool_calls.py:320-326` and `analyze_tool_latency.py:126` skip the
sample rather than fabricate a zero, and the latency section of the report is empty.

Caveat that no amount of ASR fixes: our agent-side timestamp is *first text token*, not
first audio sample, because this checkpoint's agent channel is text (`tokens_audio: None`).
So the latency numbers this unlocks are "time to first token on the audio clock", not the
published "time to first speech". Directionally useful, not comparable.

Usage (inside a GPU allocation):
    python scripts/fdb_v3_asr_input.py --provider nemo
    python scripts/fdb_v3_asr_input.py --provider nemo --dry-run   # what it would touch
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FDB_V3_DIR = Path("/fsx/home/kai.li/code/Full-Duplex-Bench/v3")
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"  # run_tool_benchmark.py:61
TURN_GAP_S = 2.0  # run_tool_benchmark.py:358


def _ffmpeg() -> str:
    local = Path(sys.executable).parent / "ffmpeg"
    return str(local) if local.exists() else "ffmpeg"


def speech_end(chunks) -> float:
    """End of the user's first turn, by the benchmark's own rule."""
    if not chunks:
        return 0.0
    for cur, nxt in zip(chunks, chunks[1:]):
        if nxt["timestamp"][0] - cur["timestamp"][1] > TURN_GAP_S:
            return cur["timestamp"][1]
    return chunks[-1]["timestamp"][1]


def transcribe(model, wav: Path, tmp: Path):
    """Mono 16 kHz, then word-level timestamps in the reference's chunk format."""
    mono = tmp / "input_mono.wav"
    # The reference writes input_mono.wav into the example folder; we keep the benchmark
    # checkout clean. It also leaves the sample rate alone -- Parakeet is a 16 kHz model and
    # NeMo would resample internally, so doing it here is explicit, not different.
    proc = subprocess.run(
        [_ffmpeg(), "-y", "-i", str(wav), "-ac", "1", "-ar", "16000", str(mono)],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {wav}: {proc.stderr.decode()[-500:]}")

    out = model.transcribe([str(mono)], timestamps=True)
    if not out:
        return "", []
    result = out[0]
    chunks = [
        {"text": w["word"], "timestamp": [w["start"], w["end"]]}
        for w in getattr(result, "timestamp", {}).get("word", [])
    ]
    text = " ".join(c["text"] for c in chunks).strip() or getattr(result, "text", "")
    return text, chunks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default="nemo")
    p.add_argument("--data-dir", type=Path, default=FDB_V3_DIR / "fdb_v3_data_released")
    p.add_argument("--force", action="store_true", help="Re-transcribe files that already have it.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    results = sorted(args.data_dir.glob(f"*/result_{args.provider}.json"))
    todo = []
    for r in results:
        data = json.loads(r.read_text())
        if args.force or data.get("user_speech_end_rel") is None:
            todo.append(r)
    print(f"{len(todo)} of {len(results)} result_{args.provider}.json files need ASR", flush=True)
    if args.dry_run or not todo:
        for r in todo[:5]:
            print(f"   would transcribe {r.parent.name}")
        return 0

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    if hasattr(model, "cuda"):
        model = model.cuda()

    failures = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i, r in enumerate(todo, 1):
            data = json.loads(r.read_text())
            try:
                text, chunks = transcribe(model, r.parent / "input.wav", tmp)
            except Exception as exc:
                print(f"[{i}/{len(todo)}] {r.parent.name}: FAILED {exc}", flush=True)
                failures += 1
                continue
            end = speech_end(chunks)
            data["input_transcript"] = text
            data["input_asr_chunks"] = chunks
            data["user_speech_end_rel"] = end
            data.setdefault("notes", {})["user_speech_end_rel_source"] = (
                f"{ASR_MODEL_NAME} over input.wav, first-turn rule from "
                f"run_tool_benchmark.py:352-363"
            )
            r.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            agent_start = data.get("audio_agent_speech_start")
            lat = f"{agent_start - end:+.2f}s" if agent_start is not None else "no agent output"
            print(f"[{i}/{len(todo)}] {r.parent.name}: user ends {end:.2f}s, first token {lat}",
                  flush=True)

    print(f"done, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
