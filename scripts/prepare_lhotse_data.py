#!/usr/bin/env python3
"""
Prepare training data in Lhotse Shar format for DuplexSTTModel.

This script converts text conversations (from tau2-bench oracle trajectories or other sources)
into the Lhotse CutSet format expected by the NeMo speechlm2 data pipeline.

Pipeline:
  1. Load text conversations (JSON format)
  2. Synthesize per-turn audio (TTS)
  3. Build two full-conversation recordings:
     - source recording (user turns, 16kHz, silence during agent turns)
     - target recording (agent turns, 22050Hz, silence during user turns)
  4. Create Lhotse CutSet with speaker-annotated supervisions + custom["target_audio"]
  5. Export as Lhotse Shar (sharded format)

Input format (conversations.json):
[
  {
    "id": "conversation_001",
    "turns": [
      {"role": "user", "text": "I need to transfer $500 to my checking account"},
      {"role": "agent", "text": "I can help you with that. Let me verify your identity first."},
      ...
    ],
    "system_prompt": "You are a helpful banking assistant..."
  },
  ...
]

Output: Lhotse Shar directory ready for training with DuplexS2SDataset.

The dataset expects:
  - cut.recording = source (user) audio at source_sample_rate
  - cut.custom["target_audio"] = Recording of agent audio at target_sample_rate
  - cut.custom["system_prompt"] = system prompt text
  - cut.supervisions = time-ordered SupervisionSegments with speaker="user"/"agent"

Usage:
  python scripts/prepare_lhotse_data.py \
      --input conversations.json \
      --output_dir /path/to/lhotse_shards \
      --tts_backend edge_tts \
      --shard_size 100
"""

import argparse
import io
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# Voices for diversity
USER_VOICES = [
    "en-US-GuyNeural",
    "en-US-ChristopherNeural",
    "en-US-EricNeural",
    "en-US-AndrewNeural",
    "en-GB-RyanNeural",
]

AGENT_VOICE = "en-US-JennyNeural"


@dataclass
class SynthesizedTurn:
    role: str
    text: str
    audio: np.ndarray  # shape: (samples,)
    sample_rate: int
    duration: float


def synthesize_edge_tts(text: str, voice: str, target_sr: int) -> tuple[np.ndarray, float]:
    """Synthesize speech using Edge TTS. Returns (audio_array, duration)."""
    import asyncio
    import tempfile
    import edge_tts

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    async def _synthesize():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)

    asyncio.run(_synthesize())

    audio, sr = sf.read(tmp_path, dtype="float32")
    os.unlink(tmp_path)

    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import torch
        import torchaudio
        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        audio = waveform[0].numpy()

    duration = len(audio) / target_sr
    return audio, duration


def synthesize_dummy(text: str, sample_rate: int) -> tuple[np.ndarray, float]:
    """Generate dummy near-silence audio. ~150ms per word."""
    words = len(text.split())
    duration = max(0.5, words * 0.15)
    samples = int(duration * sample_rate)
    audio = np.random.randn(samples).astype(np.float32) * 0.001
    return audio, duration


def synthesize_conversation(
    conv: dict,
    source_sr: int = 16000,
    target_sr: int = 22050,
    tts_backend: str = "dummy",
    gap_between_turns: float = 0.3,
    user_voice: Optional[str] = None,
) -> list[SynthesizedTurn]:
    """Synthesize audio for each turn in a conversation."""
    turns = conv["turns"]
    results = []

    if user_voice is None:
        user_voice = random.choice(USER_VOICES)

    for turn in turns:
        role = turn["role"]
        text = turn["text"].strip()
        if not text:
            continue

        sr = source_sr if role == "user" else target_sr

        if tts_backend == "edge_tts":
            voice = user_voice if role == "user" else AGENT_VOICE
            audio, duration = synthesize_edge_tts(text, voice, sr)
        else:
            audio, duration = synthesize_dummy(text, sr)

        results.append(SynthesizedTurn(
            role=role,
            text=text,
            audio=audio,
            sample_rate=sr,
            duration=duration,
        ))

    return results


def build_lhotse_cut(
    conv_id: str,
    synthesized_turns: list[SynthesizedTurn],
    system_prompt: str,
    source_sr: int = 16000,
    target_sr: int = 22050,
    gap_between_turns: float = 0.3,
):
    """
    Build a Lhotse MonoCut with:
      - recording = full source (user) audio at source_sr
      - custom["target_audio"] = Recording of full target (agent) audio at target_sr
      - custom["system_prompt"] = system prompt
      - supervisions = time-ordered segments for each turn

    Both recordings span the entire conversation duration.
    User turns have audio in the source recording (silence in target),
    and vice versa for agent turns.
    """
    from lhotse import MonoCut, Recording, SupervisionSegment
    from lhotse.audio import AudioSource

    # Compute timeline: each turn occupies [start, start+duration], with gaps between
    timeline = []
    current_time = 0.0
    for turn in synthesized_turns:
        timeline.append((current_time, turn.duration))
        current_time += turn.duration + gap_between_turns

    total_duration = current_time - gap_between_turns if synthesized_turns else 0.0
    if total_duration <= 0:
        return None

    # Build source (user) recording: user audio at correct positions, silence elsewhere
    source_samples = int(total_duration * source_sr)
    source_audio = np.zeros(source_samples, dtype=np.float32)

    # Build target (agent) recording: agent audio at correct positions, silence elsewhere
    target_samples = int(total_duration * target_sr)
    target_audio = np.zeros(target_samples, dtype=np.float32)

    supervisions = []

    for i, (turn, (start_time, duration)) in enumerate(zip(synthesized_turns, timeline)):
        if turn.role == "user":
            # Place user audio in source recording
            start_sample = int(start_time * source_sr)
            end_sample = start_sample + len(turn.audio)
            end_sample = min(end_sample, source_samples)
            actual_len = end_sample - start_sample
            source_audio[start_sample:end_sample] = turn.audio[:actual_len]
        else:
            # Place agent audio in target recording
            start_sample = int(start_time * target_sr)
            end_sample = start_sample + len(turn.audio)
            end_sample = min(end_sample, target_samples)
            actual_len = end_sample - start_sample
            target_audio[start_sample:end_sample] = turn.audio[:actual_len]

        sup = SupervisionSegment(
            id=f"{conv_id}_turn{i:03d}_{turn.role}",
            recording_id=conv_id,
            start=start_time,
            duration=duration,
            text=turn.text,
            speaker=turn.role,
        )
        supervisions.append(sup)

    # Create source Recording from array
    source_recording = _create_recording_from_array(
        source_audio.reshape(1, -1), source_sr, conv_id
    )

    # Create target Recording from array
    target_recording = _create_recording_from_array(
        target_audio.reshape(1, -1), target_sr, f"{conv_id}_target"
    )

    # Build MonoCut
    cut = MonoCut(
        id=conv_id,
        start=0.0,
        duration=total_duration,
        channel=0,
        recording=source_recording,
        supervisions=supervisions,
        custom={
            "target_audio": target_recording,
            "system_prompt": system_prompt,
        },
    )

    return cut


def _create_recording_from_array(samples: np.ndarray, sampling_rate: int, recording_id: str):
    """Create a Lhotse Recording from a numpy array. Mirrors NeMo's implementation."""
    from lhotse import Recording

    buf = io.BytesIO()
    sf.write(buf, samples.T, samplerate=sampling_rate, format='WAV')
    buf.seek(0)
    return Recording.from_bytes(buf.read(), recording_id=recording_id)


def process_conversations(
    conversations: list[dict],
    source_sr: int = 16000,
    target_sr: int = 22050,
    tts_backend: str = "dummy",
    gap_between_turns: float = 0.3,
    user_voice: Optional[str] = None,
):
    """Process all conversations into Lhotse MonoCuts."""
    from lhotse import CutSet

    cuts = []
    for i, conv in enumerate(conversations):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(conversations)}...")

        synthesized = synthesize_conversation(
            conv,
            source_sr=source_sr,
            target_sr=target_sr,
            tts_backend=tts_backend,
            gap_between_turns=gap_between_turns,
            user_voice=user_voice,
        )

        if not synthesized:
            print(f"  WARNING: No turns synthesized for {conv['id']}, skipping")
            continue

        cut = build_lhotse_cut(
            conv_id=conv["id"],
            synthesized_turns=synthesized,
            system_prompt=conv.get("system_prompt", ""),
            source_sr=source_sr,
            target_sr=target_sr,
            gap_between_turns=gap_between_turns,
        )

        if cut is not None:
            cuts.append(cut)

    return CutSet.from_cuts(cuts)


def export_to_shar(cutset, output_dir: str, shard_size: int = 100) -> str:
    """Export CutSet to Lhotse Shar format with both source and target audio."""
    shar_dir = os.path.join(output_dir, "shards")
    os.makedirs(shar_dir, exist_ok=True)

    cutset.to_shar(
        shar_dir,
        fields={"recording": "wav", "target_audio": "wav"},
        shard_size=shard_size,
    )
    print(f"  Exported {len(cutset)} cuts to {shar_dir}")
    return shar_dir


def validate_cut(cut, source_sr: int, target_sr: int, frame_length: float = 0.08):
    """Validate a single cut matches DuplexS2SDataset expectations."""
    issues = []

    # Check recording exists and has correct sample rate
    if cut.recording is None:
        issues.append("Missing source recording")
    elif cut.recording.sampling_rate != source_sr:
        issues.append(f"Source SR: {cut.recording.sampling_rate} != {source_sr}")

    # Check target_audio
    if "target_audio" not in cut.custom:
        issues.append("Missing custom['target_audio']")
    else:
        target_rec = cut.custom["target_audio"]
        if target_rec.sampling_rate != target_sr:
            issues.append(f"Target SR: {target_rec.sampling_rate} != {target_sr}")

    # Check supervisions have speaker field
    for sup in cut.supervisions:
        if sup.speaker not in ("user", "agent"):
            issues.append(f"Supervision {sup.id}: unexpected speaker '{sup.speaker}'")
        if not sup.text or not sup.text.strip():
            issues.append(f"Supervision {sup.id}: empty text")

    # Check system_prompt
    if "system_prompt" not in cut.custom:
        issues.append("Missing custom['system_prompt']")

    # Check frame alignment feasibility
    num_frames = int(cut.duration / frame_length)
    if num_frames < 2:
        issues.append(f"Too short: {cut.duration:.2f}s = {num_frames} frames")

    return issues


def print_cut_summary(cut, source_sr: int, target_sr: int):
    """Print a human-readable summary of a cut for review."""
    print(f"\n{'─' * 60}")
    print(f"Cut ID: {cut.id}")
    print(f"Duration: {cut.duration:.2f}s")
    print(f"Source recording: {cut.recording.sampling_rate}Hz, {cut.recording.num_samples} samples")
    target = cut.custom.get("target_audio")
    if target:
        print(f"Target recording: {target.sampling_rate}Hz, {target.num_samples} samples")
    print(f"System prompt: {cut.custom.get('system_prompt', '')[:80]}...")
    print(f"Supervisions ({len(cut.supervisions)}):")
    for sup in cut.supervisions:
        print(f"  [{sup.start:.2f}s - {sup.start + sup.duration:.2f}s] "
              f"{sup.speaker:>5}: {sup.text[:60]}{'...' if len(sup.text) > 60 else ''}")
    issues = validate_cut(cut, source_sr, target_sr)
    if issues:
        print(f"  ISSUES: {issues}")
    else:
        print(f"  ✓ Valid")
    print(f"{'─' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Lhotse Shar training data")
    parser.add_argument("--input", required=True, help="Path to conversations JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for Lhotse Shar data")
    parser.add_argument("--tts_backend", default="dummy", choices=["dummy", "edge_tts"],
                        help="TTS backend for audio synthesis")
    parser.add_argument("--user_voice", default=None,
                        help="Edge TTS voice for user (default: random per conversation)")
    parser.add_argument("--source_sr", type=int, default=16000, help="User audio sample rate")
    parser.add_argument("--target_sr", type=int, default=22050, help="Agent audio sample rate")
    parser.add_argument("--gap", type=float, default=0.3, help="Gap between turns in seconds")
    parser.add_argument("--shard_size", type=int, default=100, help="Number of cuts per shard")
    parser.add_argument("--max_conversations", type=int, default=None,
                        help="Limit number of conversations (for testing)")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate existing shar, don't generate")
    parser.add_argument("--preview", type=int, default=0,
                        help="Print detailed summaries for N cuts after generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("LHOTSE DATA PREPARATION")
    print("=" * 60)
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output_dir}")
    print(f"  TTS backend: {args.tts_backend}")
    print(f"  Source SR: {args.source_sr} Hz")
    print(f"  Target SR: {args.target_sr} Hz")
    print(f"  Gap between turns: {args.gap}s")

    # Load conversations
    print(f"\nLoading conversations...")
    with open(args.input) as f:
        conversations = json.load(f)
    if args.max_conversations:
        conversations = conversations[:args.max_conversations]
    print(f"  Loaded {len(conversations)} conversations")

    # Process
    print(f"\nSynthesizing audio and building cuts...")
    t0 = time.time()
    cutset = process_conversations(
        conversations,
        source_sr=args.source_sr,
        target_sr=args.target_sr,
        tts_backend=args.tts_backend,
        gap_between_turns=args.gap,
        user_voice=args.user_voice,
    )
    elapsed = time.time() - t0
    print(f"  Built {len(cutset)} cuts in {elapsed:.1f}s")

    # Validate
    print(f"\nValidating cuts...")
    valid = 0
    invalid = 0
    for cut in cutset:
        issues = validate_cut(cut, args.source_sr, args.target_sr)
        if issues:
            invalid += 1
            if invalid <= 3:
                print(f"  INVALID {cut.id}: {issues}")
        else:
            valid += 1
    print(f"  Valid: {valid}, Invalid: {invalid}")

    if invalid > 0:
        print(f"  WARNING: {invalid} cuts have issues!")

    # Preview
    if args.preview > 0:
        print(f"\nPreviewing first {args.preview} cuts:")
        for cut in list(cutset)[:args.preview]:
            print_cut_summary(cut, args.source_sr, args.target_sr)

    # Export
    print(f"\nExporting to Lhotse Shar format...")
    shar_dir = export_to_shar(cutset, args.output_dir, shard_size=args.shard_size)

    # Save metadata
    total_duration = sum(cut.duration for cut in cutset)
    metadata = {
        "num_conversations": len(conversations),
        "num_cuts": len(cutset),
        "total_duration_hours": total_duration / 3600,
        "source_sample_rate": args.source_sr,
        "target_sample_rate": args.target_sr,
        "tts_backend": args.tts_backend,
        "gap_between_turns": args.gap,
        "shar_path": shar_dir,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save CutSet manifest for inspection (may fail if recordings are in-memory bytes)
    cutset_path = os.path.join(args.output_dir, "cuts.jsonl.gz")
    try:
        cutset.to_file(cutset_path)
        print(f"  CutSet manifest: {cutset_path}")
    except TypeError:
        print(f"  (Skipped CutSet manifest — in-memory recordings not JSON-serializable)")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"\n  Output directory: {args.output_dir}")
    print(f"  Shar path: {shar_dir}")
    print(f"  Total duration: {total_duration:.1f}s ({total_duration/3600:.2f}h)")
    print(f"  Average duration: {total_duration/len(cutset):.1f}s per cut")
    print(f"\n  For training config:")
    print(f"    data.train_ds.input_cfg.0.shar_path={shar_dir}")


if __name__ == "__main__":
    main()
