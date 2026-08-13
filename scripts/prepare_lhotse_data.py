#!/usr/bin/env python3
"""
Prepare training data in Lhotse Shar format for DuplexSTTModel.

This script converts text conversations (from tau2-bench oracle trajectories or other sources)
into the Lhotse CutSet format expected by the NeMo speechlm2 data pipeline.

Pipeline:
  1. Load text conversations (JSON format)
  2. Synthesize user audio (TTS, 16kHz)
  3. Synthesize agent audio (TTS, 22050Hz)
  4. Create Lhotse CutSet with speaker-annotated supervisions
  5. Export as Lhotse Shar (sharded format)

Input format (conversations.json):
[
  {
    "id": "conversation_001",
    "turns": [
      {"role": "user", "text": "I need to transfer $500 to my checking account"},
      {"role": "agent", "text": "I can help you with that. Let me verify your identity first. ..."},
      ...
    ],
    "system_prompt": "You are a helpful banking assistant..."
  },
  ...
]

Output: Lhotse Shar directory ready for training.

Usage:
  python scripts/prepare_lhotse_data.py \
      --input conversations.json \
      --output_dir /path/to/lhotse_shards \
      --user_tts_model edge_tts \
      --agent_tts_model edge_tts \
      --shard_size 100
"""

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# NOTE: This script requires:
#   pip install lhotse soundfile torchaudio
# For TTS synthesis, one of:
#   pip install edge-tts   (free, Microsoft Edge TTS)
#   pip install TTS        (Coqui TTS, local)


@dataclass
class Turn:
    role: str  # "user" or "agent"
    text: str
    start_time: float = 0.0
    duration: float = 0.0
    audio_path: Optional[str] = None


def synthesize_turn_edge_tts(text: str, output_path: str, voice: str, rate: str = "+0%") -> float:
    """Synthesize speech using Edge TTS (async, requires edge-tts package)."""
    import asyncio
    import edge_tts

    async def _synthesize():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)

    asyncio.run(_synthesize())

    import soundfile as sf
    info = sf.info(output_path)
    return info.duration


def synthesize_turn_dummy(text: str, output_path: str, sample_rate: int) -> float:
    """Generate dummy silence audio (for testing pipeline without TTS)."""
    import numpy as np
    import soundfile as sf

    # Rough estimate: 150ms per word
    words = len(text.split())
    duration = max(0.5, words * 0.15)
    samples = int(duration * sample_rate)
    audio = np.random.randn(samples).astype(np.float32) * 0.001  # near-silence
    sf.write(output_path, audio, sample_rate)
    return duration


def resample_audio(input_path: str, output_path: str, target_sr: int):
    """Resample audio file to target sample rate."""
    import torchaudio

    waveform, sr = torchaudio.load(input_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    torchaudio.save(output_path, waveform, target_sr)


def process_conversation(
    conv: dict,
    output_dir: str,
    user_voice: str = "en-US-GuyNeural",
    agent_voice: str = "en-US-JennyNeural",
    source_sr: int = 16000,
    target_sr: int = 22050,
    tts_backend: str = "dummy",
    gap_between_turns: float = 0.3,
) -> dict:
    """
    Process a single conversation: synthesize audio for each turn and create metadata.

    Returns a dict with all the info needed for Lhotse CutSet creation.
    """
    conv_id = conv["id"]
    turns = conv["turns"]
    system_prompt = conv.get("system_prompt", "")

    conv_audio_dir = os.path.join(output_dir, "audio", conv_id)
    os.makedirs(conv_audio_dir, exist_ok=True)

    processed_turns = []
    current_time = 0.0

    for i, turn in enumerate(turns):
        role = turn["role"]
        text = turn["text"]

        # Skip empty turns
        if not text.strip():
            continue

        # Determine sample rate based on role
        sr = source_sr if role == "user" else target_sr

        # Synthesize audio
        audio_filename = f"turn_{i:03d}_{role}.wav"
        audio_path = os.path.join(conv_audio_dir, audio_filename)

        if tts_backend == "edge_tts":
            voice = user_voice if role == "user" else agent_voice
            duration = synthesize_turn_edge_tts(text, audio_path, voice)
            # Resample to correct sample rate
            resample_audio(audio_path, audio_path, sr)
        else:
            duration = synthesize_turn_dummy(text, audio_path, sr)

        processed_turns.append(Turn(
            role=role,
            text=text,
            start_time=current_time,
            duration=duration,
            audio_path=audio_path,
        ))

        current_time += duration + gap_between_turns

    return {
        "id": conv_id,
        "duration": current_time,
        "turns": processed_turns,
        "system_prompt": system_prompt,
    }


def create_lhotse_cutset(processed_conversations: list, output_dir: str):
    """Create a Lhotse CutSet from processed conversations."""
    from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
    from lhotse.audio import AudioSource

    cuts = []

    for conv in processed_conversations:
        # Create a multi-channel recording (or separate recordings per role)
        # For simplicity, we create one cut per conversation with multiple supervisions
        conv_id = conv["id"]

        # Merge all audio into one recording per channel
        user_turns = [t for t in conv["turns"] if t.role == "user"]
        agent_turns = [t for t in conv["turns"] if t.role == "agent"]

        # Create supervisions for each turn
        supervisions = []
        for turn in conv["turns"]:
            sup = SupervisionSegment(
                id=f"{conv_id}_{turn.role}_{turn.start_time:.3f}",
                recording_id=conv_id,
                start=turn.start_time,
                duration=turn.duration,
                text=turn.text,
                speaker=turn.role,
            )
            supervisions.append(sup)

        # Create a recording that references the individual audio files
        # In practice, you'd concatenate them or use multi-source recordings
        total_duration = conv["duration"]

        # For now, create a placeholder recording
        # The actual implementation should concatenate user/agent audio
        if conv["turns"]:
            first_audio = conv["turns"][0].audio_path
            recording = Recording.from_file(first_audio, recording_id=conv_id)
        else:
            continue

        cut = MonoCut(
            id=conv_id,
            start=0.0,
            duration=total_duration,
            channel=0,
            recording=recording,
            supervisions=supervisions,
        )
        cuts.append(cut)

    cutset = CutSet.from_cuts(cuts)
    return cutset


def export_to_shar(cutset, output_dir: str, shard_size: int = 100):
    """Export CutSet to Lhotse Shar format."""
    shar_dir = os.path.join(output_dir, "shards")
    os.makedirs(shar_dir, exist_ok=True)

    cutset.to_shar(
        shar_dir,
        fields={"recording": "wav"},
        shard_size=shard_size,
    )
    print(f"  Exported {len(cutset)} cuts to {shar_dir}")
    return shar_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare Lhotse Shar training data")
    parser.add_argument("--input", required=True, help="Path to conversations JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for Lhotse Shar data")
    parser.add_argument("--tts_backend", default="dummy", choices=["dummy", "edge_tts"],
                        help="TTS backend for audio synthesis")
    parser.add_argument("--user_voice", default="en-US-GuyNeural", help="Edge TTS voice for user")
    parser.add_argument("--agent_voice", default="en-US-JennyNeural", help="Edge TTS voice for agent")
    parser.add_argument("--source_sr", type=int, default=16000, help="User audio sample rate")
    parser.add_argument("--target_sr", type=int, default=22050, help="Agent audio sample rate")
    parser.add_argument("--shard_size", type=int, default=100, help="Number of cuts per shard")
    parser.add_argument("--max_conversations", type=int, default=None, help="Limit number of conversations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("LHOTSE DATA PREPARATION")
    print("=" * 60)

    # Load conversations
    print(f"\nLoading conversations from: {args.input}")
    with open(args.input) as f:
        conversations = json.load(f)
    if args.max_conversations:
        conversations = conversations[:args.max_conversations]
    print(f"  Loaded {len(conversations)} conversations")

    # Process each conversation
    print(f"\nSynthesizing audio (backend: {args.tts_backend})...")
    processed = []
    for i, conv in enumerate(conversations):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(conversations)}...")
        result = process_conversation(
            conv,
            args.output_dir,
            user_voice=args.user_voice,
            agent_voice=args.agent_voice,
            source_sr=args.source_sr,
            target_sr=args.target_sr,
            tts_backend=args.tts_backend,
        )
        processed.append(result)

    # Create Lhotse CutSet
    print(f"\nCreating Lhotse CutSet...")
    cutset = create_lhotse_cutset(processed, args.output_dir)
    print(f"  Created CutSet with {len(cutset)} cuts")

    # Export to Shar
    print(f"\nExporting to Lhotse Shar format...")
    shar_dir = export_to_shar(cutset, args.output_dir, shard_size=args.shard_size)

    # Save metadata
    metadata = {
        "num_conversations": len(conversations),
        "num_cuts": len(cutset),
        "source_sample_rate": args.source_sr,
        "target_sample_rate": args.target_sr,
        "tts_backend": args.tts_backend,
        "shar_path": shar_dir,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Shar path (for config): {shar_dir}")
    print(f"\nTo use in training config:")
    print(f"  data.train_ds.input_cfg.0.shar_path={shar_dir}")


if __name__ == "__main__":
    main()
