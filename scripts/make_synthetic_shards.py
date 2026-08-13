#!/usr/bin/env python3
"""
Generate structurally-valid synthetic Lhotse Shar data for DuplexSTTModel training.

Purpose: de-risk a training run WITHOUT waiting for real tau2 audio. This exercises the parts
that inference cannot — backward pass, optimizer step, DDP, gradient bucketing, and peak memory
at realistic sequence lengths. The audio is noise and the text is filler, so the LOSS VALUES ARE
MEANINGLESS; only "does a step complete, and how much memory does it take" is meaningful here.

This is deliberately NOT prepare_lhotse_data.py. That script targets real tau2 conversations and
its create_lhotse_cutset() is a known skeleton (it attaches only the first turn's audio while
declaring the full conversation duration, so the cut is internally inconsistent). Rather than
half-fix it against fake data, this writes the contract DuplexS2SDataset actually reads:

  source_audio  <- cut.recording                    resampled to data.source_sample_rate (16k)
  target_audio  <- cut.custom['target_audio']       resampled to data.target_sample_rate (22050)
  system prompt <- cut.custom['system_prompt']
  turns         <- cut.supervisions, speaker in data.input_roles / data.output_roles,
                   with start/duration laid out along one shared timeline

Both recordings span the whole conversation; each role's audio is non-silent only during its own
turns, which is what makes the duplex timeline meaningful.

Usage:
  python scripts/make_synthetic_shards.py --output_dir /tmp/synth --num_conversations 64
"""

import argparse
import os

import numpy as np
import soundfile as sf

SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one."
)

# Filler turns. Length matters (it sets token count per turn); content does not.
USER_LINES = [
    "Hi, I need to transfer five hundred dollars to my checking account please.",
    "Can you tell me what my current balance is on the savings account?",
    "I would like to book a hotel room in Boston for next Tuesday night.",
    "Actually, could you make that a double room instead of a single?",
    "What is the cancellation policy on that reservation?",
    "Thanks, that is all I needed for today.",
]
AGENT_LINES = [
    "Of course, I can help you with that. Let me verify your identity first.",
    "Your current balance is two thousand four hundred and thirty dollars.",
    "I found several options available for that date. The Harbor Inn has rooms open.",
    "No problem, I have updated that to a double room for you.",
    "You can cancel free of charge up to twenty four hours before check in.",
    "You're very welcome. Have a great day!",
]


def synth_speechlike(duration_s, sample_rate, rng):
    """Noise shaped to sit in the speech band, with an amplitude envelope.

    Plain white noise is fine for a memory test, but band-limited noise with syllable-rate
    modulation keeps the mel filterbank and the encoder's normalisation in a realistic range,
    so activation memory is representative rather than accidentally tiny.
    """
    n = int(duration_s * sample_rate)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    sig = rng.standard_normal(n).astype(np.float32)
    # Cheap band-pass: difference-of-smoothing, ~300 Hz to ~3.4 kHz.
    def smooth(x, w):
        w = max(1, int(w))
        k = np.ones(w, dtype=np.float32) / w
        return np.convolve(x, k, mode="same")
    sig = smooth(sig, sample_rate / 3400) - smooth(sig, sample_rate / 300)
    # ~4 Hz syllable envelope, never fully silent so the encoder sees continuous speech.
    t = np.arange(n, dtype=np.float32) / sample_rate
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6.28))
    sig = sig * env
    peak = np.abs(sig).max()
    return (0.25 * sig / peak).astype(np.float32) if peak > 0 else sig


def build_conversation(idx, audio_dir, num_turns, seconds_per_turn, src_sr, tgt_sr, rng):
    """Lay alternating user/agent turns on one timeline; write one recording per role."""
    from lhotse import Recording, SupervisionSegment

    total_s = num_turns * seconds_per_turn
    src = np.zeros(int(total_s * src_sr), dtype=np.float32)
    tgt = np.zeros(int(total_s * tgt_sr), dtype=np.float32)

    supervisions = []
    for turn_i in range(num_turns):
        is_user = turn_i % 2 == 0
        role = "user" if is_user else "agent"
        text = (USER_LINES if is_user else AGENT_LINES)[turn_i // 2 % len(USER_LINES)]
        start = turn_i * seconds_per_turn
        # Leave a small gap so turns do not butt up against each other.
        dur = seconds_per_turn * 0.9

        sr = src_sr if is_user else tgt_sr
        buf = src if is_user else tgt
        seg = synth_speechlike(dur, sr, rng)
        a = int(start * sr)
        buf[a:a + len(seg)] = seg[: max(0, len(buf) - a)]

        supervisions.append(
            SupervisionSegment(
                id=f"conv{idx:05d}_t{turn_i:02d}_{role}",
                recording_id=f"conv{idx:05d}",
                start=start,
                duration=dur,
                text=text,
                speaker=role,
            )
        )

    src_path = os.path.join(audio_dir, f"conv{idx:05d}_source.wav")
    tgt_path = os.path.join(audio_dir, f"conv{idx:05d}_target.wav")
    sf.write(src_path, src, src_sr)
    sf.write(tgt_path, tgt, tgt_sr)

    source_rec = Recording.from_file(src_path, recording_id=f"conv{idx:05d}")
    target_rec = Recording.from_file(tgt_path, recording_id=f"conv{idx:05d}_target")
    return source_rec, target_rec, supervisions, total_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_conversations", type=int, default=64)
    p.add_argument("--num_turns", type=int, default=6, help="Alternating user/agent turns")
    p.add_argument("--seconds_per_turn", type=float, default=4.0)
    p.add_argument("--source_sample_rate", type=int, default=16000)
    p.add_argument("--target_sample_rate", type=int, default=22050)
    p.add_argument("--shard_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from lhotse import CutSet, MonoCut

    rng = np.random.default_rng(args.seed)
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    conv_s = args.num_turns * args.seconds_per_turn
    print(f"Generating {args.num_conversations} conversations x {conv_s:.1f}s "
          f"({args.num_turns} turns @ {args.seconds_per_turn}s)")

    cuts = []
    for i in range(args.num_conversations):
        source_rec, target_rec, sups, total_s = build_conversation(
            i, audio_dir, args.num_turns, args.seconds_per_turn,
            args.source_sample_rate, args.target_sample_rate, rng,
        )
        cuts.append(
            MonoCut(
                id=f"conv{i:05d}",
                start=0.0,
                duration=total_s,
                channel=0,
                recording=source_rec,
                supervisions=sups,
                # The dataset reads the agent stream from custom['target_audio'] and the prompt
                # from custom['system_prompt'] (s2s_dataset.py ~L1351 and ~L2150).
                custom={
                    "target_audio": target_rec,
                    "system_prompt": SYSTEM_PROMPT,
                    "total_turns": args.num_turns,
                },
            )
        )

    cutset = CutSet.from_cuts(cuts)
    shar_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shar_dir, exist_ok=True)
    cutset.to_shar(
        shar_dir,
        fields={"recording": "wav", "target_audio": "wav"},
        shard_size=args.shard_size,
    )
    total_audio_s = sum(c.duration for c in cuts)
    print(f"Wrote {len(cuts)} cuts ({total_audio_s / 60:.1f} min audio) to {shar_dir}")
    print(f"  shard files: {sorted(os.listdir(shar_dir))[:6]}")


if __name__ == "__main__":
    main()
