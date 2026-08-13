#!/usr/bin/env python3
"""
Extract STT model weights from the combined NemotronVoiceChat checkpoint.

The released checkpoint (nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) contains both
stt_model.* and tts_model.* weights in a single model.safetensors file.
DuplexSTTModel expects weights WITHOUT the 'stt_model.' prefix.

This script:
  1. Loads the combined checkpoint
  2. Extracts keys prefixed with 'stt_model.'
  3. Strips the prefix
  4. Saves as a new model.safetensors in HuggingFace format

Usage:
  python scripts/extract_stt_checkpoint.py \
      --input_dir /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
      --output_dir /path/to/stt_checkpoint

  Then use the output_dir as `model.pretrained_s2s_model` in the training config.
"""

import argparse
import json
import os
import time

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Extract STT weights from combined NemotronVoiceChat checkpoint")
    parser.add_argument("--input_dir", required=True, help="Path to combined HF checkpoint directory")
    parser.add_argument("--output_dir", required=True, help="Path to save extracted STT checkpoint")
    parser.add_argument("--also_extract_tts", action="store_true", help="Also extract TTS weights separately")
    args = parser.parse_args()

    input_safetensors = os.path.join(args.input_dir, "model.safetensors")
    if not os.path.exists(input_safetensors):
        raise FileNotFoundError(f"No model.safetensors found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint from: {input_safetensors}")
    t0 = time.time()

    stt_state_dict = {}
    tts_state_dict = {}
    other_keys = []

    with safe_open(input_safetensors, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        print(f"  Total keys in checkpoint: {len(all_keys)}")

        for key in all_keys:
            if key.startswith("stt_model."):
                new_key = key[len("stt_model."):]
                stt_state_dict[new_key] = f.get_tensor(key)
            elif key.startswith("tts_model."):
                if args.also_extract_tts:
                    new_key = key[len("tts_model."):]
                    tts_state_dict[new_key] = f.get_tensor(key)
            else:
                other_keys.append(key)

    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  STT keys: {len(stt_state_dict)}")
    if args.also_extract_tts:
        print(f"  TTS keys: {len(tts_state_dict)}")
    if other_keys:
        print(f"  Other keys (skipped): {other_keys[:10]}{'...' if len(other_keys) > 10 else ''}")

    # Save STT checkpoint
    stt_output_path = os.path.join(args.output_dir, "model.safetensors")
    print(f"\nSaving STT checkpoint to: {stt_output_path}")
    t0 = time.time()
    save_file(stt_state_dict, stt_output_path)
    print(f"  Saved in {time.time() - t0:.1f}s")

    stt_params = sum(t.numel() for t in stt_state_dict.values())
    print(f"  STT total parameters: {stt_params:,} ({stt_params / 1e9:.2f}B)")

    # Copy config.json if it exists (may need modification)
    config_path = os.path.join(args.input_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract the STT-relevant portion of the config if possible
        output_config = {"source": "extracted_from_nemotron_voicechat_11b", "original_config": config}
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(output_config, f, indent=2)
        print(f"  Config saved to {args.output_dir}/config.json")

    # Optionally save TTS checkpoint
    if args.also_extract_tts and tts_state_dict:
        tts_output_dir = args.output_dir + "_tts"
        os.makedirs(tts_output_dir, exist_ok=True)
        tts_output_path = os.path.join(tts_output_dir, "model.safetensors")
        print(f"\nSaving TTS checkpoint to: {tts_output_path}")
        t0 = time.time()
        save_file(tts_state_dict, tts_output_path)
        print(f"  Saved in {time.time() - t0:.1f}s")

        tts_params = sum(t.numel() for t in tts_state_dict.values())
        print(f"  TTS total parameters: {tts_params:,} ({tts_params / 1e9:.2f}B)")

    # Print key structure summary
    print("\n" + "=" * 60)
    print("STT KEY STRUCTURE SUMMARY")
    print("=" * 60)
    prefixes = {}
    for key in stt_state_dict.keys():
        prefix = key.split(".")[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"  {prefix}: {count} tensors")

    print("\nDone! Use this directory as `model.pretrained_s2s_model` in the training config.")


if __name__ == "__main__":
    main()
