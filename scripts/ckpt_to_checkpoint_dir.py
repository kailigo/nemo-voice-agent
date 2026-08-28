#!/usr/bin/env python
"""
Convert a Lightning .ckpt (produced by s2s_duplex_stt_train.py) into the HF-style
checkpoint directory (config.json + model.safetensors) that
model.pretrained_s2s_model / tau2's nemo provider expects.

The .ckpt's state_dict keys already match DuplexSTTModel.state_dict() 1:1 (verified:
missing=0, unexpected=0 when loaded directly), so this is a pure format conversion --
no key remapping. See [[lora-checkpoint-key-rename-trap]] memory for why that check
matters in general; it does not apply here.

Usage:
  python scripts/ckpt_to_checkpoint_dir.py \
      --ckpt-path logs/sft_train8_0826/exp/checkpoints/step-500.ckpt \
      --template-config /fsx/home/kai.li/data/voicechat/stt_extracted_lora/config.json \
      --output-dir /fsx/home/kai.li/data/voicechat/sft_step500_airline_retail
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--template-config", required=True,
                    help="config.json from an existing checkpoint dir (e.g. stt_extracted_lora); "
                         "copied verbatim since the load path only reads model.safetensors, "
                         "but downstream tooling expects the file to exist.")
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[convert] loading {args.ckpt_path} ...", flush=True)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    print(f"[convert] {len(sd)} tensors", flush=True)

    # safetensors requires contiguous, non-shared-storage tensors
    clean = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        clean[k] = v.contiguous().clone()

    dst_safetensors = out / "model.safetensors"
    print(f"[convert] writing {dst_safetensors} ...", flush=True)
    save_file(clean, str(dst_safetensors))

    dst_config = out / "config.json"
    shutil.copyfile(args.template_config, dst_config)
    with open(dst_config) as f:
        cfg = json.load(f)
    cfg["source"] = f"sft_from_ckpt:{args.ckpt_path}"
    with open(dst_config, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[convert] wrote {dst_safetensors} ({dst_safetensors.stat().st_size / 1e9:.1f} GB) "
          f"and {dst_config}", flush=True)


if __name__ == "__main__":
    main()
