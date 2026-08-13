#!/usr/bin/env python3
"""
Rewrite an extracted STT checkpoint so its key names match a LoRA-wrapped DuplexSTTModel.

Why this is needed: DuplexSTTModel.__init__ installs LoRA (line ~313) BEFORE it loads
pretrained_s2s_model (line ~321). peft renames every LLM parameter —
  llm.layers.0.mixer.in_proj.weight  ->  llm.base_model.model.layers.0.mixer.in_proj.weight
and for LoRA-targeted projections also inserts .base_layer —
  ...mixer.q_proj.weight             ->  ...mixer.q_proj.base_layer.weight
The loader copies tensors by exact key match, so with a checkpoint extracted from the
released HF model NONE of the 339 LLM tensors match. It logs a warning and silently leaves
the whole 7.75B backbone at random init. This script closes that gap up front, so the
documented launch command works unchanged.

The mapping is not hand-written: we instantiate the model from the same config, then map
each model key back through the peft renaming to find its checkpoint counterpart. Whatever
peft does is therefore reflected automatically.

Usage:
  python scripts/remap_checkpoint_for_lora.py \
      --checkpoint /path/to/stt_extracted \
      --output     /path/to/stt_extracted_lora \
      [--config conf/finetune/s2s_duplex_stt_11b.yaml] \
      [--warm-start-asr-head]

Verify the result with:
  python scripts/verify_checkpoint_load.py --checkpoint /path/to/stt_extracted_lora
"""

import argparse
import json
import os
import shutil
import sys

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

# Warm-start pairs: (model key with no checkpoint weights, checkpoint key to copy from).
# The ASR channel is absent from the released 11B (use_separate_asr_head=false,
# predict_user_text=false), so asr_head/embed_asr_tokens have no pretrained counterpart.
# They are the same shape as the text-channel equivalents (131072 x 4480, same tokenizer),
# so copying those is a strictly better starting point than random init.
WARM_START_PAIRS = (
    ("asr_head.weight", "lm_head.weight"),
    ("embed_asr_tokens.weight", "embed_tokens.weight"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Input dir containing model.safetensors")
    parser.add_argument("--output", required=True, help="Output dir for the remapped checkpoint")
    parser.add_argument("--config", default="conf/finetune/s2s_duplex_stt_11b.yaml")
    parser.add_argument(
        "--warm-start-asr-head",
        action="store_true",
        help="Also initialise asr_head/embed_asr_tokens from the text-channel lm_head/embed_tokens "
        "instead of leaving them at random init (1.17B params).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)
    sys.path.insert(0, script_dir)

    from verify_checkpoint_load import FRESH_BY_DESIGN, normalize

    config_path = os.path.join(project_root, "examples", "speechlm2", args.config)
    if not os.path.exists(config_path):
        config_path = args.config
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "model.pretrained_weights", False)
    OmegaConf.update(cfg, "model.pretrained_s2s_model", None)
    OmegaConf.update(cfg, "data.train_ds.input_cfg", [{"type": "lhotse_shar", "shar_path": "/tmp/dummy"}])
    OmegaConf.update(cfg, "data.validation_ds.datasets.val_set_0.shar_path", "/tmp/dummy_val")
    OmegaConf.resolve(cfg)

    from nemo.collections.speechlm2.models import DuplexSTTModel

    print("Instantiating model from config to learn the LoRA-wrapped key names...")
    model = DuplexSTTModel(OmegaConf.to_container(cfg, resolve=True))
    model_sd = model.state_dict()

    # checkpoint-name -> model-name, derived from the model itself.
    ckpt_to_model = {}
    for mk in model_sd:
        if FRESH_BY_DESIGN.search(mk):
            continue  # LoRA adapters are meant to start fresh
        ckpt_to_model.setdefault(normalize(mk), mk)
    del model_sd
    del model

    src = os.path.join(args.checkpoint, "model.safetensors")
    os.makedirs(args.output, exist_ok=True)

    renamed, unchanged, dropped = 0, 0, []
    out = {}
    print(f"Reading {src} ...")
    with safe_open(src, framework="pt", device="cpu") as f:
        for key in f.keys():
            target = ckpt_to_model.get(key)
            if target is None:
                dropped.append(key)
                continue
            out[target] = f.get_tensor(key)
            if target == key:
                unchanged += 1
            else:
                renamed += 1

    if args.warm_start_asr_head:
        for dst, src_key in WARM_START_PAIRS:
            if dst in out:
                print(f"  {dst} already present in checkpoint — leaving it alone")
            elif src_key in out:
                out[dst] = out[src_key].clone()
                print(f"  warm start: {dst} <- {src_key}  {tuple(out[dst].shape)}")
            else:
                print(f"  [WARN] cannot warm start {dst}: {src_key} not in checkpoint")

    dst_path = os.path.join(args.output, "model.safetensors")
    print(f"Writing {len(out)} tensors to {dst_path} ...")
    save_file(out, dst_path, metadata={"format": "pt"})

    cfg_src = os.path.join(args.checkpoint, "config.json")
    if os.path.exists(cfg_src):
        shutil.copy2(cfg_src, os.path.join(args.output, "config.json"))

    print("\n" + "=" * 70)
    print(f"  renamed for LoRA wrapper : {renamed}")
    print(f"  already matching         : {unchanged}")
    print(f"  dropped (no home in model): {len(dropped)}")
    for k in dropped[:10]:
        print(f"      {k}")
    print(f"  total written            : {len(out)}")
    print("\nNow verify with:")
    print(f"  python scripts/verify_checkpoint_load.py --checkpoint {args.output}")


if __name__ == "__main__":
    main()
