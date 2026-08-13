#!/usr/bin/env python3
"""
Verify that the training setup works end-to-end without needing the full pretrained weights.

This script checks:
  1. Config loading and resolution
  2. Model class instantiation (with random weights, no pretrained download)
  3. Dataset class instantiation
  4. Forward pass with dummy data
  5. Loss computation

Usage:
  python scripts/verify_training_setup.py [--config conf/finetune/s2s_duplex_stt_11b.yaml]

Run this BEFORE downloading the full checkpoint to catch config/code issues early.

A GPU is required for the forward-pass check (step 3). Nemotron-H's Mamba mixer calls
torch.cuda.default_stream(hidden_states.device) unconditionally, so the LLM cannot run on
CPU at all — steps 1, 2 and 4 work CPU-only, step 3 does not. Pass --device cpu to skip
straight to the (expected) CPU failure, or --skip_forward to leave step 3 out entirely.
"""

import argparse
import os
import sys
import traceback

import torch
from omegaconf import OmegaConf


def create_dummy_batch(model, cfg):
    """Build a batch for the forward-pass check.

    Uses DuplexS2SDataset._create_minimal_batch() rather than hand-rolling the dict.
    training_step() reads ~20 keys from batch["audio_data"] (source_token_lens,
    sample_id, is_minimal_batch, all_texts, ...), so a hand-written subset drifts out
    of date and fails on a missing key that says nothing about the config being tested.
    Letting the real dataset build it keeps this check honest as the format evolves.
    """
    from nemo.collections.speechlm2.data.s2s_dataset import DuplexS2SDataset

    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=list(cfg.data.input_roles),
        output_roles=list(cfg.data.output_roles),
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
    )
    # training_step() expects the collated dict nested under "audio_data", with
    # "text_data" present (None = audio-only step, which is what STT training does).
    return {"audio_data": dataset._create_minimal_batch(), "text_data": None}


def move_to_device(obj, device):
    """Recursively move every tensor in a (possibly nested) batch to `device`."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/finetune/s2s_duplex_stt_11b.yaml")
    parser.add_argument("--skip_model", action="store_true", help="Skip model instantiation (just check config)")
    parser.add_argument("--skip_forward", action="store_true", help="Skip the forward-pass check (step 3)")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the forward-pass check. Nemotron-H's Mamba mixer is CUDA-only, so "
        "step 3 can only pass on 'cuda'.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Dtype for the forward-pass check. Default matches trainer.precision=bf16-true.",
    )
    args = parser.parse_args()

    # Add project to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)

    print("=" * 60)
    print("STEP 1: Load and resolve config")
    print("=" * 60)

    config_path = os.path.join(project_root, "examples", "speechlm2", args.config)
    if not os.path.exists(config_path):
        config_path = args.config

    cfg = OmegaConf.load(config_path)

    # Override ??? placeholders for verification
    OmegaConf.update(cfg, "model.pretrained_s2s_model", None)
    OmegaConf.update(cfg, "data.train_ds.input_cfg", [{"type": "lhotse_shar", "shar_path": "/tmp/dummy"}])
    OmegaConf.update(cfg, "data.validation_ds.datasets.val_set_0.shar_path", "/tmp/dummy_val")
    OmegaConf.update(cfg, "trainer.max_steps", 10)

    OmegaConf.resolve(cfg)

    print(f"  Config loaded from: {config_path}")
    print(f"  LLM: {cfg.model.pretrained_llm}")
    print(f"  ASR encoder: {cfg.model.pretrained_asr}")
    print(f"  LoRA enabled: {'lora' in cfg.model}")
    if 'lora' in cfg.model:
        print(f"    rank: {cfg.model.lora.r}, alpha: {cfg.model.lora.lora_alpha}")
    print(f"  Trainer devices: {cfg.trainer.devices}")
    print(f"  Trainer precision: {cfg.trainer.precision}")
    print(f"  Max steps: {cfg.trainer.max_steps}")
    print(f"  Grad accumulation: {cfg.trainer.accumulate_grad_batches}")
    print("  [OK] Config loaded successfully\n")

    if args.skip_model:
        print("Skipping model instantiation (--skip_model)")
        return

    print("=" * 60)
    print("STEP 2: Instantiate model (random weights, no downloads)")
    print("=" * 60)

    # Skip loading the 11B LLM weights — we only want the architecture here.
    # pretrained_asr is deliberately left alone: setup_speech_encoder() fills in
    # perception.preprocessor and perception.encoder from the ASR .nemo, and the
    # training config relies on that rather than duplicating them. Blanking it
    # takes the "embedded perception config" branch instead and dies on a missing
    # perception.preprocessor key, which looks like a config bug but isn't.
    OmegaConf.update(cfg, "model.pretrained_weights", False)
    OmegaConf.update(cfg, "model.pretrained_s2s_model", None)

    from nemo.collections.speechlm2.models import DuplexSTTModel

    print("  Instantiating DuplexSTTModel with random weights...")
    model = DuplexSTTModel(OmegaConf.to_container(cfg, resolve=True))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.1f}M)")
    print(f"  Frozen parameters: {(total_params - trainable_params):,}")
    print("  [OK] Model instantiated\n")

    print("=" * 60)
    print("STEP 3: Verify forward pass with dummy data")
    print("=" * 60)

    if args.skip_forward:
        print("  Skipped (--skip_forward)\n")
    else:
        run_forward_check(model, cfg, args)

    report_components(model)


def run_forward_check(model, cfg, args):
    batch = create_dummy_batch(model, cfg)

    # Nemotron-H's mixer does `torch.cuda.stream(torch.cuda.default_stream(device))`
    # with no CPU fallback, so both model and batch have to be on the GPU. bf16 also
    # matches trainer.precision=bf16-true and is what the mamba-ssm kernels expect.
    dtype = getattr(torch, args.dtype)
    print(f"  Moving model to {args.device} ({args.dtype})...")
    model = model.to(device=args.device, dtype=dtype)
    batch = move_to_device(batch, args.device)

    print("  Running training_step with dummy batch...")
    try:
        model.train()
        loss = model.training_step(batch, batch_idx=0)
        print(f"  Training step returned: {type(loss)}")
        if isinstance(loss, dict):
            for k, v in loss.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.item():.4f}")
        elif isinstance(loss, torch.Tensor):
            print(f"    loss: {loss.item():.4f}")
        print("  [OK] Forward pass succeeded\n")
    except Exception as e:
        # Print the full traceback: a bare one-line message here previously hid a real
        # perception dim mismatch behind "this may be expected", which is exactly the
        # class of bug this script exists to catch.
        print(f"  [FAIL] Forward pass failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n  Investigate before launching training — this is not expected to fail.\n")


def report_components(model):
    print("=" * 60)
    print("STEP 4: Check component structure")
    print("=" * 60)

    components = {
        "LLM (llm)": model.llm,
        "Text head (lm_head)": model.lm_head,
        "Embeddings (embed_tokens)": model.embed_tokens,
        "Perception": model.perception if hasattr(model, 'perception') else None,
    }
    if hasattr(model, 'asr_head'):
        components["ASR head (asr_head)"] = model.asr_head
    if hasattr(model, 'function_head') and model.function_head is not None:
        components["Function head (function_head)"] = model.function_head

    for name, component in components.items():
        if component is None:
            print(f"  {name}: NOT PRESENT")
        else:
            params = sum(p.numel() for p in component.parameters())
            print(f"  {name}: {params:,} params ({params / 1e9:.2f}B)")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Download model: huggingface-cli download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B")
    print("  2. Extract STT weights: python scripts/extract_stt_checkpoint.py --input_dir <hf_dir> --output_dir <stt_dir>")
    print("  3. Prepare training data in Lhotse Shar format")
    print("  4. Run training: torchrun --nproc_per_node=8 examples/speechlm2/s2s_duplex_stt_train.py ...")


if __name__ == "__main__":
    main()
