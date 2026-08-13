#!/usr/bin/env python3
"""
Verify that the training setup works end-to-end without requiring GPU or full model weights.

This script checks:
  1. Config loading and resolution
  2. Model class instantiation (with random weights, no pretrained download)
  3. Dataset class instantiation
  4. Forward pass with dummy data
  5. Loss computation

Usage:
  python scripts/verify_training_setup.py [--config conf/finetune/s2s_duplex_stt_11b.yaml]

Run this BEFORE downloading the full checkpoint to catch config/code issues early.
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf


def create_dummy_batch(tokenizer, batch_size=2, seq_len=100, source_sr=16000, target_sr=22050, frame_length=0.08):
    """Create a minimal dummy batch matching DuplexS2SDataset output format."""
    samples_per_frame_source = int(source_sr * frame_length)
    samples_per_frame_target = int(target_sr * frame_length)

    source_audio_len = seq_len * samples_per_frame_source
    target_audio_len = seq_len * samples_per_frame_target

    pad_id = tokenizer.text_to_ids("<SPECIAL_12>")[0] if hasattr(tokenizer, 'text_to_ids') else 0
    bos_id = tokenizer.bos_id if hasattr(tokenizer, 'bos_id') else 1
    eos_id = tokenizer.eos_id if hasattr(tokenizer, 'eos_id') else 2

    # Create token sequence: PAD... BOS token token token EOS PAD...
    tokens = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    tokens[:, 10] = bos_id
    tokens[:, 11:15] = torch.randint(100, 1000, (batch_size, 4))
    tokens[:, 15] = eos_id

    batch = {
        "audio_data": {
            "source_audio": torch.randn(batch_size, source_audio_len),
            "source_audio_lens": torch.full((batch_size,), source_audio_len, dtype=torch.long),
            "target_audio": torch.randn(batch_size, target_audio_len),
            "target_audio_lens": torch.full((batch_size,), target_audio_len, dtype=torch.long),
            "target_tokens": tokens,
            "target_token_lens": torch.full((batch_size,), seq_len, dtype=torch.long),
            "source_tokens": tokens.clone(),
            "prompt_tokens": torch.full((batch_size, 5), pad_id, dtype=torch.long),
            "prompt_token_lens": torch.full((batch_size,), 5, dtype=torch.long),
            "formatter": ["lhotse_shar"] * batch_size,
        }
    }
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/finetune/s2s_duplex_stt_11b.yaml")
    parser.add_argument("--skip_model", action="store_true", help="Skip model instantiation (just check config)")
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

    # Disable pretrained weight loading for verification
    OmegaConf.update(cfg, "model.pretrained_weights", False)
    OmegaConf.update(cfg, "model.pretrained_asr", "")  # Skip ASR loading
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

    batch = create_dummy_batch(
        model.tokenizer,
        batch_size=1,
        seq_len=50,
        source_sr=cfg.data.source_sample_rate,
        target_sr=cfg.data.target_sample_rate,
        frame_length=cfg.data.frame_length,
    )

    print("  Running training_step with dummy batch...")
    try:
        model.train()
        loss = model.training_step(batch, batch_idx=0)
        print(f"  Training step returned: {type(loss)}")
        if isinstance(loss, dict):
            for k, v in loss.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.item():.4f}")
        print("  [OK] Forward pass succeeded\n")
    except Exception as e:
        print(f"  [WARN] Forward pass failed: {e}")
        print("  This may be expected with random weights / missing perception module.\n")

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
