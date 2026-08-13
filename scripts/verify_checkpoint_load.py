#!/usr/bin/env python3
"""
Verify that the extracted STT checkpoint actually lands in the model we build from the config.

This answers "are we starting from the pretrained model, or from noise?" — a question
DuplexSTTModel's own loader cannot answer. That loader (duplex_stt_model.py ~line 333)
iterates over *checkpoint* keys and warns only about keys present in the checkpoint but
absent from the model. It never reports the reverse direction, so any model parameter that
receives no checkpoint tensor silently keeps whatever it was built with.

Note what "built with" means, because it makes the failure quiet rather than loud: the LLM
body, lm_head and embed_tokens come from cfg.pretrained_llm via HF (line 228) BEFORE this
load, and asr_head/embed_asr_tokens are deep-copied from lm_head (line 275). So an unloaded
LLM falls back to base Nemotron-Nano-9B-v2 — fluent, but missing every bit of VoiceChat
duplex/turn-taking/function-calling finetuning. Only components with no HF source (notably
perception.proj) are genuinely random. Either way you are not finetuning the released model.

The direction that matters is therefore MODEL-ONLY keys, which this script reports loudly.

Usage:
  python scripts/verify_checkpoint_load.py \
      --checkpoint /path/to/stt_extracted \
      [--config conf/finetune/s2s_duplex_stt_11b.yaml]

Exit code is 1 if any non-LoRA model parameter would go unloaded.
"""

import argparse
import os
import re
import sys
from collections import Counter

import torch
from omegaconf import OmegaConf
from safetensors import safe_open

# peft wraps model.llm, which renames every LLM key. These transforms undo that renaming so
# we can tell a genuine missing weight apart from a cosmetic prefix difference.
PEFT_RENAMES = (
    ("llm.base_model.model.", "llm."),  # get_peft_model() wrapper nesting
    (".base_layer.", "."),  # LoRA-adapted Linear keeps its original weight here
)

# Keys that are SUPPOSED to be absent from the checkpoint: freshly initialised LoRA adapters.
FRESH_BY_DESIGN = re.compile(r"\.lora_(A|B|embedding_A|embedding_B|magnitude_vector)\.")


def normalize(key):
    for src, dst in PEFT_RENAMES:
        key = key.replace(src, dst)
    return key


def group_of(key):
    """Coarse component label, for a readable summary instead of 1000 key names."""
    top = key.split(".")[0]
    if top == "perception":
        parts = key.split(".")
        return f"perception.{parts[1]}" if len(parts) > 1 else top
    return top


def summarize(title, keys, numels, limit=6):
    print(f"\n  {title}: {len(keys)} tensors, {sum(numels.get(k, 0) for k in keys) / 1e9:.3f}B params")
    if not keys:
        return
    by_group = Counter()
    params_by_group = Counter()
    for k in keys:
        by_group[group_of(k)] += 1
        params_by_group[group_of(k)] += numels.get(k, 0)
    for g, n in by_group.most_common():
        print(f"      {g:32s} {n:5d} tensors  {params_by_group[g] / 1e9:8.3f}B")
    print("    examples:")
    for k in sorted(keys)[:limit]:
        print(f"      {k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Directory containing model.safetensors")
    parser.add_argument("--config", default="conf/finetune/s2s_duplex_stt_11b.yaml")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)

    config_path = os.path.join(project_root, "examples", "speechlm2", args.config)
    if not os.path.exists(config_path):
        config_path = args.config
    cfg = OmegaConf.load(config_path)

    # Build the architecture only. pretrained_weights=False skips pulling the ASR .nemo
    # weights, and pretrained_s2s_model=None skips the load we are auditing — neither
    # affects the parameter NAMES, which is all we compare here.
    OmegaConf.update(cfg, "model.pretrained_weights", False)
    OmegaConf.update(cfg, "model.pretrained_s2s_model", None)
    OmegaConf.update(cfg, "data.train_ds.input_cfg", [{"type": "lhotse_shar", "shar_path": "/tmp/dummy"}])
    OmegaConf.update(cfg, "data.validation_ds.datasets.val_set_0.shar_path", "/tmp/dummy_val")
    OmegaConf.resolve(cfg)

    from nemo.collections.speechlm2.models import DuplexSTTModel

    print("=" * 70)
    print("Instantiating model from config (architecture only)")
    print("=" * 70)
    model = DuplexSTTModel(OmegaConf.to_container(cfg, resolve=True))
    model_sd = model.state_dict()
    model_numel = {k: v.numel() for k, v in model_sd.items()}
    model_shape = {k: tuple(v.shape) for k, v in model_sd.items()}

    ckpt_path = os.path.join(args.checkpoint, "model.safetensors")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        ckpt_shape = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    ckpt_numel = {k: int(torch.tensor(list(s)).prod()) if s else 1 for k, s in ckpt_shape.items()}

    print("=" * 70)
    print("Key-set comparison")
    print("=" * 70)
    print(f"  model state_dict : {len(model_sd)} tensors, {sum(model_numel.values()) / 1e9:.3f}B params")
    print(f"  checkpoint       : {len(ckpt_shape)} tensors, {sum(ckpt_numel.values()) / 1e9:.3f}B params")

    # --- as the real loader sees it: exact string match, no normalization ---
    exact_hits = [k for k in ckpt_shape if k in model_sd]
    print(f"\n  Exact-name matches (what the loader in duplex_stt_model.py would copy): "
          f"{len(exact_hits)} / {len(ckpt_shape)}")

    # --- after undoing peft's renaming ---
    norm_to_model = {}
    for k in model_sd:
        norm_to_model.setdefault(normalize(k), []).append(k)

    matched, shape_mismatch, ckpt_only = [], [], []
    for k, shp in ckpt_shape.items():
        targets = norm_to_model.get(normalize(k), [])
        if not targets:
            ckpt_only.append(k)
        elif any(model_shape[t] == shp for t in targets):
            matched.append(k)
        else:
            shape_mismatch.append((k, shp, [model_shape[t] for t in targets]))

    matched_model_keys = set()
    for k in matched:
        matched_model_keys.update(norm_to_model[normalize(k)])
    model_only = [k for k in model_sd if k not in matched_model_keys]
    model_only_real = [k for k in model_only if not FRESH_BY_DESIGN.search(k)]
    model_only_lora = [k for k in model_only if FRESH_BY_DESIGN.search(k)]

    print(f"  Matches after undoing peft renaming: {len(matched)} / {len(ckpt_shape)}")

    summarize("CHECKPOINT-ONLY (in checkpoint, no home in model — discarded)", ckpt_only, ckpt_numel)
    summarize("LoRA adapters (absent by design, initialised fresh)", model_only_lora, model_numel)
    summarize("MODEL-ONLY (gets NO checkpoint weights — base HF LLM or random init)",
              model_only_real, model_numel)

    if shape_mismatch:
        print(f"\n  SHAPE MISMATCHES: {len(shape_mismatch)}")
        for k, cs, ms in shape_mismatch[:10]:
            print(f"      {k}: checkpoint {cs} vs model {ms}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    loaded_params = sum(model_numel[k] for k in matched_model_keys)
    total_params = sum(model_numel.values())
    print(f"  Pretrained coverage: {loaded_params / 1e9:.3f}B / {total_params / 1e9:.3f}B "
          f"({100 * loaded_params / total_params:.1f}%) of model params")

    ok = True
    # Only checkpoint tensors that HAVE a counterpart in the model need to match by exact
    # name. Extra tensors the model has no slot for (e.g. rnnt_decoder, or asr_head when
    # predict_user_text is false) are discarded by design and are not a rename failure.
    matchable = len(ckpt_shape) - len(ckpt_only)
    if len(exact_hits) < matchable:
        print(f"  [FAIL] Of the {matchable} checkpoint tensors that map to a model parameter, "
              f"the loader matches only {len(exact_hits)} by exact name. Remap the checkpoint "
              f"(scripts/remap_checkpoint_for_lora.py) or install LoRA after loading.")
        ok = False
    if model_only_real:
        print(f"  [FAIL] {len(model_only_real)} model tensors "
              f"({sum(model_numel[k] for k in model_only_real) / 1e9:.3f}B params) receive "
              f"nothing from the checkpoint.")
        ok = False
    if shape_mismatch:
        print(f"  [FAIL] {len(shape_mismatch)} tensors have incompatible shapes.")
        ok = False
    if ok:
        print("  [OK] Every non-LoRA model parameter is covered by the checkpoint, shapes agree.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
