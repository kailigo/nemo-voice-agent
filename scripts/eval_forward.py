#!/usr/bin/env python
"""
Fast teacher-forced eval: load a base or SFT'd DuplexSTTModel, iterate a Lhotse Shar
val set, run one forward pass per cut, report per-cut and mean loss / function_loss /
token_accuracy.

This is NOT a tau2 metric. It measures next-token perplexity on held-out cuts —
answers "did SFT move the model in a direction that generalizes?", not "does it pass
tau2 episodes." For the latter, autoregressive eval on trimmed val cuts is the next
script; container path requires .ckpt→HF conversion (deferred).

Usage:

  # base model (no ckpt)
  python scripts/eval_forward.py \
      --config-path examples/speechlm2/conf/finetune \
      --config-name s2s_duplex_stt_11b \
      --val-shard-path data/tau2_training_samples/mix_full86/val \
      --device cuda \
      --output-json /tmp/eval_base.json

  # SFT'd
  python scripts/eval_forward.py \
      ... --ckpt-path .../step-500.ckpt \
      --output-json /tmp/eval_sft.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Make repo importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True,
                    help="Directory containing the Hydra config yaml (relative to repo root or absolute)")
    ap.add_argument("--config-name", required=True, help="Config file basename (no .yaml)")
    ap.add_argument("--val-shard-path", required=True, help="Lhotse Shar dir for val")
    ap.add_argument("--ckpt-path", default=None, help="Optional Lightning .ckpt to load")
    ap.add_argument("--pretrained-s2s-model", default=None,
                    help="Override model.pretrained_s2s_model")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="If >0, cap number of val cuts scored")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--strict-load", action="store_true",
                    help="Require exact state_dict key match when loading ckpt")
    return ap.parse_args()


def build_model_and_dataset(cfg, val_shard_path):
    """Constructs DuplexSTTModel and DuplexS2SDataset from a resolved OmegaConf cfg."""
    from nemo.collections.speechlm2 import DuplexS2SDataset, DuplexSTTModel

    # Point val to our shards; don't touch the train config
    OmegaConf.update(cfg, "data.validation_ds.datasets.val_set_0.shar_path",
                     str(val_shard_path), merge=True)

    model_cfg = OmegaConf.to_container(cfg, resolve=True)
    model = DuplexSTTModel(model_cfg)

    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=list(cfg.data.input_roles),
        output_roles=list(cfg.data.output_roles),
        cfg=OmegaConf.to_container(cfg.data, resolve=False),
        model_cfg=OmegaConf.to_container(cfg.model, resolve=False),
        force_align_user_text=False,
        early_interruption_prob=0.0,
    )
    return model, dataset


def load_ckpt_into_model(model, ckpt_path: str, strict: bool):
    """Load a Lightning .ckpt's state_dict into the constructed model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return {"missing": len(missing), "unexpected": len(unexpected),
            "missing_examples": missing[:5], "unexpected_examples": unexpected[:5]}


def iter_val_batches(dataset, val_shard_path, limit: int):
    """One-cut-at-a-time iteration; batch_size=1 to isolate per-cut loss."""
    from lhotse import CutSet
    cuts = CutSet.from_shar(in_dir=str(val_shard_path)).to_eager()
    for i, cut in enumerate(cuts):
        if limit and i >= limit:
            break
        batch = dataset[CutSet.from_cuts([cut])]
        yield cut.id, batch


def score_batch_forward(model, batch, device):
    """Runs the training_step forward+loss path with self.log patched, returns metrics."""
    # Patch logging to no-op so we don't need a Trainer
    model.log = lambda *a, **kw: None
    model.log_dict = lambda *a, **kw: None
    model._trainer = None  # training_step reads self.trainer.optimizers if _trainer is set

    # Move batch tensors to device
    def to_dev(x):
        if torch.is_tensor(x):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_dev(v) for v in x]
        return x

    batch = to_dev(batch)
    model.eval()
    with torch.no_grad():
        res = model.training_step(batch, batch_idx=0)

    # Extract scalar metrics from res dict
    metrics = {}
    for k, v in res.items():
        if torch.is_tensor(v) and v.numel() == 1:
            metrics[k] = float(v.detach().cpu().item())
    return metrics


def main():
    args = parse_args()
    cfg_dir = str((REPO / args.config_path).resolve())

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=args.config_name)

    # Sanity: point val to our shards, override pretrained_s2s_model if requested
    if args.pretrained_s2s_model:
        OmegaConf.update(cfg, "model.pretrained_s2s_model", args.pretrained_s2s_model)
    # Turn off things that only make sense inside a Trainer run
    OmegaConf.update(cfg, "model.debug_fc", False, force_add=True)

    print(f"[eval] building model from cfg (pretrained_s2s_model={cfg.model.pretrained_s2s_model})...",
          flush=True)
    t0 = time.time()
    model, dataset = build_model_and_dataset(cfg, args.val_shard_path)
    print(f"[eval] model built in {time.time()-t0:.1f}s", flush=True)

    load_info = None
    if args.ckpt_path:
        print(f"[eval] loading ckpt: {args.ckpt_path}", flush=True)
        t0 = time.time()
        load_info = load_ckpt_into_model(model, args.ckpt_path, strict=args.strict_load)
        print(f"[eval] ckpt loaded in {time.time()-t0:.1f}s  "
              f"(missing={load_info['missing']}, unexpected={load_info['unexpected']})",
              flush=True)
        if load_info["missing"] > 0:
            print(f"  missing keys (first 5): {load_info['missing_examples']}", flush=True)
        if load_info["unexpected"] > 0:
            print(f"  unexpected keys (first 5): {load_info['unexpected_examples']}", flush=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16, device=device)
    else:
        model = model.to(device)
    print(f"[eval] model on {device} (dtype={next(model.parameters()).dtype})", flush=True)

    per_cut = []
    print(f"[eval] iterating val cuts from {args.val_shard_path}", flush=True)
    for cid, batch in iter_val_batches(dataset, args.val_shard_path, args.limit_batches):
        t0 = time.time()
        m = score_batch_forward(model, batch, device)
        dt = time.time() - t0
        row = {"cut_id": cid, "seconds": round(dt, 2), **m}
        per_cut.append(row)
        loss = m.get("loss")
        fl = m.get("function_loss")
        ta = m.get("token_accuracy")
        print(f"  {cid:24s}  loss={loss}  function_loss={fl}  token_acc={ta}  ({dt:.1f}s)",
              flush=True)

    # Aggregate mean per metric (only cuts that produced numeric loss)
    keys = set()
    for r in per_cut:
        keys |= {k for k, v in r.items() if isinstance(v, (int, float)) and k not in ("seconds",)}
    means = {}
    for k in keys:
        vals = [r[k] for r in per_cut if isinstance(r.get(k), (int, float))]
        if vals:
            means[k] = sum(vals) / len(vals)

    out = {
        "ckpt_path": args.ckpt_path,
        "val_shard_path": args.val_shard_path,
        "n_cuts": len(per_cut),
        "load_info": load_info,
        "mean": means,
        "per_cut": per_cut,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[eval] mean over {len(per_cut)} cuts:")
    for k in sorted(means):
        print(f"  {k}: {means[k]:.4f}")
    print(f"\n[eval] wrote {args.output_json}")


if __name__ == "__main__":
    main()
