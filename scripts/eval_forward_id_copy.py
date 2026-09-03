#!/usr/bin/env python
"""
Teacher-forced ID-copy check: exactly the training objective (ground-truth previous-step
tokens as input), scored narrowly on whether the model's predicted function-channel tokens
correctly reproduce a spelled-id value, not aggregate loss/token_accuracy over the whole cut.

Why this and not eval_forward.py's existing metrics: token_accuracy there is averaged over
every text+function token in the cut, so a handful of id-copy tokens are diluted among
everything else. This isolates the same question §14/the live-batch investigation raised --
"did training even inject the right behavior, separate from the free-running exposure-bias
question" -- by decoding BOTH the model's teacher-forced predicted function-channel tokens
and the ground-truth target tokens to text, parsing a tool call out of each the same way
mode_a_readout1.py does, and comparing the id argument value directly. No task lookup needed:
the ground-truth id comes from the cut's own training label, not an external source.

Usage (in-domain):
  python scripts/eval_forward_id_copy.py \
      --config-path examples/speechlm2/conf/finetune --config-name s2s_duplex_stt_11b \
      --val-shard-path data/tau2_training_samples/airline/val \
      --ckpt-path logs/sft_train8_0826/exp/checkpoints/step-500.ckpt \
      --output-json logs/eval_forward_id_copy_airline_sft500.json

Cross-domain:
  ... --val-shard-path data/tau2_training_samples/shards-0828/val ...
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

from eval_forward import build_model_and_dataset, load_ckpt_into_model  # noqa: E402
from mode_a_probe import normalise_id, edit_distance  # noqa: E402


def parse_toolcall(text: str):
    m = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", text, re.S)
    if not m:
        return None
    try:
        calls = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return calls[0] if calls else None


def find_any_id_like_value(obj):
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(find_any_id_like_value(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(find_any_id_like_value(v))
    elif isinstance(obj, str):
        out.append(obj)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--val-shard-path", required=True)
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--pretrained-s2s-model", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit-batches", type=int, default=0)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--strict-load", action="store_true")
    return ap.parse_args()


def iter_val_batches(dataset, val_shard_path, limit: int):
    from lhotse import CutSet
    cuts = CutSet.from_shar(in_dir=str(val_shard_path)).to_eager()
    for i, cut in enumerate(cuts):
        if limit and i >= limit:
            break
        batch = dataset[CutSet.from_cuts([cut])]
        yield cut.id, batch


def to_dev(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_dev(v, device) for v in x]
    return x


def score_id_copy(model, batch, device):
    """Teacher-forced forward pass (same inputs training_step uses), but returns decoded
    predicted vs. target function-channel text instead of aggregate loss."""
    batch = to_dev(batch, device)
    model.eval()
    with torch.no_grad():
        inputs = model.prepare_inputs(batch["audio_data"], include_asr_loss=False)
        forward_outputs = model(inputs["input_embeds"], compute_asr=inputs["compute_asr"])
        function_logits = forward_outputs.get("function_logits")
        if function_logits is None or "function_labels" not in inputs or inputs["function_labels"] is None:
            return None
        function_predicted_tokens = torch.argmax(function_logits, dim=-1)[0]  # (T,)
        function_target_tokens = inputs["function_labels"][0]  # (T,)

    pad_id = model.text_pad_id
    target_mask = function_target_tokens != pad_id
    pred_tokens = function_predicted_tokens[target_mask].cpu().tolist()
    target_tokens = function_target_tokens[target_mask].cpu().tolist()

    pred_text = model.tokenizer.ids_to_text([t for t in pred_tokens if t != pad_id])
    target_text = model.tokenizer.ids_to_text([t for t in target_tokens if t != pad_id])
    return pred_text, target_text


def main():
    args = parse_args()
    cfg_dir = str((REPO / args.config_path).resolve())

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=args.config_name)

    if args.pretrained_s2s_model:
        OmegaConf.update(cfg, "model.pretrained_s2s_model", args.pretrained_s2s_model)
    OmegaConf.update(cfg, "model.debug_fc", False, force_add=True)

    print(f"[id-copy] building model (pretrained_s2s_model={cfg.model.pretrained_s2s_model})...", flush=True)
    model, dataset = build_model_and_dataset(cfg, args.val_shard_path)

    load_info = None
    if args.ckpt_path:
        print(f"[id-copy] loading ckpt: {args.ckpt_path}", flush=True)
        load_info = load_ckpt_into_model(model, args.ckpt_path, strict=args.strict_load)
        print(f"[id-copy] ckpt loaded (missing={load_info['missing']}, unexpected={load_info['unexpected']})",
              flush=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(dtype=torch.bfloat16, device=device) if device.type == "cuda" else model.to(device)
    print(f"[id-copy] model on {device}", flush=True)

    results = []
    for cid, batch in iter_val_batches(dataset, args.val_shard_path, args.limit_batches):
        t0 = time.time()
        out = score_id_copy(model, batch, device)
        dt = time.time() - t0
        if out is None:
            print(f"  {cid:28s}  no function_labels in this cut, skipped  ({dt:.1f}s)", flush=True)
            continue
        pred_text, target_text = out

        target_call = parse_toolcall(target_text)
        pred_call = parse_toolcall(pred_text)
        target_values = find_any_id_like_value(target_call.get("arguments", {})) if target_call else []
        pred_values = find_any_id_like_value(pred_call.get("arguments", {})) if pred_call else []

        # For each ground-truth argument value, does teacher-forced prediction reproduce it?
        matches = []
        for tv in target_values:
            tv_n = normalise_id(tv)
            if not tv_n or tv_n.isdigit() and len(tv_n) <= 2:
                continue  # skip trivially short/numeric values (e.g. flags), not id-like
            got_it = any(normalise_id(pv) == tv_n for pv in pred_values)
            best_edit = min((edit_distance(normalise_id(pv), tv_n) for pv in pred_values), default=None)
            matches.append({"target_value": tv, "got_it_right": got_it, "best_edit_distance": best_edit})

        row = {
            "cut_id": cid, "seconds": round(dt, 2),
            "target_call": target_call, "pred_call": pred_call,
            "target_values": target_values, "pred_values": pred_values,
            "id_matches": matches,
        }
        results.append(row)
        n_right = sum(m["got_it_right"] for m in matches)
        print(f"  {cid:28s}  target_values={target_values}  pred_values={pred_values}  "
              f"{n_right}/{len(matches)} id(s) correct  ({dt:.1f}s)", flush=True)

    all_matches = [m for r in results for m in r["id_matches"]]
    n_right = sum(m["got_it_right"] for m in all_matches)
    n_total = len(all_matches)
    out = {
        "ckpt_path": args.ckpt_path, "val_shard_path": args.val_shard_path,
        "n_cuts": len(results), "n_id_values": n_total, "n_correct": n_right,
        "accuracy": (n_right / n_total) if n_total else None,
        "per_cut": results,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[id-copy] teacher-forced id-copy accuracy: {n_right}/{n_total} "
          f"({(n_right/n_total*100) if n_total else 0:.1f}%) over {len(results)} cuts")
    print(f"[id-copy] wrote {args.output_json}")


if __name__ == "__main__":
    main()
