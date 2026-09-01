#!/usr/bin/env python
"""
FDB-v3-style argument-accuracy scorer for DuplexSTTModel, on the FIRST tool call of each
held-out cut.

WHY "FIRST CALL ONLY" -- read this before trusting the numbers
----------------------------------------------------------------
offline_inference() with no function_call_steps/function_responses given generates the
agent's text+function channels completely freely, from its own step-by-step loop
(_step_zero/_step_inference) -- the same path already validated for correctness this
session (test_hybrid_cache_correctness.py, exact match vs. no-cache reference).

That free generation has no memory of the REAL agent history preceding the point we care
about -- it would have to re-decide and re-generate every earlier turn/call itself, and by
the time it reaches a LATER call its own (possibly wrong) prior turns are its context, not
the teacher's. There is also no tool-executor here to feed back real tool results between
calls. So scoring calls 2, 3, 4... would conflate "is this call right" with "did the model's
own earlier improvised turns happen to match the teacher's" -- a different, murkier question.

The FIRST call in a cut needs neither: real user audio truncated right before it, plus the
system prompt, is the complete real context leading up to it. So this script scores only
that one call per cut. It is a real, honest measurement of "does the model call the right
tool with the right arguments, given real preceding audio" -- just for one call position,
not the whole episode.

WHAT "EXPECTED" MEANS HERE -- also read before trusting the numbers
----------------------------------------------------------------
Our held-out cuts have no field tracing back to an externally-verified tau2 task/mock-env
ground truth -- only what the teacher itself called. So "expected" == "what the teacher
did", and a low score here means "diverged from the teacher's own trajectory", not
necessarily "objectively wrong". This is a distillation-fidelity measurement, close to but
not identical to TAU_VOICE_SFT_RL_PROGRAM.md §6's real gate (which needs a live tau2
episode against the actual mock environment).

SCORING RULES -- reused directly from Full-Duplex-Bench/v3/evaluate_tool_calls.py
----------------------------------------------------------------
Per call: if the model emits no <TOOLCALL> at all -> mode E, score 0. If it emits one with
a name not in <AVAILABLE_TOOLS> -> mode F (invented name), score 0. If the name is a real
tool but not the expected one -> name_mismatch, score 0. If the name matches, arguments are
judged with exact_match_args (same normalization: lowercase, strip, underscore->space;
"$..." expected values skipped as dynamic references) -- the exact function imported from
Full-Duplex-Bench/v3, not reimplemented.

Usage:
  python scripts/eval_first_call_argument_accuracy.py \
      --config-path examples/speechlm2/conf/finetune --config-name s2s_duplex_stt_11b \
      --val-shard-path data/tau2_training_samples/shards-0828/val \
      --pretrained-s2s-model /fsx/home/kai.li/data/voicechat/stt_extracted_lora \
      --ckpt-path logs/sft_newdomains_0831/exp/checkpoints/step-500.ckpt \
      --output-json logs/eval_argacc_0831/newdom_step500.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
FDB_V3_DIR = Path("/fsx/home/kai.li/code/Full-Duplex-Bench/v3")
sys.path.insert(0, str(FDB_V3_DIR))

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

from evaluate_tool_calls import exact_match_args  # noqa: E402  (Full-Duplex-Bench/v3, reused as-is)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--val-shard-path", required=True)
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--pretrained-s2s-model", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit-cuts", type=int, default=0)
    ap.add_argument("--pad-seconds", type=float, default=10.0,
                     help="Silence appended after the truncation point so the model has room to act")
    ap.add_argument("--output-json", required=True)
    return ap.parse_args()


def extract_available_tools(system_prompt: str) -> set:
    m = re.search(r"<AVAILABLE_TOOLS>(.*?)</AVAILABLE_TOOLS>", system_prompt, re.S)
    if not m:
        return set()
    try:
        tools = json.loads(m.group(1))
    except json.JSONDecodeError:
        return set()
    names = set()
    for t in tools:
        if isinstance(t, dict):
            if "function" in t and isinstance(t["function"], dict):
                names.add(t["function"].get("name"))
            elif "name" in t:
                names.add(t["name"])
    return names


def first_expected_call(cut):
    """Returns (call_start_seconds, {"name":..., "arguments":...}) for the first agent
    <TOOLCALL> in the cut, or (None, None) if there is none."""
    for s in cut.supervisions:
        if s.speaker != "agent":
            continue
        fn = (s.custom or {}).get("function") or ""
        if "<TOOLCALL>" not in fn:
            continue
        m = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", fn, re.S)
        if not m:
            continue
        try:
            calls = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if calls:
            return s.start, calls[0]
    return None, None


def parse_predicted_call(text: str):
    m = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", text, re.S)
    if not m:
        return None
    try:
        calls = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return calls[0] if calls else None


def score_cut(model, device, cut, pad_seconds: float):
    t_call, expected = first_expected_call(cut)
    if expected is None:
        return {"cut_id": cut.id, "skipped": "no tool call in this cut"}

    system_prompt = cut.custom["system_prompt"]
    available_tools = extract_available_tools(system_prompt)

    audio = cut.load_audio()[0]  # (N,) float32, cut.recording.sampling_rate
    sr = cut.recording.sampling_rate
    n_samples = int(t_call * sr)
    truncated = audio[:n_samples]
    pad = np.zeros(int(pad_seconds * sr), dtype=truncated.dtype)
    truncated = np.concatenate([truncated, pad])

    input_signal = torch.tensor(truncated, dtype=torch.float32, device=device).unsqueeze(0)
    input_signal_lens = torch.tensor([input_signal.shape[1]], device=device)

    tok = model.tokenizer
    prompt_ids = [tok.bos] + tok.text_to_ids(system_prompt) + [tok.eos]
    prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_token_lens = torch.tensor([len(prompt_ids)], device=device)

    with torch.no_grad():
        out = model.offline_inference(
            input_signal, input_signal_lens,
            prompt_tokens=prompt_tokens, prompt_token_lens=prompt_token_lens,
        )

    func_tokens = out["tokens_function"][0].tolist()
    func_tokens = [t for t in func_tokens if t != model.text_pad_id]
    func_text = model.tokenizer.ids_to_text(func_tokens)
    predicted = parse_predicted_call(func_text)

    expected_name = expected.get("name")
    expected_args = expected.get("arguments", {})

    if predicted is None:
        mode, score, reason = "E_no_call", 0.0, "model emitted no <TOOLCALL>"
    else:
        pred_name = predicted.get("name")
        if pred_name not in available_tools:
            mode, score, reason = "F_invented_name", 0.0, f"'{pred_name}' not in AVAILABLE_TOOLS"
        elif pred_name != expected_name:
            mode, score, reason = "name_mismatch", 0.0, f"expected '{expected_name}', got '{pred_name}'"
        else:
            ok, explanation = exact_match_args(expected_args, predicted.get("arguments", {}))
            mode = "name_match_args_ok" if ok else "name_match_args_wrong"
            score = 1.0 if ok else 0.0
            reason = explanation

    return {
        "cut_id": cut.id,
        "t_call": round(t_call, 2),
        "expected": expected,
        "predicted": predicted,
        "mode": mode,
        "score": score,
        "reason": reason,
        "func_text_full": func_text,
    }


def main():
    args = parse_args()
    cfg_dir = str((REPO / args.config_path).resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=args.config_name)

    if args.pretrained_s2s_model:
        OmegaConf.update(cfg, "model.pretrained_s2s_model", args.pretrained_s2s_model)
    OmegaConf.update(cfg, "model.debug_fc", False, force_add=True)

    from nemo.collections.speechlm2 import DuplexSTTModel

    print(f"[eval] building model (pretrained_s2s_model={cfg.model.pretrained_s2s_model})...", flush=True)
    model_cfg = OmegaConf.to_container(cfg, resolve=True)
    model = DuplexSTTModel(model_cfg)

    if args.ckpt_path:
        print(f"[eval] loading ckpt: {args.ckpt_path}", flush=True)
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[eval] ckpt loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    device = torch.device(args.device)
    model = model.to(dtype=torch.bfloat16, device=device).eval()

    from lhotse import CutSet
    cuts = CutSet.from_shar(in_dir=args.val_shard_path).to_eager()

    results = []
    for i, cut in enumerate(cuts):
        if args.limit_cuts and i >= args.limit_cuts:
            break
        t0 = time.time()
        r = score_cut(model, device, cut, args.pad_seconds)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"  {r['cut_id']:20s} {r.get('mode', r.get('skipped')):24s} "
              f"score={r.get('score')}  ({r['seconds']}s)", flush=True)

    scored = [r for r in results if "score" in r]
    n_skipped = len(results) - len(scored)
    mean_score = sum(r["score"] for r in scored) / len(scored) if scored else 0.0
    mode_counts = {}
    for r in scored:
        mode_counts[r["mode"]] = mode_counts.get(r["mode"], 0) + 1

    out = {
        "ckpt_path": args.ckpt_path,
        "val_shard_path": args.val_shard_path,
        "n_cuts": len(results),
        "n_scored": len(scored),
        "n_skipped_no_call": n_skipped,
        "mean_argument_accuracy": round(mean_score, 4),
        "mode_counts": mode_counts,
        "per_cut": results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[eval] scored {len(scored)}/{len(results)} cuts (skipped {n_skipped}, no call in cut)")
    print(f"[eval] mean_argument_accuracy: {mean_score:.4f}")
    print(f"[eval] mode_counts: {mode_counts}")
    print(f"[eval] wrote {args.output_json}")


if __name__ == "__main__":
    main()
