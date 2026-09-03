#!/usr/bin/env python
"""
FDB-v3-style argument accuracy on tau2 tool calls, scored against our own held-out
Lhotse cuts -- the actual §6 gate in TAU_VOICE_SFT_RL_PROGRAM.md ("FDB-v3-style argument
accuracy on tau2 voice tool calls, and per-mode counts"), not the token_accuracy/sotc_acc
forward-loss proxies used so far.

Algorithm replicated faithfully from Full-Duplex-Bench/v3/evaluate_tool_calls.py and
evaluate_pass_rate.py (read in full, not guessed):
  - tool_selection: multiset F1 of recall/precision over tool NAMES only.
  - argument_accuracy: for each EXPECTED call, look up an ACTUAL call with the same name
    (FIFO across duplicates), judge args (exact-match by default; --use-llm routes through
    the same Bedrock-judge shim as scripts/fdb_v3_evaluate.py); a name never called scores 0.
  - pass_at_1: strict binary -- tool_selection recall==1 AND precision==1 AND every called
    function's arguments judged correct.

Ground truth and "actual" calls both come from the SAME source in a fundamentally different
way here than in FDB-v3: expected_calls are parsed out of the cut's own <TOOLCALL> supervisions
(the teacher's trajectory -- there is no separate gold field, confirmed by inspection), and
actual_calls come from running the model on the cut's real audio in a genuinely FREE-RUNNING
generation (offline_inference with no function_call_steps/lengths/responses passed --
confirmed those are optional kwargs that skip _expand_for_function_calling entirely when
absent, so the model decides on its own whether/when/what to call, not just teacher-forced
argmax). This means later calls in a multi-hop cut are scored against whatever context the
model's OWN earlier behaviour produced, including its own errors compounding -- which is a
closer analogue to a live episode than teacher-forced per-position accuracy, not a bug.

Usage:
    python scripts/eval_argument_accuracy.py \
        --config-path examples/speechlm2/conf/finetune --config-name s2s_duplex_stt_11b \
        --val-shard-path data/tau2_training_samples/shards-0828/val \
        --ckpt-path logs/sft_newdomains_0831/exp/checkpoints/step-500.ckpt \
        --output-json logs/eval_argacc_0831/step500.json
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
sys.path.insert(0, str(REPO / "scripts"))

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

from eval_forward import build_model_and_dataset, load_ckpt_into_model  # noqa: E402

TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.S)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--val-shard-path", required=True)
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--pretrained-s2s-model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-cuts", type=int, default=0)
    ap.add_argument("--use-llm", action="store_true",
                     help="Route argument judging through the Bedrock judge shim "
                          "(scripts/fdb_v3_evaluate.py's _BedrockJudge), same as FDB-v3's "
                          "--use-llm. Without this, exact-match fallback only.")
    ap.add_argument("--output-json", required=True)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# FDB-v3 algorithm, replicated from Full-Duplex-Bench/v3/evaluate_tool_calls.py and
# evaluate_pass_rate.py. Schema translated: our cuts use {"name", "arguments"}; FDB-v3's
# scorer uses {"function", "args"} -- translated once at extraction time (extract_calls),
# the scoring functions below match FDB-v3's own field names exactly.
# ---------------------------------------------------------------------------

def normalize(v):
    if isinstance(v, str):
        return v.lower().strip().replace("_", " ")
    return v


def exact_match_args(expected: dict, actual: dict):
    for key, exp_val in expected.items():
        if key not in actual:
            return False, f"Missing argument: {key}"
        if isinstance(exp_val, str) and exp_val.startswith("$"):
            continue
        if normalize(exp_val) != normalize(actual.get(key)):
            return False, f"Mismatch '{key}': expected={exp_val}, got={actual.get(key)}"
    return True, "All arguments match"


_judge = None


def llm_judge_args(expected_args: dict, actual_args: dict, function_name: str, judge):
    prompt = f"""You are evaluating whether an AI voice agent called a function with correct arguments.

Function: {function_name}
Expected arguments: {json.dumps(expected_args)}
Actual arguments: {json.dumps(actual_args)}

Rules:
1. Arguments that start with "$" are dynamic references -- any plausible real value passes.
2. Minor formatting differences are fine: "August 20" == "2026-08-20", "New York" == "new york".
3. Abbreviations and common aliases are acceptable ("Las Vegas" == "Vegas").
4. Numeric tolerance: +/-5% is acceptable.
5. doc_type: "driver_license" == "driver license".

Respond with ONLY a JSON object:
{{"correct": true/false, "explanation": "brief reason"}}"""
    try:
        resp = judge.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        result = json.loads(raw)
        return result["correct"], result["explanation"]
    except Exception as e:
        ok, why = exact_match_args(expected_args, actual_args)
        return ok, f"{why} (judge failed: {e})"


def evaluate_tool_selection(expected_calls, actual_calls):
    expected_names = [c["function"] for c in expected_calls]
    actual_names = [c["function"] for c in actual_calls]
    exp_remaining, act_remaining = list(expected_names), list(actual_names)
    matched = 0
    for fn in list(exp_remaining):
        if fn in act_remaining:
            matched += 1
            exp_remaining.remove(fn)
            act_remaining.remove(fn)
    total_expected, total_actual = len(expected_names), len(actual_names)
    recall = matched / total_expected if total_expected > 0 else 1.0
    precision = matched / total_actual if total_actual > 0 else 1.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return {
        "score": round(f1, 3), "recall": round(recall, 3), "precision": round(precision, 3),
        "matched": matched, "total_expected": total_expected, "total_actual": total_actual,
        "unmatched_expected": exp_remaining, "unexpected_calls": act_remaining,
    }


def evaluate_argument_accuracy(expected_calls, actual_calls, use_llm, judge):
    if not expected_calls:
        return {"score": 1.0, "note": "No expected calls", "details": []}
    actual_by_func = {}
    for ac in actual_calls:
        actual_by_func.setdefault(ac["function"], []).append(ac)
    call_scores = []
    for ec in expected_calls:
        func = ec["function"]
        expected_args = ec.get("args", {})
        if func not in actual_by_func or not actual_by_func[func]:
            call_scores.append({"function": func, "score": 0.0, "reason": "Function not called"})
            continue
        actual_call = actual_by_func[func].pop(0)
        actual_args = actual_call.get("args", {})
        if use_llm:
            is_ok, explanation = llm_judge_args(expected_args, actual_args, func, judge)
        else:
            is_ok, explanation = exact_match_args(expected_args, actual_args)
        call_scores.append({
            "function": func, "score": 1.0 if is_ok else 0.0,
            "expected_args": expected_args, "actual_args": actual_args, "explanation": explanation,
        })
    avg = sum(c["score"] for c in call_scores) / len(call_scores) if call_scores else 0.0
    return {"score": round(avg, 3), "details": call_scores}


def evaluate_pass_at_1(tool_sel: dict, arg_acc: dict) -> bool:
    if tool_sel["recall"] != 1.0 or tool_sel["precision"] != 1.0:
        return False
    return all(d["score"] == 1.0 for d in arg_acc.get("details", []))


# ---------------------------------------------------------------------------
# Extraction: cut supervisions -> expected_calls; model's free-running generation -> actual_calls
# ---------------------------------------------------------------------------

def extract_calls_from_toolcall_blocks(text: str):
    """Parse every <TOOLCALL>[...]</TOOLCALL> block into FDB-v3's {"function","args"} schema.

    Our schema is {"name": str, "arguments": dict} per call (confirmed by inspecting
    data/tau2_training_samples/shards-0828 cuts directly) -- renamed here, once, at the
    boundary, rather than threading two schemas through the scoring functions.
    """
    calls = []
    for block in TOOLCALL_RE.findall(text):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed = [parsed]
        for c in parsed:
            if isinstance(c, dict) and "name" in c:
                calls.append({"function": c["name"], "args": c.get("arguments", {}) or {}})
    return calls


def expected_calls_for_cut(cut):
    calls = []
    for sup in cut.supervisions:
        fn_str = (sup.custom or {}).get("function") or ""
        if "<TOOLCALL>" in fn_str:
            calls.extend(extract_calls_from_toolcall_blocks(fn_str))
    return calls


def run_model_free_running(model, batch, device):
    """offline_inference with every FC-position kwarg left None: no forced call timing, no
    pre-filled tool responses -- _expand_for_function_calling is skipped entirely
    (duplex_stt_model.py:5134 gates it on function_call_lengths/function_call_steps being
    not-None), so the model decides everything about tool use on its own."""
    def to_dev(x):
        if torch.is_tensor(x):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return x

    batch = to_dev(batch)
    model.eval()
    with torch.no_grad():
        out = model.offline_inference(
            batch["source_audio"], batch["source_audio_lens"],
            prompt_tokens=batch.get("prompt_tokens"),
            prompt_token_lens=batch.get("prompt_token_lens"),
            sample_id=batch.get("sample_id"),
        )
    return out


def main():
    args = parse_args()
    cfg_dir = str((REPO / args.config_path).resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=args.config_name)
    if args.pretrained_s2s_model:
        OmegaConf.update(cfg, "model.pretrained_s2s_model", args.pretrained_s2s_model)
    OmegaConf.update(cfg, "model.debug_fc", False, force_add=True)

    print(f"[argacc] building model (pretrained_s2s_model={cfg.model.pretrained_s2s_model})...",
          flush=True)
    model, dataset = build_model_and_dataset(cfg, args.val_shard_path)

    if args.ckpt_path:
        info = load_ckpt_into_model(model, args.ckpt_path, strict=False)
        print(f"[argacc] loaded ckpt {args.ckpt_path} "
              f"(missing={info['missing']}, unexpected={info['unexpected']})", flush=True)

    device = torch.device(args.device)
    model = model.to(dtype=torch.bfloat16, device=device)
    print(f"[argacc] model on {device}", flush=True)

    judge = None
    if args.use_llm:
        sys.path.insert(0, str(REPO / "scripts"))
        from fdb_v3_evaluate import _BedrockJudge, DEFAULT_JUDGE  # noqa
        judge = _BedrockJudge(DEFAULT_JUDGE)
        print(f"[argacc] using Bedrock judge: {DEFAULT_JUDGE}", flush=True)

    from lhotse import CutSet
    cuts = CutSet.from_shar(in_dir=str(args.val_shard_path)).to_eager()

    per_cut = []
    for i, cut in enumerate(cuts):
        if args.limit_cuts and i >= args.limit_cuts:
            break
        expected_calls = expected_calls_for_cut(cut)
        t0 = time.time()
        batch = dataset[CutSet.from_cuts([cut])]["audio_data"]
        out = run_model_free_running(model, batch, device)
        dt = time.time() - t0

        gen_function = out.get("tokens_function")
        lengths = out.get("tokens_len")
        function_text = ""
        if gen_function is not None:
            from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
            function_text = tokens_to_str(
                gen_function, lengths, tokenizer=model.tokenizer, pad_id=model.text_pad_id,
            )[0]
        actual_calls = extract_calls_from_toolcall_blocks(function_text)

        tool_sel = evaluate_tool_selection(expected_calls, actual_calls)
        arg_acc = evaluate_argument_accuracy(expected_calls, actual_calls, args.use_llm, judge)
        passed = evaluate_pass_at_1(tool_sel, arg_acc)

        row = {
            "cut_id": cut.id, "seconds": round(dt, 2),
            "n_expected": len(expected_calls), "n_actual": len(actual_calls),
            "tool_selection": tool_sel, "argument_accuracy": arg_acc, "pass_at_1": passed,
        }
        per_cut.append(row)
        print(f"  {cut.id:28s} exp={len(expected_calls)} act={len(actual_calls)} "
              f"tool_sel_f1={tool_sel['score']:.2f} arg_acc={arg_acc['score']:.2f} "
              f"pass@1={passed} ({dt:.1f}s)", flush=True)

    n = len(per_cut)
    mean_tool_sel = sum(r["tool_selection"]["score"] for r in per_cut) / n if n else 0.0
    mean_arg_acc = sum(r["argument_accuracy"]["score"] for r in per_cut) / n if n else 0.0
    pass_rate = sum(1 for r in per_cut if r["pass_at_1"]) / n if n else 0.0

    out = {
        "ckpt_path": args.ckpt_path, "val_shard_path": args.val_shard_path,
        "use_llm": args.use_llm, "n_cuts": n,
        "mean": {"tool_selection_f1": round(mean_tool_sel, 4),
                  "argument_accuracy": round(mean_arg_acc, 4),
                  "pass_at_1": round(pass_rate, 4)},
        "per_cut": per_cut,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[argacc] mean over {n} cuts: tool_selection_f1={mean_tool_sel:.3f} "
          f"argument_accuracy={mean_arg_acc:.3f} pass_at_1={pass_rate:.3f}")
    print(f"[argacc] wrote {args.output_json}")


if __name__ == "__main__":
    main()
