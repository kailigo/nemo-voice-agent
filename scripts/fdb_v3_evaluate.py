#!/usr/bin/env python3
"""Run Full-Duplex-Bench v3's own evaluators, with a Bedrock judge in place of gpt-4o.

Why a shim instead of just running their scripts
------------------------------------------------
Two of the three published numbers need an LLM judge:

  * Tool Selection (82.5% on the model card) is pure set arithmetic over tool names.
    `evaluate_tool_calls.py::evaluate_tool_selection` calls nothing. This number is
    reproducible with no credentials at all -- run their script directly.
  * Argument accuracy (42.2%) and Pass@1 (33%) route every argument comparison through
    `gpt-4o` ("semantic argument matching": "August 20" == "2026-08-20", "Vegas" ==
    "Las Vegas", +-5% numeric tolerance). Without `--use-llm` they silently fall back to
    `exact_match_args`, which would understate both by an unknown margin -- so running
    without a judge is not a neutral choice, it is a different metric.

We have no OpenAI key. We do have Bedrock through the instance IAM role. The judge is a
plain text-in/JSON-out classifier, so substituting Claude changes the judge but not the
metric definition. That IS a deviation from the published setup and it is printed at the
top of every run and recorded in the report: a stricter or looser judge moves argument
accuracy directly.

Their scripts are imported and patched at the one seam each uses to obtain a client
(`_get_openai_client`, or the module-level `OpenAI` in the latency script). Nothing in
Full-Duplex-Bench/ is edited.

Usage:
    python scripts/fdb_v3_evaluate.py --provider nemo                 # all three steps
    python scripts/fdb_v3_evaluate.py --provider nemo --no-llm        # tool selection only
    python scripts/fdb_v3_evaluate.py --provider nemo --steps tool_calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

FDB_V3_DIR = Path("/fsx/home/kai.li/code/Full-Duplex-Bench/v3")
DEFAULT_JUDGE = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"


class _BedrockJudge:
    """Enough of the OpenAI client surface for the three call sites that use it.

    All three do exactly `client.chat.completions.create(model="gpt-4o", messages=[...],
    temperature=0, max_tokens=N)` and read `.choices[0].message.content`. litellm's
    ModelResponse has that shape, so the response needs no adaptation -- only the request
    model name is overridden, and it is overridden loudly (`self.calls` is reported).
    """

    def __init__(self, model: str):
        import litellm

        self._litellm = litellm
        self.model = model
        self.calls = 0
        self.failures = 0
        self.chat = self  # client.chat.completions.create(...)
        self.completions = self

    def create(self, model=None, messages=None, **kwargs):
        self.calls += 1
        kwargs.pop("model", None)
        try:
            return self._litellm.completion(
                model=self.model, messages=messages, num_retries=4, **kwargs
            )
        except Exception:
            self.failures += 1
            raise


def _patch(module, judge) -> None:
    """Redirect a benchmark evaluator's judge to `judge`, whichever seam it uses."""
    patched = []
    if hasattr(module, "_get_openai_client"):
        module._get_openai_client = lambda: judge
        patched.append("_get_openai_client")
    if hasattr(module, "OpenAI"):
        module.OpenAI = lambda *a, **k: judge
        patched.append("OpenAI")
    if not patched:
        raise RuntimeError(
            f"{module.__name__} exposes neither _get_openai_client nor OpenAI; the judge "
            f"would silently stay pointed at OpenAI. Re-read the file."
        )
    print(f"   patched {module.__name__}: {', '.join(patched)}", flush=True)


def _run(module_name: str, argv: list[str], judge) -> None:
    import importlib

    module = importlib.import_module(module_name)
    if judge is not None:
        _patch(module, judge)
    saved = sys.argv
    sys.argv = [module_name] + argv
    try:
        module.main()
    finally:
        sys.argv = saved


def coverage(provider: str) -> None:
    """Report how much of the benchmark actually has a result, before scoring it.

    `evaluate_tool_calls.py:640-645` globs for result files and silently drops any scenario
    that has none. A half-finished run therefore produces a confident-looking accuracy over
    half the data, with nothing in the report to say so. This is the one check that catches
    a dead shard, so it runs unconditionally.

    Note the released data covers 79 of the benchmark's 100 scenario ids across 100 audio
    folders (21 ids have two speaker renditions, 21 have no audio at all). 100 result files
    is therefore full coverage, and matches the README's "all 100 audio samples".
    """
    import collections

    data_dir = FDB_V3_DIR / "fdb_v3_data_released"
    dirs = [d for d in data_dir.iterdir() if d.is_dir() and (d / "input.wav").exists()]
    results = [d / f"result_{provider}.json" for d in dirs]
    present = [r for r in results if r.exists()]
    statuses = collections.Counter()
    silent = 0
    for r in present:
        data = json.loads(r.read_text())
        statuses[data.get("status", "?")] += 1
        if not data.get("transcript", "").strip() and not data.get("asr_chunks"):
            silent += 1

    print(f"\ncoverage: {len(present)}/{len(dirs)} audio folders have result_{provider}.json")
    for status, n in statuses.most_common():
        print(f"   {status}: {n}")
    print(f"   silent (no transcript and no chunks -> scores 0 on everything): {silent}")
    if len(present) < len(dirs):
        print(f"   WARNING: {len(dirs) - len(present)} missing. The evaluators drop these from "
              f"the denominator, so every number below is over a subset.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default="nemo")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    p.add_argument("--no-llm", action="store_true",
                   help="Skip the judge entirely. Tool selection is unaffected; argument "
                        "accuracy falls back to exact string match and response quality "
                        "is not scored.")
    p.add_argument("--steps", nargs="+", default=["tool_calls", "pass_rate", "latency"],
                   choices=["tool_calls", "pass_rate", "latency"])
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where the reports go. Defaults to nemo-voice-agent/logs/fdb_v3, "
                        "so the third-party checkout stays clean.")
    args = p.parse_args()

    out_dir = args.out_dir or (Path(__file__).resolve().parent / ".." / "logs" / "fdb_v3").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Their scripts resolve --results-dir and --benchmark relative to the cwd.
    os.chdir(FDB_V3_DIR)
    sys.path.insert(0, str(FDB_V3_DIR))

    judge = None
    if not args.no_llm:
        judge = _BedrockJudge(args.judge_model)
        print("=" * 78)
        print("JUDGE SUBSTITUTION: the published numbers used gpt-4o as the LLM judge.")
        print(f"This run uses {args.judge_model} (no OpenAI key; Bedrock authenticates")
        print("off the instance IAM role). Tool Selection does not use the judge and is")
        print("unaffected. Argument accuracy and Pass@1 do, and are not strictly")
        print("comparable to the model card until re-run with gpt-4o.")
        print("=" * 78, flush=True)

    coverage(args.provider)

    common = ["--results-dir", "fdb_v3_data_released", "--provider", args.provider]
    use_llm = [] if args.no_llm else ["--use-llm"]

    if "tool_calls" in args.steps:
        print("\n--- [1] evaluate_tool_calls.py (tool selection, argument accuracy) ---", flush=True)
        _run("evaluate_tool_calls",
             ["--benchmark", "benchmark_data_v2.json", *common,
              "--output", str(out_dir / f"{args.provider}_evaluation_report.json"), *use_llm],
             judge)

    if "pass_rate" in args.steps:
        print("\n--- [2] evaluate_pass_rate.py (Pass@1) ---", flush=True)
        _run("evaluate_pass_rate",
             ["--benchmark", "benchmark_data_v2.json", *common,
              "--output", str(out_dir / f"{args.provider}_pass_rate_report.json"), *use_llm],
             judge)

    if "latency" in args.steps:
        # Only meaningful with --asr-input inference: without user_speech_end_rel every
        # sample reports "unavailable" rather than a made-up zero.
        print("\n--- [3] analyze_tool_latency.py ---", flush=True)
        _run("analyze_tool_latency",
             [*common, "--output", str(out_dir / f"{args.provider}_latency_report.json")],
             judge)

    if judge is not None:
        print(f"\njudge calls: {judge.calls} ({judge.failures} failed) via {judge.model}")

    # A one-screen comparison against the model card, so the answer to "did we reproduce
    # it" does not require opening three JSON files.
    card = {"tool_selection_acc": 0.825, "argument_acc": 0.422, "pass_rate": 0.33}
    report_path = out_dir / f"{args.provider}_evaluation_report.json"
    pass_path = out_dir / f"{args.provider}_pass_rate_report.json"
    if report_path.exists():
        rep = json.loads(report_path.read_text())
        bm = rep["by_metric"]
        print("\n" + "=" * 78)
        print(f"{args.provider} vs the NVIDIA-NemotronLabs-VoiceChat-11B model card")
        print("=" * 78)
        tt = rep["turn_taking"]
        print(f"turn-taking: {tt['turn_taken']}/{tt['total']} samples produced any output "
              f"({tt['turn_take_rate']}); the headline metrics are over those only.")
        print(f"\n{'metric':<26} {'ours':>10} {'card':>10}   n")
        # Both variants are printed because the card does not say which it quotes:
        # `*_all` scores a silent sample 0, the default drops it from the denominator.
        rows = [
            ("tool_selection_acc", bm.get("tool_selection_acc"), card["tool_selection_acc"],
             tt["turn_taken"]),
            ("  ...incl. no-response", bm.get("tool_selection_acc_all"), card["tool_selection_acc"],
             tt["total"]),
            ("argument_acc", bm.get("argument_acc"), card["argument_acc"], tt["turn_taken"]),
            ("  ...incl. no-response", bm.get("argument_acc_all"), card["argument_acc"], tt["total"]),
        ]
        if pass_path.exists():
            pr = json.loads(pass_path.read_text())
            rows.append(("pass_rate (Pass@1)", pr.get("overall_pass_rate"), card["pass_rate"],
                         pr.get("total_scenarios")))
        for name, ours, theirs, n in rows:
            ours_s = f"{ours:.1%}" if isinstance(ours, (int, float)) else "n/a"
            print(f"{name:<26} {ours_s:>10} {theirs:>9.1%}   {n}")
        print(f"\nreports in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
