#!/usr/bin/env python
"""Score FDB-v3 Argument accuracy with an LLM judge, so the card's 44.2 % is comparable.

`evaluate_tool_calls.py --use-llm` is how the card's Argument accuracy was produced: its judge
is instructed to forgive `$RESULT` dynamic references, date formats ("August 20" == "2026-08-20"),
aliases ("Las Vegas" == "Vegas"), +-5 % numeric drift and underscore-vs-space. The default
`exact_match_args` forgives none of that, so the 22.2 % / 36.7 % / 32.8 % we measured are a strict
LOWER BOUND on the same behaviour, not a deficit against 44.2 %.

We cannot reproduce the card's judge: it is `gpt-4o`, there is no OpenAI key on this cluster and
Bedrock's catalogue has no gpt-4o. So this reports OUR judge reading THE CARD'S rubric, and it runs
two judges from different families to expose judge-induced bias rather than hide it. Agreement
within a point or two means the number is robust; divergence IS the finding, and then the honest
output is a range.

    python scripts/fdb_v3_judge_args.py                       # both judges, all three arms
    python scripts/fdb_v3_judge_args.py --judge sonnet        # one judge
    python scripts/fdb_v3_judge_args.py --provider nemo_rt --show-flips

WHY IT DOES NOT JUST CALL `--use-llm`: `llm_judge_argument` wraps its API call and its JSON parse
in a bare `except Exception: return exact_match_args(...)`. A truncated response, a preamble before
the JSON, a throttle or a bad credential therefore degrades to exact-match SILENTLY, and the run
reports ~22 % looking like it judged. gpt-oss in particular emits a reasoning preamble and blew
through a 64-token cap in testing. So every verdict here is accounted for -- judged, parse-failed
or api-failed -- and a nonzero fallback count suppresses the number instead of averaging over it.

WHY TWO EVALUATOR PASSES: `evaluate_argument_accuracy` pairs expected to actual calls FIFO by
function name and skips the judge entirely for functions that were never called. Re-deriving that
pairing here to batch the API calls would be a second implementation of it, and a divergence would
be invisible. Instead pass 1 runs the real evaluator with a recording judge that makes no API calls,
which enumerates exactly the triples the real run will ask about; those are judged concurrently;
pass 2 replays through a cache whose misses raise. A wrong enumeration fails loudly.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import importlib.util
import inspect
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

V3 = Path("/fsx/home/kai.li/code/Full-Duplex-Bench/v3")
DEFAULT_PROVIDERS = ["nemo_research", "nemo_rt", "nemo_rt_jinja"]

# Different families on purpose. Sonnet is the credential path already verified in this repo
# (tau2_smoke_nemo.sh points all three tau-voice eval models at it); gpt-oss is the closest
# OpenAI-lineage model in Bedrock's catalogue, which is the useful cross-check against gpt-4o.
JUDGES = {
    "sonnet":  "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "gpt-oss": "bedrock/openai.gpt-oss-120b-1:0",
    "haiku":   "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",  # opt-in tie-break
}
DEFAULT_JUDGES = ["sonnet", "gpt-oss"]

# Verbatim from evaluate_tool_calls.py::llm_judge_argument. It MUST stay byte-identical or we are
# scoring a different rubric than the card; assert_rubric_unchanged() checks for silent drift.
PROMPT = """You are evaluating whether an AI voice agent called a function with correct arguments.

Function: {function_name}
Expected arguments: {expected}
Actual arguments: {actual}

Rules:
1. Arguments that start with "$" (like "$RESULT_0.flights[0].flight_id") are dynamic references —
   the actual value should be any real value that could plausibly come from a previous API call.
2. Minor formatting differences are fine: "August 20" == "2026-08-20", "New York" == "new york".
3. "Las Vegas" == "Vegas" — abbreviations and common aliases are acceptable.
4. Numeric tolerance: ±5% is acceptable.
5. doc_type: "driver_license" == "driver license" (underscore vs space).

Respond with ONLY a JSON object:
{{"correct": true/false, "explanation": "brief reason"}}"""

RUBRIC_MARKERS = [
    'Arguments that start with "$"',
    '"Las Vegas" == "Vegas"',
    "Numeric tolerance",
    '"driver_license" == "driver license"',
]


def load_evaluator():
    """Import the benchmark's evaluator by path; it is not a package and not on sys.path."""
    spec = importlib.util.spec_from_file_location("fdb_eval", V3 / "evaluate_tool_calls.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fdb_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def assert_rubric_unchanged(mod) -> None:
    src = inspect.getsource(mod.llm_judge_argument)
    missing = [m for m in RUBRIC_MARKERS if m not in src]
    if missing:
        print("WARNING: the benchmark's judge rubric has changed; PROMPT above is now a different\n"
              f"         rubric than --use-llm would apply. Missing markers: {missing}")


def parse_verdict(text: str) -> bool:
    """Tolerate fences and a reasoning preamble; raise if there is no parseable JSON verdict.

    Deliberately strict about the *verdict* while lenient about wrapping: guessing here is how a
    judge failure turns into a silent exact-match number.
    """
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    m = re.search(r'\{[^{}]*"correct"\s*:\s*(true|false)[^{}]*\}', s, re.IGNORECASE | re.DOTALL)
    if m:
        return json.loads(m.group(0).replace("True", "true").replace("False", "false"))["correct"]
    m = re.search(r'"correct"\s*:\s*(true|false)', s, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    raise ValueError(f"no verdict in {text[:200]!r}")


class Judge:
    """One judge model. Counts every outcome so a degraded run cannot look like a clean one."""

    def __init__(self, model: str, retries: int = 4):
        import litellm
        litellm.suppress_debug_info = True
        self._litellm = litellm
        self.model = model
        self.retries = retries
        self.audit: list[dict] = []
        self.counts = collections.Counter()
        self._lock = threading.Lock()

    def __call__(self, key: tuple[str, str, str]) -> bool | None:
        function_name, expected, actual = key
        prompt = PROMPT.format(function_name=function_name, expected=expected, actual=actual)
        last = ""
        for attempt in range(self.retries):
            try:
                r = self._litellm.completion(
                    model=self.model, temperature=0,
                    # 512, not the evaluator's 200: gpt-oss prefaces its JSON with reasoning and a
                    # truncated body is indistinguishable from a refusal after parsing.
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = r.choices[0].message.content or ""
                # An empty completion is retryable, not a verdict. Bedrock's content filter
                # returns one for some inputs (a passport-number comparison, in practice) and
                # parse_verdict would otherwise class it as unparseable on the first try.
                if not text.strip():
                    raise ValueError("empty completion")
                verdict = parse_verdict(text)
                with self._lock:
                    self.counts["judged"] += 1
                    self.audit.append({"function": function_name, "expected": expected,
                                       "actual": actual, "verdict": verdict, "raw": text[:400]})
                return verdict
            except Exception as e:  # throttles and transient 5xx are the common case
                last = f"{type(e).__name__}: {e}"
                if attempt < self.retries - 1 and _retryable(e):
                    time.sleep(2 ** attempt)
                    continue
                break
        with self._lock:
            self.counts["parse_fail" if "no verdict" in last else "api_fail"] += 1
            self.audit.append({"function": function_name, "expected": expected,
                               "actual": actual, "verdict": None, "error": last})
        return None


def _retryable(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return any(t in s for t in ("throttl", "toomanyrequests", "timeout", "429", "500", "503",
                                "serviceunavailable", "connection", "empty completion"))


def build_entries(mod, provider: str) -> tuple[dict, list]:
    """Same discovery as the evaluator's main(): rglob result_<provider>.json under the data dir."""
    benchmark = json.loads((V3 / "benchmark_data_v2.json").read_text())
    smap = {s["id"]: s for s in benchmark["scenarios"]}
    entries = []
    for f in sorted((V3 / "fdb_v3_data_released").rglob(f"result_{provider}.json")):
        data = json.loads(f.read_text())
        eid = data.get("example_id")
        if eid in smap:
            entries.append({"scenario": smap[eid], "calls": data.get("actual_tool_calls", []),
                            "transcript": data.get("transcript", ""), "result_data": data})
    return benchmark, entries


def arg_acc(mod, benchmark, entries, judge_fn) -> tuple[float, float, int, list]:
    """Run the real evaluator with `judge_fn` installed. Returns (turn_taken, all, n, results).

    judge_fn=None means the stock exact-match path (use_llm=False) rather than a patched judge --
    `exact_match_args` cannot be passed here directly, since the judge hook is called with three
    positional args (expected, actual, function_name) and exact_match_args takes two.
    """
    orig_arg, orig_resp = mod.llm_judge_argument, mod.evaluate_response_quality
    if judge_fn is not None:
        mod.llm_judge_argument = judge_fn
    # use_llm=True would also fire the response-quality judge. Pin it to the non-LLM behaviour so
    # we pay only for argument verdicts; evaluate_all_v2 already filters score=None out.
    mod.evaluate_response_quality = lambda scenario, transcript, use_llm=False: {
        "score": None, "explanation": "response judge disabled by fdb_v3_judge_args.py"}
    try:
        report = mod.evaluate_all_v2(benchmark, entries, use_llm=judge_fn is not None)
    finally:
        mod.llm_judge_argument, mod.evaluate_response_quality = orig_arg, orig_resp
    bm = report["by_metric"]
    return (bm["argument_acc"], bm["argument_acc_all"],
            report["turn_taking"]["turn_taken"], report["scenario_results"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", action="append", default=None,
                   help=f"repeatable; default: {' '.join(DEFAULT_PROVIDERS)}")
    p.add_argument("--judge", action="append", choices=sorted(JUDGES), default=None,
                   help=f"repeatable; default: {' '.join(DEFAULT_JUDGES)}")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--audit-dir", type=Path,
                   default=Path("/fsx/home/kai.li/code/nemo-voice-agent/logs/fdb_v3_judge"))
    p.add_argument("--show-flips", action="store_true",
                   help="print the calls exact-match rejected and the judge accepted")
    p.add_argument("--dry-run", action="store_true",
                   help="enumerate the judge calls and print the exact-match baseline, then stop")
    p.add_argument("--max-missing-frac", type=float, default=0.05,
                   help="tolerate this fraction of unobtainable verdicts by reporting an interval "
                        "instead of a point; above it the judge's number is suppressed entirely")
    args = p.parse_args()
    providers = args.provider or DEFAULT_PROVIDERS
    judges = args.judge or DEFAULT_JUDGES

    mod = load_evaluator()
    assert_rubric_unchanged(mod)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    # --- exact-match baseline: the lower bound every judged number is compared against --------
    print("collecting the exact-match baseline and the triples the judge will be asked about")
    base, cases = {}, {}
    for prov in providers:
        benchmark, entries = build_entries(mod, prov)
        if not entries:
            print(f"  no results for {prov}; skipping")
            continue
        tt, _all, n, _ = arg_acc(mod, benchmark, entries, None)

        # Pass 1: record, no API calls. Returning a constant is safe -- we throw the scores away.
        seen: list[tuple[str, str, str]] = []
        def record(expected_args, actual_args, function_name, _seen=seen):
            _seen.append((function_name, json.dumps(expected_args), json.dumps(actual_args)))
            return False, "recording"
        arg_acc(mod, benchmark, entries, record)

        base[prov] = (tt, n, benchmark, entries)
        cases[prov] = seen
        print(f"  {prov:<16} exact-match {tt:6.1%} on {n} turn-taken, {len(seen)} judge calls")

    todo = sorted({c for v in cases.values() for c in v})
    print(f"\n{len(todo)} distinct verdicts needed per judge "
          f"({sum(len(v) for v in cases.values())} calls before dedup)")
    if args.dry_run:
        for fn, e, a in todo[:5]:
            print(f"  {fn}\n    expected {e}\n    actual   {a}")
        return 0

    summary: dict[str, dict] = {}
    for jname in judges:
        judge = Judge(JUDGES[jname])
        print(f"\n=== judge {jname}  ({JUDGES[jname]}) ===")
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            verdicts = dict(zip(todo, ex.map(judge, todo)))
        c = judge.counts
        print(f"  {c['judged']} judged, {c['parse_fail']} unparseable, {c['api_fail']} api-failed "
              f"in {time.time() - t0:.0f}s")
        (args.audit_dir / f"audit_{jname}.jsonl").write_text(
            "\n".join(json.dumps(a) for a in judge.audit) + "\n")

        # A verdict we could not get is not a verdict. Under --use-llm it would silently become
        # exact-match; here it is either bounded or the number is suppressed, never averaged over.
        bad = c["parse_fail"] + c["api_fail"]
        if bad > max(1, int(args.max_missing_frac * len(todo))):
            print(f"  REFUSING to report a number: {bad}/{len(todo)} verdicts are missing, more "
                  f"than the {args.max_missing_frac:.0%} that can be usefully bounded. See "
                  f"{args.audit_dir / f'audit_{jname}.jsonl'}")
            summary[jname] = {"suppressed": bad}
            continue
        if bad:
            print(f"  {bad} verdict(s) unobtainable; scoring them both ways to bound the result")

        # Pass 2: replay through the cache. A miss means pass 1 enumerated the wrong triples.
        # `missing_as` substitutes for verdicts we never got: False gives the lower bound on the
        # arm's score, True the upper, so the true value is inside the interval by construction.
        for prov in base:
            tt_exact, n, benchmark, entries = base[prov]
            bounds, flips = [], []

            def replay(missing_as: bool, _f: list) -> float:
                def cached(expected_args, actual_args, function_name):
                    key = (function_name, json.dumps(expected_args), json.dumps(actual_args))
                    if key not in verdicts:
                        raise KeyError(f"pass 1 did not enumerate {key}; the batching is wrong")
                    v = verdicts[key]
                    v = missing_as if v is None else v
                    em, _ = mod.exact_match_args(expected_args, actual_args)
                    if v != em:
                        _f.append((function_name, expected_args, actual_args, em, v))
                    return v, "judged"
                return arg_acc(mod, benchmark, entries, cached)[0]

            bounds.append(replay(False, flips))
            bounds.append(replay(True, []) if bad else bounds[0])
            lo, hi = min(bounds), max(bounds)
            up = sum(1 for f in flips if f[4] and not f[3])
            down = sum(1 for f in flips if f[3] and not f[4])
            summary.setdefault(jname, {})[prov] = (tt_exact, lo, hi, up, down)
            shown = f"{lo:.1%}" if lo == hi else f"{lo:.1%}-{hi:.1%}"
            print(f"  {prov:<16} exact {tt_exact:6.1%} -> judged {shown:>13}   "
                  f"(+{up} accepted, -{down} rejected, of {len(cases[prov])})")
            if args.show_flips:
                for fn, e, a, _em, v in flips:
                    print(f"     {'ACCEPT' if v else 'REJECT'} {fn}\n"
                          f"       expected {json.dumps(e, sort_keys=True)}\n"
                          f"       actual   {json.dumps(a, sort_keys=True)}")

    # --- verdict ------------------------------------------------------------------------------
    print(f"\n{'=' * 78}\nArgument accuracy vs the card's 44.2 % (turn-taken denominator)\n{'=' * 78}")
    usable = [j for j in judges if j in summary and "suppressed" not in summary[j]]
    print(f"  {'arm':<18}{'exact-match':>13}" + "".join(f"{j:>16}" for j in usable))
    for prov in base:
        row = f"  {prov:<18}{base[prov][0]:>12.1%}"
        for j in usable:
            _e, lo, hi, _u, _d = summary[j][prov]
            row += f"{(f'{lo:.1%}' if lo == hi else f'{lo:.1%}-{hi:.1%}'):>16}"
        print(row)
    if len(usable) >= 2:
        # Compare the intervals, not their midpoints: two judges "agree" only if their bounds
        # overlap or nearly do, otherwise the disagreement is real and a point estimate hides it.
        a, b = usable[0], usable[1]
        spread = max(max(summary[a][p][1] - summary[b][p][2],
                         summary[b][p][1] - summary[a][p][2], 0.0) for p in base)
        print(f"\n  worst-arm gap between {a} and {b}: {spread:.1%} — "
              + ("intervals overlap or nearly so; quote the number" if spread <= 0.02 else
                 "judges DISAGREE; quote a range and name the judge"))
    for j in judges:
        if summary.get(j, {}).get("suppressed"):
            print(f"\n  {j}: SUPPRESSED, {summary[j]['suppressed']} verdicts missing")
    print(f"\n  audit logs: {args.audit_dir}")
    print("  NOTE: the card's judge is gpt-4o, which is unavailable here (no OpenAI key, not in")
    print("  Bedrock's catalogue). These are our judges on the card's rubric, not the card's number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
