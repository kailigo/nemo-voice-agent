#!/usr/bin/env python
"""Score FDB-v3 arms on tool *names* and on names+arguments, to expose an eagerness artifact.

`evaluate_tool_calls.py::evaluate_tool_selection` -- the metric behind the card's 82.5 % -- is
multiset F1 over function *names*. Arguments never enter it. So a model that fires the right
name with a fabricated argument scores identically to one that gets both right, and a model
that declines to call rather than guess scores zero.

That is not hypothetical. On the 30 `hard` scenarios the research path beats the container by
11.8 points on names and *loses* on names+arguments (`FDB_V3_REPRODUCTION.md` §4). This script
is what produces both columns, so the reversal can be rechecked rather than taken on trust.

Two argument conventions, because the strict one alone would be easy to dismiss as an artifact
of demanding byte-equal dicts:

  lenient  the actual call must match every argument the expected call specifies; extra
           arguments the model volunteered are forgiven.
  strict   the argument dicts must match exactly.

Both normalise case, surrounding whitespace, leading articles ("the gym" == "gym") and
punctuation, and both drop expected calls whose arguments contain a `$RESULT_n` placeholder --
those refer to a previous call's output, which no result file can reproduce, so scoring them is
guaranteed-fail noise (13 of 72 expected calls in the hard slice). Dropping them changes the
name-only column slightly against §4's table, which keeps them; that is the only difference.

Matching is greedy and one-to-one: each expected call consumes at most one actual call, so
firing the same tool five times cannot earn five credits.

    python scripts/fdb_v3_name_vs_args.py --provider nemo_research_hard --provider nemo_rt
    python scripts/fdb_v3_name_vs_args.py --difficulty hard --show-mismatches
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
from pathlib import Path

V3 = Path("/fsx/home/kai.li/code/Full-Duplex-Bench/v3")
DEFAULT_PROVIDERS = ["nemo_research_hard", "nemo_rt", "nemo_rt_jinja"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--provider", action="append", default=None,
                   help=f"repeatable; default: {' '.join(DEFAULT_PROVIDERS)}")
    p.add_argument("--v3", type=Path, default=V3)
    p.add_argument("--data-dir", default="fdb_v3_data_released")
    p.add_argument("--difficulty", action="append", choices=["easy", "medium", "hard"],
                   default=None, help="repeatable; default: whatever the arms cover")
    p.add_argument("--intersect", action="store_true", default=True,
                   help="score only examples every named arm has a result for (default)")
    p.add_argument("--show-mismatches", action="store_true",
                   help="print calls that matched on name but not on arguments")
    return p.parse_args()


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def norm(v: object) -> object:
    """Compare arguments the way a human grader would, without a judge."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    s = re.sub(r"^(the|my|a|an)\s+", "", s)
    return re.sub(r"[^a-z0-9. ]", "", s).strip()


def normed(d: dict | None) -> dict:
    return {k: norm(v) for k, v in (d or {}).items()}


def expected_args(call: dict) -> dict:
    """Ground truth uses `arguments`; our result files use `args`. Accept either."""
    return call.get("arguments") or call.get("args") or {}


def has_placeholder(call: dict) -> bool:
    return any(isinstance(v, str) and v.startswith("$RESULT")
               for v in expected_args(call).values())


def multiset_name_f1(actual: list[str], expected: list[str]) -> float:
    a, e = collections.Counter(actual), collections.Counter(expected)
    m = sum((a & e).values())
    return f1(m / len(actual) if actual else 0.0, m / len(expected) if expected else 0.0)


def score_example(expected: list[dict], actual: list[dict],
                  mismatches: list | None = None, eid: str = "") -> tuple[float, float, float]:
    """(name-only F1, lenient name+args F1, strict name+args F1) for one example."""
    if not expected:
        return 0.0, 0.0, 0.0
    used: set[int] = set()
    m_lenient = m_strict = 0
    for ec in expected:
        want = normed(expected_args(ec))
        for i, ac in enumerate(actual):
            if i in used or ac["function"] != ec["function"]:
                continue
            got = normed(ac.get("args"))
            if all(k in got and got[k] == v for k, v in want.items()):
                used.add(i)
                m_lenient += 1
                m_strict += got == want
                break
        else:
            # Name fired but no argument-compatible instance: the artifact this script exists for
            if mismatches is not None and any(a["function"] == ec["function"] for a in actual):
                mismatches.append((eid, ec["function"], want,
                                   [normed(a.get("args")) for a in actual
                                    if a["function"] == ec["function"]]))
    n_a, n_e = len(actual), len(expected)
    return (multiset_name_f1([c["function"] for c in actual],
                             [c["function"] for c in expected]),
            f1(m_lenient / n_a if n_a else 0.0, m_lenient / n_e),
            f1(m_strict / n_a if n_a else 0.0, m_strict / n_e))


def main() -> int:
    args = parse_args()
    providers = args.provider or DEFAULT_PROVIDERS
    meta = {s["id"]: s for s in
            json.loads((args.v3 / "benchmark_data_v2.json").read_text())["scenarios"]}

    # Restrict to the dirs every arm covers, or the comparison is between different slices.
    dirsets = []
    for prov in providers:
        found = {os.path.dirname(p) for p in
                 glob.glob(str(args.v3 / args.data_dir / f"*/result_{prov}.json"))}
        if not found:
            print(f"no result_{prov}.json under {args.v3 / args.data_dir}")
            return 1
        dirsets.append(found)
    dirs = sorted(set.intersection(*dirsets) if args.intersect else set.union(*dirsets))
    if args.difficulty:
        wanted = set(args.difficulty)

        def difficulty_of(d: str) -> str:
            eid = json.loads(Path(d, f"result_{providers[0]}.json").read_text())["example_id"]
            return meta[eid]["difficulty"]

        dirs = [d for d in dirs if difficulty_of(d) in wanted]

    dropped = kept = 0
    print(f"n={len(dirs)} examples common to {', '.join(providers)}"
          f"{' (difficulty ' + '+'.join(sorted(args.difficulty)) + ')' if args.difficulty else ''}")
    rows = []
    for prov in providers:
        mismatches: list = [] if args.show_mismatches else None
        names, lenient, strict = [], [], []
        for d in dirs:
            r = json.loads(Path(d, f"result_{prov}.json").read_text())
            raw = meta[r["example_id"]]["expected_tool_calls"]
            exp = [c for c in raw if not has_placeholder(c)]
            dropped += len(raw) - len(exp)
            kept += len(exp)
            n, le, st = score_example(exp, r.get("actual_tool_calls") or [],
                                      mismatches, r["example_id"])
            names.append(n)
            lenient.append(le)
            strict.append(st)
        mean = lambda v: sum(v) / len(v) * 100  # noqa: E731
        rows.append((prov, mean(names), mean(lenient), mean(strict)))
        if mismatches:
            print(f"\n  {prov}: right name, wrong arguments ({len(mismatches)})")
            for eid, fn, want, got in mismatches:
                print(f"    {eid:<16} {fn}")
                print(f"      expected {json.dumps(want, sort_keys=True)}")
                for g in got:
                    print(f"      actual   {json.dumps(g, sort_keys=True)}")

    print(f"\n  expected calls scored: {kept}, dropped as $RESULT placeholders: "
          f"{dropped // len(providers)}\n")
    print(f"  {'arm':<24}{'name-only':>11}{'name+args lenient':>19}{'strict':>9}")
    for prov, n, le, st in rows:
        print(f"  {prov:<24}{n:>10.1f}%{le:>18.1f}%{st:>8.1f}%")

    best_name = max(rows, key=lambda r: r[1])[0]
    best_args = max(rows, key=lambda r: r[2])[0]
    print(f"\n  best on names: {best_name}   best on names+args: {best_args}"
          f"{'   <-- REVERSED' if best_name != best_args else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
