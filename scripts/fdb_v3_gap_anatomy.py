#!/usr/bin/env python
"""Anatomy of the FDB-v3 Tool Selection gap: where the missing ~11 points are.

`FDB_V3_REPRODUCTION.md` §4 eliminates eleven candidate deployment discrepancies (checkpoint
provenance, prompt branch, capture window, VAD layer, latency profile, sample rate, ASR
quality, tool schemas, instructions, tool-result payloads, reject accounting, decode config).
Nothing in the plumbing explains the gap, so this script asks what the remaining behaviour
looks like -- and prints every number that document quotes, so they can be rechecked.

Four questions, in the order they narrow the answer:

  1. Is it recall or precision? Split by how many calls the scenario expects. Recall turns out
     flat across 1/2/3-call scenarios, so the model is *not* failing to chain; precision
     collapses. The arithmetic then closes off recall entirely -- reaching F1 0.825 at our
     precision needs recall > 1.
  2. What are the excess calls? Every call is classified as matched / never-expected /
     exact-duplicate / over-count-with-new-args. `--show-spurious` prints the offenders with
     their arguments, which is what shows they are scenario-grounded hallucinations rather
     than mislabelled ground truth.
  3. What would fixing them buy? An oracle deletes spurious and duplicate calls while keeping
     recall untouched, giving an upper bound on any precision-side fix.
  4. Is the metric convention itself the gap? Seven alternative conventions plus per-scenario
     aggregation, since the released data has 100 recordings over only 79 distinct scenarios.

The multiset-F1 column reproduces `evaluate_tool_calls.py::evaluate_tool_selection` exactly
(71.7 % on `nemo_rt`); if it ever stops doing so, the rest of the output is not trustworthy.

    python scripts/fdb_v3_gap_anatomy.py --provider nemo_rt --provider nemo_rt_jinja
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics as st
from pathlib import Path

V3 = Path("/fsx/home/kai.li/code/Full-Duplex-Bench/v3")
CARD_TOOL_SELECTION = 0.825

# The four domains the benchmark's own instructions advertise ("12 APIs across 4 domains"),
# used only to say whether a spurious call is cross-domain -- which 31 of 43 turn out to be.
DOMAIN = {
    "search_flights": "travel", "book_flight": "travel", "update_identity_doc": "travel",
    "get_card_benefits": "finance", "get_exchange_rate": "finance", "modify_autopay": "finance",
    "search_apartments": "housing", "calculate_commute": "housing",
    "update_search_filter": "housing",
    "track_order": "ecom", "search_products": "ecom", "add_to_cart": "ecom",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--provider", action="append", default=None,
                   help="repeatable; default: nemo_rt and nemo_rt_jinja")
    p.add_argument("--v3", type=Path, default=V3)
    p.add_argument("--data-dir", default="fdb_v3_data_released")
    p.add_argument("--show-spurious", action="store_true",
                   help="list every never-expected call with its arguments and timestamp")
    return p.parse_args()


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def multiset_f1(actual: list[str], expected: list[str]) -> float:
    """`evaluate_tool_calls.py:180-215` -- multiset intersection, harmonic mean."""
    a, e = collections.Counter(actual), collections.Counter(expected)
    matched = sum((a & e).values())
    return f1(matched / len(actual) if actual else 0.0,
              matched / len(expected) if expected else 0.0)


def load(v3: Path, data_dir: str, provider: str) -> tuple[list[dict], dict]:
    scenarios = json.loads((v3 / "benchmark_data_v2.json").read_text())["scenarios"]
    meta = {s["id"]: s for s in scenarios}
    rows = []
    for path in sorted(glob.glob(str(v3 / data_dir / f"*/result_{provider}.json"))):
        d = json.loads(Path(path).read_text())
        eid = d["example_id"]
        chunks = d.get("input_asr_chunks") or []
        rows.append({
            "dirname": os.path.basename(os.path.dirname(path)),
            "eid": eid,
            "expected": [c["function"] for c in meta[eid]["expected_tool_calls"]],
            "difficulty": meta[eid]["difficulty"],
            # end of *real* user audio: input_duration_s includes our 20 s trailing silence
            "audio_end": chunks[-1]["timestamp"][1] if chunks else None,
            "calls": d.get("actual_tool_calls") or [],
            "rejected": d.get("rejected_tool_calls") or [],
            "dropped": d.get("dropped_calls", 0) or 0,
        })
    return rows, meta


def by_expected_count(rows: list[dict]) -> None:
    """Question 1: recall or precision? Bucketed by scenario call count."""
    buckets = collections.defaultdict(lambda: dict(n=0, matched=0, exp=0, act=0, f1=0.0))
    for r in rows:
        e = collections.Counter(r["expected"])
        a = collections.Counter(c["function"] for c in r["calls"])
        b = buckets[len(r["expected"])]
        b["n"] += 1
        b["matched"] += sum((a & e).values())
        b["exp"] += len(r["expected"])
        b["act"] += len(r["calls"])
        b["f1"] += multiset_f1([c["function"] for c in r["calls"]], r["expected"])

    print(f"  {'#exp':>5}{'n':>5}{'exp':>6}{'act':>6}{'matched':>9}{'recall':>9}"
          f"{'prec':>8}{'meanF1':>9}")
    tot = collections.Counter()
    for k in sorted(buckets):
        b = buckets[k]
        print(f"  {k:>5}{b['n']:>5}{b['exp']:>6}{b['act']:>6}{b['matched']:>9}"
              f"{b['matched'] / b['exp'] * 100:>8.1f}%"
              f"{(b['matched'] / b['act'] * 100 if b['act'] else 0):>7.1f}%"
              f"{b['f1'] / b['n'] * 100:>8.1f}%")
        for key in ("n", "matched", "exp", "act"):
            tot[key] += b[key]
    prec, rec = tot["matched"] / tot["act"], tot["matched"] / tot["exp"]
    print(f"  {'all':>5}{tot['n']:>5}{tot['exp']:>6}{tot['act']:>6}{tot['matched']:>9}"
          f"{rec * 100:>8.1f}%{prec * 100:>7.1f}%")

    # Solving f1(p, r) = target for the missing side. > 1.0 means that side cannot close it.
    t = CARD_TOOL_SELECTION
    need_r = t * prec / (2 * prec - t) if 2 * prec > t else float("inf")
    need_p = t * rec / (2 * rec - t) if 2 * rec > t else float("inf")
    print(f"\n  to reach F1={t:.3f} at our pooled precision {prec:.3f}: recall must be "
          f"{need_r:.3f} (we have {rec:.3f}){'  <-- impossible' if need_r > 1 else ''}")
    print(f"  to reach F1={t:.3f} at our pooled recall {rec:.3f}: precision must be "
          f"{need_p:.3f} (we have {prec:.3f})")


def classify(rows: list[dict], show: bool) -> None:
    """Question 2: what are the excess calls?"""
    cat = collections.Counter()
    cross = same = 0
    gaps: list[float] = []
    dup_examples = 0
    spurious: list[tuple] = []
    pos_matched: list[float] = []
    pos_wrong: list[float] = []

    for r in rows:
        e = collections.Counter(r["expected"])
        seen_args: set = set()
        seen_fn: collections.Counter = collections.Counter()
        had_dup = False
        for c in r["calls"]:
            fn = c["function"]
            key = (fn, json.dumps(c.get("args"), sort_keys=True))
            frac = (c["timestamp_start"] / r["audio_end"]) if r["audio_end"] else None
            if key in seen_args:
                cat["exact duplicate (same fn+args)"] += 1
                had_dup = True
                first = next(x for x in r["calls"]
                             if (x["function"], json.dumps(x.get("args"), sort_keys=True)) == key)
                gaps.append(round(c["timestamp_start"] - first["timestamp_start"], 1))
            elif e[fn] == 0:
                cat["function not expected at all"] += 1
                if DOMAIN.get(fn) in {DOMAIN.get(x) for x in e}:
                    same += 1
                else:
                    cross += 1
                spurious.append((r["eid"], c["timestamp_start"], fn, c.get("args")))
            elif seen_fn[fn] >= e[fn]:
                cat["over-count of an expected fn (new args)"] += 1
            else:
                cat["matched an expected slot"] += 1
            seen_fn[fn] += 1
            seen_args.add(key)
            if frac is not None:
                (pos_wrong if e[fn] == 0 else pos_matched).append(frac)
        dup_examples += had_dup

    total = sum(cat.values())
    print(f"  {total} calls total")
    for k, v in cat.most_common():
        print(f"    {v:4d}  {k}")
    print(f"    of the never-expected calls: {cross} cross-domain, {same} same-domain")
    print(f"    examples containing >=1 exact duplicate: {dup_examples}/{len(rows)}")
    if gaps:
        gaps.sort()
        print(f"    duplicate re-issue gap: min {gaps[0]}s  p50 {gaps[len(gaps) // 2]}s  "
              f"max {gaps[-1]}s")
    if pos_matched and pos_wrong:
        print(f"    position in audio (1.0 = end of real user speech): "
              f"matched p50 {st.median(pos_matched):.2f}, wrong p50 {st.median(pos_wrong):.2f}")
    # Zero rejects is what makes the strict-vs-lenient scoring question empty (see the doc).
    print(f"    rejected calls: {sum(len(r['rejected']) for r in rows)}   "
          f"unparseable arguments: {sum(r['dropped'] for r in rows)}")
    if show:
        print()
        for eid, t, fn, a in sorted(spurious):
            print(f"    {eid:<16} t={t:6.1f}s  {fn}({json.dumps(a)})")


def oracle(rows: list[dict]) -> None:
    """Question 3: upper bound on any precision-side fix. Recall is never touched."""
    variants = {"as measured": [], "- exact duplicates": [], "- never-expected fns": [],
                "- both (oracle)": []}
    by_diff = collections.defaultdict(lambda: ([], []))
    for r in rows:
        e = collections.Counter(r["expected"])
        calls = [(c["function"], json.dumps(c.get("args"), sort_keys=True)) for c in r["calls"]]

        def dedup(items):
            seen, out = set(), []
            for fn, k in items:
                if (fn, k) in seen:
                    continue
                seen.add((fn, k))
                out.append(fn)
            return out

        base = [fn for fn, _ in calls]
        no_wrong = [fn for fn, _ in calls if e[fn] > 0]
        variants["as measured"].append(multiset_f1(base, r["expected"]))
        variants["- exact duplicates"].append(multiset_f1(dedup(calls), r["expected"]))
        variants["- never-expected fns"].append(multiset_f1(no_wrong, r["expected"]))
        both = dedup([(fn, k) for fn, k in calls if e[fn] > 0])
        variants["- both (oracle)"].append(multiset_f1(both, r["expected"]))
        by_diff[r["difficulty"]][0].append(variants["as measured"][-1])
        by_diff[r["difficulty"]][1].append(variants["- both (oracle)"][-1])

    mean = lambda v: sum(v) / len(v) * 100  # noqa: E731
    baseline = mean(variants["as measured"])
    for label, vals in variants.items():
        delta = f"   (+{mean(vals) - baseline:.1f})" if label != "as measured" else ""
        print(f"    {label:<24}{mean(vals):6.1f} %{delta}")
    print(f"    {'card':<24}{CARD_TOOL_SELECTION * 100:6.1f} %")
    print("    by difficulty (as measured -> oracle):")
    for k in ("easy", "medium", "hard"):
        if by_diff[k][0]:
            print(f"       {k:<7} n={len(by_diff[k][0]):2d}  {mean(by_diff[k][0]):5.1f} % -> "
                  f"{mean(by_diff[k][1]):5.1f} %")


def conventions(rows: list[dict]) -> None:
    """Question 4: is the card scoring the same behaviour a different way?"""
    def agg(fn):
        return sum(fn(r) for r in rows) / len(rows) * 100

    def counts(r):
        return (collections.Counter(r["expected"]),
                collections.Counter(c["function"] for c in r["calls"]),
                [c["function"] for c in r["calls"]])

    def multiset(r):
        _, _, a = counts(r)
        return multiset_f1(a, r["expected"])

    def set_f1(r):
        e, _, a = counts(r)
        es, as_ = set(e), set(a)
        m = len(es & as_)
        return f1(m / len(as_) if as_ else 0.0, m / len(es) if es else 0.0)

    def recall_ms(r):
        e, a, _ = counts(r)
        return sum((e & a).values()) / len(r["expected"])

    def prec_ms(r):
        e, a, calls = counts(r)
        return sum((e & a).values()) / len(calls) if calls else 0.0

    def recall_set(r):
        e, _, a = counts(r)
        return len(set(e) & set(a)) / len(set(e))

    def coverage(r):
        e, _, a = counts(r)
        return 1.0 if set(e) <= set(a) else 0.0

    def exact(r):
        e, a, _ = counts(r)
        return 1.0 if e == a else 0.0

    for label, fn in [("multiset F1 (official)", multiset), ("set F1 (dedup)", set_f1),
                      ("recall multiset", recall_ms), ("precision multiset", prec_ms),
                      ("recall set", recall_set), ("set coverage (all expected hit)", coverage),
                      ("exact multiset match", exact)]:
        print(f"    {label:<34}{agg(fn):6.1f} %")

    E, A, M = collections.Counter(), collections.Counter(), 0
    for r in rows:
        e, a, _ = counts(r)
        E += e
        A += a
        M += sum((e & a).values())
    print(f"    {'micro multiset F1 (pooled)':<34}"
          f"{f1(M / sum(A.values()), M / sum(E.values())) * 100:6.1f} %")

    # 100 recordings over 79 scenarios: 21 scenarios have two speaker renditions.
    per = collections.defaultdict(list)
    for r in rows:
        per[r["eid"]].append(r)
    first = sum(multiset(v[0]) for v in per.values()) / len(per) * 100
    best = sum(max(multiset(x) for x in v) for v in per.values()) / len(per) * 100
    print(f"    per-scenario (n={len(per)}): first speaker {first:.1f} %, "
          f"best speaker {best:.1f} %")


def main() -> int:
    args = parse_args()
    for provider in (args.provider or ["nemo_rt", "nemo_rt_jinja"]):
        rows, _ = load(args.v3, args.data_dir, provider)
        if not rows:
            print(f"no result_{provider}.json under {args.v3 / args.data_dir}")
            continue
        print("=" * 88)
        print(f"{provider}   n={len(rows)}   card Tool Selection = "
              f"{CARD_TOOL_SELECTION * 100:.1f} %")
        print("=" * 88)
        print("\n1. recall or precision, by number of expected calls")
        by_expected_count(rows)
        print("\n2. what the calls are")
        classify(rows, args.show_spurious)
        print("\n3. oracle: delete spurious + duplicate calls, keep recall")
        oracle(rows)
        print("\n4. metric conventions -- is 82.5 % reachable by rescoring?")
        conventions(rows)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
