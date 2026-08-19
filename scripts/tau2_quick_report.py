#!/usr/bin/env python3
"""Report rewards and tool-call outcomes for a fan-out run, using only the stdlib.

WHY NOT tau2_merge_results.py
    That one imports tau2, which costs ~90 s of torch/NeMo import before it prints
    anything. A watchdog that polls every two minutes cannot pay that, so this reads the
    persisted `simulations/<id>.json` directly. The tradeoff is no `Results` validation
    and no merged directory -- use the other script for the artefact you keep, this one
    for the number you want now.

Usage:
    python3 scripts/tau2_quick_report.py --run stage2_protocol [--json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DATA = Path("/fsx/home/kai.li/code/tau-voice-2/data/simulations")


def episodes(run_dir: Path):
    """Yield (domain, task_slug, sim_dict) for every persisted simulation."""
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        # The domain is only recoverable from the directory name the fan-out chose;
        # `environment_info` is per-run and a merged view can hold only one.
        domain, _, slug = d.name.partition("__")
        for f in sorted((d / "simulations").glob("*.json")):
            try:
                yield domain, slug, json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                # A simulation killed mid-write. Skipped rather than fatal: the point of
                # this script is to report on the episodes that did land.
                continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="stage2_protocol")
    ap.add_argument("--json", action="store_true", help="machine-readable, for the watchdog")
    args = ap.parse_args()

    run_dir = DATA / args.run
    if not run_dir.exists():
        print(f"no such run: {run_dir}")
        return 1

    rows = []
    tools: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    bad_names: dict[str, int] = defaultdict(int)
    for domain, slug, sim in episodes(run_dir):
        ticks = sim.get("ticks") or []
        calls = errors = 0
        for t in ticks:
            by_id = {c.get("id"): c.get("name") for c in (t.get("agent_tool_calls") or [])}
            for c in t.get("agent_tool_calls") or []:
                tools[c.get("name", "?")][0] += 1
                calls += 1
            for r in t.get("agent_tool_results") or []:
                if not r.get("error"):
                    continue
                name = by_id.get(r.get("id"), "<unmatched>")
                tools[name][1] += 1
                errors += 1
                # "Tool 'x' not found" is the arm-A signature the protocol fix targets, so
                # it is counted separately from an argument-level error like "Order not
                # found" -- they call for opposite responses.
                if "not found" in str(r.get("content", "")) and "Tool '" in str(
                    r.get("content", "")
                ):
                    bad_names[name] += 1
        rows.append(
            {
                "domain": domain,
                "task": slug,
                "reward": (sim.get("reward_info") or {}).get("reward"),
                "ticks": len(ticks),
                "termination": sim.get("termination_reason"),
                "calls": calls,
                "errors": errors,
            }
        )

    if args.json:
        print(json.dumps({"episodes": rows, "invented_tool_calls": sum(bad_names.values())}))
        return 0

    print(f"{len(rows)} episode(s) persisted under {run_dir.name}\n")
    print(f"{'domain':9} {'task':10} {'reward':>7} {'ticks':>6} {'calls':>6} {'err':>5}  termination")
    for r in sorted(rows, key=lambda r: (r["domain"], r["task"])):
        rw = "  n/a " if r["reward"] is None else f"{r['reward']:>6.3f}"
        print(
            f"{r['domain']:9} {r['task']:10} {rw:>7} {r['ticks']:>6} "
            f"{r['calls']:>6} {r['errors']:>5}  {r['termination']}"
        )

    if tools:
        print(f"\n{'tool':34} {'calls':>6} {'errors':>7}  note")
        for name, (n, e) in sorted(tools.items(), key=lambda kv: -kv[1][0]):
            note = "TOOL DOES NOT EXIST" if bad_names.get(name) else ""
            print(f"{name:34} {n:>6} {e:>7}  {note}")
        total = sum(v[0] for v in tools.values())
        invented = sum(bad_names.values())
        print(
            f"\ninvented-tool-name calls: {invented} of {total}"
            + (f" ({invented / total:.0%})" if total else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
