#!/usr/bin/env python
"""Merge the per-episode result directories that `tau2_stage2_subset.sh` produces.

WHY THIS EXISTS
    Stage 2 runs one process per episode, each with its own `--save-to`, because a second
    episode in one process leaks an 11B model (see the script's header) and a shared
    `--save-to` routes the second process into `try_resume`, which raises
    `ValueError: Tasks were removed from the task set` as soon as the new `--task-ids` does
    not contain the previous run's task (checkpoint.py:145). So the fan-out is 24
    directories, and this puts them back together.

WHAT IT PRODUCES
    `data/simulations/<run>_merged/` in tau2's own dir format: `results.json` (metadata +
    `simulation_index`) plus `simulations/<id>.json` per episode, i.e. exactly what a
    single big run would have written. `tau2 view` and the metric code read it unchanged.

    Merging across domains is *why* this is a script rather than a `cat`: `Results.info`
    carries one `environment_info` (domain, policy, tool schemas), and the merged object
    keeps the first directory's. That is a lie about a 3-domain run, so the merged
    `results.json` is for browsing and for the summary table; per-domain reward must be
    computed from `simulations[*].task_id` and the per-directory originals, which this
    script leaves in place. It prints the per-domain breakdown for that reason.

Must run with tau2 importable (the `voicechat` env; see scripts/tau2_smoke_nemo.sh).

Usage:
    python scripts/tau2_merge_results.py --run stage2_subset
    python scripts/tau2_merge_results.py --run stage2_subset --summary-only
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--run",
        default="stage2_subset",
        help="The RUN_NAME the fan-out used; its per-episode dirs live under "
        "data/simulations/<run>/.",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the table but do not write the merged directory.",
    )
    args = ap.parse_args()

    from tau2.data_model.simulation import Results
    from tau2.utils.utils import DATA_DIR

    parent = DATA_DIR / "simulations" / args.run
    if not parent.exists():
        print(f"no such run directory: {parent}", file=sys.stderr)
        return 1

    # An episode directory is one that has its own results.json. Skip <run>_merged if a
    # previous merge landed inside the tree.
    episode_dirs = sorted(
        d for d in parent.iterdir() if d.is_dir() and (d / "results.json").exists()
    )
    if not episode_dirs:
        print(f"no episode directories with a results.json under {parent}", file=sys.stderr)
        return 1

    merged = None
    sims = []
    domain_of_sim: dict[str, str] = {}
    tasks_by_id: dict[str, object] = {}
    unreadable = []

    for d in episode_dirs:
        try:
            r = Results.load(d)
        except Exception as e:  # a killed episode can leave a half-written index
            unreadable.append((d.name, f"{type(e).__name__}: {e}"))
            continue
        if merged is None:
            merged = r
        # The fan-out names each directory `<domain>__<slugged task id>`, which is the only
        # surviving record of the domain: `Results.info.environment_info` is per-run and the
        # merged object can hold just one, and telecom is the only domain whose task ids are
        # self-identifying. Read it off the directory rather than guessing from the id.
        domain = d.name.split("__", 1)[0]
        for s in r.simulations:
            domain_of_sim[s.id] = domain
        sims.extend(r.simulations)
        for t in r.tasks:
            tasks_by_id[t.id] = t

    if merged is None:
        print("every episode directory failed to load", file=sys.stderr)
        for name, err in unreadable:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1

    merged.simulations = sims
    merged.tasks = list(tasks_by_id.values())

    # --- summary ------------------------------------------------------------
    by_domain: dict[str, list] = defaultdict(list)
    for s in sims:
        by_domain[domain_of_sim.get(s.id, "unknown")].append(s)

    print(f"{len(episode_dirs)} episode dirs, {len(sims)} simulations loaded")
    if unreadable:
        print(f"{len(unreadable)} unreadable:")
        for name, err in unreadable:
            print(f"  {name}: {err}")

    print()
    print(f"{'domain':10} {'n':>3} {'reward':>7} {'ticks':>7} {'sim_s':>8}  terminations")
    for domain in sorted(by_domain):
        group = by_domain[domain]
        rewards = [_reward(s) for s in group]
        rewards = [r for r in rewards if r is not None]
        ticks = [len(s.ticks or []) for s in group]
        secs = [s.duration or 0.0 for s in group]
        terms: dict[str, int] = defaultdict(int)
        for s in group:
            terms[str(getattr(s.termination_reason, "value", s.termination_reason))] += 1
        print(
            f"{domain:10} {len(group):>3} "
            f"{(sum(rewards) / len(rewards) if rewards else float('nan')):>7.3f} "
            f"{(sum(ticks) / len(ticks) if ticks else 0):>7.0f} "
            f"{(sum(secs) / len(secs) if secs else 0):>8.0f}  "
            + ", ".join(f"{k}={v}" for k, v in sorted(terms.items()))
        )

    # Tool-call failures are the stage-2 diagnostic: stage 1 found 10 calls, 10 failures,
    # 8 of them to tool names that do not exist. Count that across the whole subset.
    names: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # name -> [calls, errors]
    for s in sims:
        for tick in s.ticks or []:
            for call in tick.agent_tool_calls or []:
                names[call.name][0] += 1
            for res in tick.agent_tool_results or []:
                if getattr(res, "error", False):
                    names[_result_name(res, tick)][1] += 1
    if names:
        print()
        print(f"{'tool':38} {'calls':>6} {'errors':>7}")
        for name, (n_calls, n_err) in sorted(
            names.items(), key=lambda kv: -kv[1][0]
        )[:20]:
            print(f"{name:38} {n_calls:>6} {n_err:>7}")

    if args.summary_only:
        return 0

    out = DATA_DIR / "simulations" / f"{args.run}_merged"
    merged.save(out, format="dir")
    print()
    print(f"wrote {out} ({len(sims)} simulations)")
    return 0


def _reward(sim) -> float | None:
    info = getattr(sim, "reward_info", None)
    return getattr(info, "reward", None) if info is not None else None


def _result_name(res, tick) -> str:
    """A tool result carries an id, not a name; match it back to the call in this tick."""
    for call in tick.agent_tool_calls or []:
        if getattr(call, "id", None) == getattr(res, "id", None):
            return call.name
    return "<unmatched>"


if __name__ == "__main__":
    sys.exit(main())
