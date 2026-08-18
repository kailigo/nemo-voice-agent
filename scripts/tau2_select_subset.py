#!/usr/bin/env python
"""Pick a diagnostic task subset for a tau2 voice run, maximising scenario coverage.

WHY NOT `--num-tasks N`
    It is `tasks[:N]` (`tau2/runner/helpers.py:92`), and the task lists are *grouped* --
    consecutive tasks are near-duplicate variants of one scenario. So a prefix is badly
    biased, and correlated variants make the effective sample smaller than N. Measured on
    a coarse 60-char scenario key, `--num-tasks 30` covers 20 of retail's 87 scenarios and
    1 of telecom's 3.

    Telecom is the trap. Its users nearly all say the same thing -- only 3 distinct
    `reason_for_call` values across 114 tasks -- because what varies there is the
    environment: the device/network fault state and the repair sequence that fixes it. So
    striding on the user scenario is the WRONG axis for telecom, and this script picks the
    diagnostic key per domain (see DIAGNOSTIC_KEY) accordingly.

    The counts this script reports are higher than those, because it keys on a longer
    prefix (200 chars of scenario, 400 of evaluation) and so splits variants the coarse key
    merges. Either key shows the same bias; the finer one just makes "distinct" stricter.

WHAT IT DOES
    Places N evenly spaced anchors across the task list, then at each anchor takes the
    nearest task whose diagnostic key has not been used yet. That gets both properties;
    plain greedy-in-order gets only one of them. Walking the list taking each first-unseen
    key still clusters at the front whenever there are many distinct keys -- on telecom it
    picked 8 distinct fault states that were all `mobile_data_issue`, reproducing exactly
    the bias this script exists to avoid.

    Deterministic -- no RNG. Same arguments always give the same ids, which matters because
    a subset run must stay a valid subset of the eventual full run for `--save-to` resume
    to line up.

NOTE ON QUOTING
    Telecom task ids are not identifiers -- they look like
    `[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]`, with `|` and `[]`
    that a shell will happily interpret. `--format args` shell-quotes each id for this
    reason; do not hand-edit the quotes off.

Must run with tau2 importable (the `voicechat` env; see scripts/tau2_smoke_nemo.sh).

Usage:
    python scripts/tau2_select_subset.py                    # 8 tasks x 3 test domains
    python scripts/tau2_select_subset.py --per-domain 12
    python scripts/tau2_select_subset.py --domains retail --per-domain 8 --format args
"""

from __future__ import annotations

import argparse
import shlex
import sys

# Which field actually discriminates tasks, per domain. Defaults to the user scenario;
# telecom needs the environment side instead (see the module docstring).
DIAGNOSTIC_KEY = {
    "retail": "scenario",
    "airline": "scenario",
    "telecom": "evaluation",
}


def _key(task, kind: str) -> str:
    """Return the string this task is deduplicated on."""
    if kind == "evaluation":
        return str(task.evaluation_criteria)[:400]
    scenario = task.user_scenario
    instructions = scenario.instructions if scenario else None
    if instructions is None:
        return ""
    return (instructions.reason_for_call or "")[:200]


def select(tasks, n: int, kind: str) -> list:
    """Evenly spaced anchors, each resolved to the nearest task with an unused key.

    Returns tasks in original list order. Falls back to "nearest untaken" once the distinct
    keys run out, so it always returns min(n, len(tasks)) tasks rather than silently short.
    """
    total = len(tasks)
    n = min(n, total)
    keys = [_key(t, kind) for t in tasks]
    taken: set[int] = set()
    seen: set[str] = set()

    def nearest(anchor: int, require_new_key: bool) -> int | None:
        for d in range(total):
            for cand in (anchor + d, anchor - d):
                if 0 <= cand < total and cand not in taken:
                    if not require_new_key or keys[cand] not in seen:
                        return cand
        return None

    for j in range(n):
        anchor = min(total - 1, int((j + 0.5) * total / n))
        pick = nearest(anchor, require_new_key=True)
        if pick is None:  # distinct keys exhausted; spread over what is left
            pick = nearest(anchor, require_new_key=False)
        if pick is None:
            break
        taken.add(pick)
        seen.add(keys[pick])

    return [tasks[i] for i in sorted(taken)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domains", nargs="+", default=["retail", "airline", "telecom"])
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument(
        "--format",
        choices=["report", "args"],
        default="report",
        help="'args' prints just the --task-ids value, for shell interpolation.",
    )
    args = ap.parse_args()

    from tau2.registry import registry

    for domain in args.domains:
        tasks = registry.get_tasks_loader(domain)()
        kind = DIAGNOSTIC_KEY.get(domain, "scenario")
        picked = select(tasks, args.per_domain, kind)
        ids = [t.id for t in picked]

        quoted = " ".join(shlex.quote(i) for i in ids)

        if args.format == "args":
            print(f"{domain} {quoted}")
            continue

        total_keys = len({_key(t, kind) for t in tasks})
        got_keys = len({_key(t, kind) for t in picked})
        prefix_keys = len({_key(t, kind) for t in tasks[: args.per_domain]})
        # Spread: where the picks land in the list, as a sanity check against clustering.
        index = {id(t): i for i, t in enumerate(tasks)}
        positions = [index[id(t)] for t in picked]
        print(f"{domain}: {len(tasks)} tasks, {total_keys} distinct {kind}s")
        print(
            f"  selected {len(ids)}: covers {got_keys} {kind}s "
            f"(a --num-tasks {args.per_domain} prefix would cover {prefix_keys})"
        )
        print(f"  list positions: {positions}")
        print(f"  --task-ids {quoted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
