#!/usr/bin/env python
"""Merge FDB-v3 result arms into one provider, so a split run scores as a single arm.

The all-100 research arm was produced in two pieces: the 30 `hard` examples ran first as
`nemo_research_hard` (the path-vs-path test in `FDB_V3_REPRODUCTION.md` §4), and the 70
easy+medium examples ran afterwards as `nemo_research`. Scoring either alone is not
comparable to the card, which quotes all 100.

They cannot simply be concatenated, because every evaluator in `Full-Duplex-Bench/v3` finds
its inputs by globbing `*/result_<provider>.json` -- the arm *is* the filename. So merging
means copying files under the target name and rewriting the `provider` field inside them to
match, or a later reader sees a file called `result_nemo_research.json` whose contents claim
to be `nemo_research_hard`.

Two safety rules, both learned the hard way:

  * **Never overwrite an existing result.** Results are one file per example dir per provider,
    and an arm needed for comparison is destroyed silently by a name collision. This script
    refuses to clobber unless `--force` is passed, and reports what it skipped.
  * **Coverage is the verdict, not the exit code.** The evaluators drop absent scenarios from
    the denominator, so a partial merge scores *higher* while measuring less. The final count
    is printed and compared against `--expect`.

    python scripts/fdb_v3_merge_arm.py --from nemo_research_hard --into nemo_research --expect 100
    python scripts/fdb_v3_merge_arm.py --from nemo_research_hard --into nemo_research --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

V3 = Path("/fsx/home/kai.li/code/Full-Duplex-Bench/v3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--from", dest="src", required=True, help="provider to copy from")
    p.add_argument("--into", dest="dst", required=True, help="provider to copy into")
    p.add_argument("--v3", type=Path, default=V3)
    p.add_argument("--data-dir", default="fdb_v3_data_released")
    p.add_argument("--expect", type=int, default=None,
                   help="expected final example count; mismatch is reported, not fatal")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing target results (destroys an arm -- be sure)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if a.src == a.dst:
        print("--from and --into are the same provider; nothing to do")
        return 1

    root = a.v3 / a.data_dir
    copied = skipped = 0
    for path in sorted(glob.glob(str(root / f"*/result_{a.src}.json"))):
        d = Path(os.path.dirname(path))
        target = d / f"result_{a.dst}.json"
        if target.exists() and not a.force:
            skipped += 1
            print(f"  skip (exists)  {d.name}")
            continue
        payload = json.loads(Path(path).read_text())
        # The filename and the field must agree, or the arm lies about its own provenance.
        payload["provider"] = a.dst
        payload["merged_from"] = a.src
        if not a.dry_run:
            target.write_text(json.dumps(payload, indent=2))
        copied += 1
        print(f"  {'would copy' if a.dry_run else 'copied'}     {d.name}")

    total = len(glob.glob(str(root / f"*/result_{a.dst}.json")))
    if a.dry_run:
        total += copied
    print(f"\n{'would copy' if a.dry_run else 'copied'} {copied}, skipped {skipped}")
    print(f"arm '{a.dst}' now covers {total} examples"
          f"{'' if a.dry_run else ' on disk'}")
    if a.expect is not None and total != a.expect:
        # Not an error exit: a partial arm is still scoreable, it just is not comparable.
        print(f"WARNING: expected {a.expect}. The evaluators drop absent scenarios from the "
              f"denominator, so this arm will score as if the missing {a.expect - total} "
              f"never existed. Do not quote it against the card.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
