#!/usr/bin/env python3
"""Signed per-turn turn-taking latency for FDB-v3 result files.

Why this exists. The benchmark's own metric (`analyze_tool_latency.py`) computes one
latency per *example*: the agent's first speech minus the end of the user's **first** turn.
Any example where that is negative is labelled an interruption and dropped. On our
`nemo_rt` run that discarded 35 of 98 examples -- and those 35 are exactly the ones where
the model responded fastest, because it started talking before the user had finished. The
survivors' mean is therefore biased slow by construction: it conditions on the outcome.

So this script keeps the sign instead of discarding it, and measures every user turn rather
than only the first:

    signed latency of turn i = (first agent onset after turn i began) - (end of turn i)

Negative means the agent began speaking while the user was still in that turn -- the
magnitude is how far into the turn it barged in. Positive is a conventional response delay.
Nothing is dropped, so the mean is over all paired turns and the barge-in rate is a
reported quantity rather than a filter.

This is a *supplementary diagnostic, not a substitute*: it is not comparable to any
published FDB-v3 column, because no published column is computed this way. Report it
alongside the benchmark's number, never instead of it.

The two anchors, reconciled. Each result file carries both:

  * `input_asr_chunks`  -- Parakeet word timestamps over input.wav. The acoustic truth
    about when the user actually stopped. Requires `fdb_v3_asr_input.py` to have run.
  * `user_speech_end_s` -- the *server's* own ASR end-of-speech markers, i.e. when the
    model decided the user had stopped. `response_latency_s` is measured from these.

`delta = server marker - acoustic turn end` is printed per provider because it is the whole
explanation for why the two latency figures differ by ~6x. A positive delta means the
server waited past the real end (so its latency reads flatteringly small); a negative delta
means it called the turn over early and the agent talked over the user.

Usage:

    python scripts/fdb_v3_signed_latency.py --provider nemo_rt --provider nemo_rt_jinja
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import rather than re-declare: this must not drift from the turn rule the benchmark's own
# latency metric uses (fdb_v3_asr_input.py took it from run_tool_benchmark.py:358).
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fdb_v3_asr_input import TURN_GAP_S  # noqa: E402

DATA_DIR = Path("/fsx/home/kai.li/code/Full-Duplex-Bench/v3/fdb_v3_data_released")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--provider",
        action="append",
        required=True,
        help="Result-file provider name. Repeatable, to compare arms side by side.",
    )
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/fsx/home/kai.li/code/nemo-voice-agent/logs/fdb_v3"),
    )
    p.add_argument(
        "--examples",
        type=int,
        default=6,
        help="How many worst barge-ins to print per provider.",
    )
    return p.parse_args()


def user_turns(chunks: List[dict]) -> List[Tuple[float, float]]:
    """Segment Parakeet word chunks into (start, end) turns on a >TURN_GAP_S silence.

    Same rule as `fdb_v3_asr_input.speech_end`, but returning *every* turn instead of only
    the first -- that function stops at the first boundary because the benchmark's metric
    only ever looks at turn 1.
    """
    if not chunks:
        return []
    turns = []
    start = chunks[0]["timestamp"][0]
    for cur, nxt in zip(chunks, chunks[1:]):
        if nxt["timestamp"][0] - cur["timestamp"][1] > TURN_GAP_S:
            turns.append((start, cur["timestamp"][1]))
            start = nxt["timestamp"][0]
    turns.append((start, chunks[-1]["timestamp"][1]))
    return turns


def pair_onsets(
    turns: List[Tuple[float, float]],
    onsets: List[float],
    markers: List[float],
) -> Tuple[List[Dict], List[float]]:
    """Attach one agent *response* to each user turn, via the server's own turn decisions.

    Pairing on "first onset after the turn began" is wrong, and wrong in a way that
    manufactures huge fake barge-ins. The agent opens nearly every episode with an
    unprompted greeting at ~2 s ("Hi there! How can I help you today?"), so against a user
    who then monologues for 26 s that rule reports a 26-second barge-in. Measured on
    `housing_24_69a9cf80f4d7668d5c815038`: greeting at 2.16 s, user turn 1.84-28.00 s, and
    the actual response at 38.16 s.

    So pair through the **server's end-of-speech markers** (`user_speech_end_s`) instead.
    Those are the model's own decisions about when a turn ended, and every paired response in
    the harness follows one -- which is why `response_latency_s` is never negative. For each
    marker:

      * the response is the first agent onset after it (reproducing `response_latency_s`);
      * the marker belongs to the latest user turn that had already begun, i.e. the turn in
        progress or the one just finished;
      * the signed latency is measured against that turn's **acoustic** end, so a marker
        fired mid-turn yields a negative value of exactly the right depth.

    Onsets preceding every marker answer nothing and are returned separately as unprompted
    openings, rather than silently becoming turn responses.
    """
    rows: List[Dict] = []
    claimed = set()
    for m in markers:
        onset = next((o for o in onsets if o > m), None)
        idx = None
        for i, (start, _end) in enumerate(turns):
            if start <= m:
                idx = i
        if idx is None:
            # A marker before the user has said anything: the server called end-of-speech on
            # silence. Not attributable to a turn, but worth surfacing rather than dropping.
            rows.append(
                {"turn": None, "turn_start": None, "turn_end": None, "marker": m, "onset": onset}
            )
            continue
        claimed.add(idx)
        start, end = turns[idx]
        rows.append(
            {"turn": idx, "turn_start": start, "turn_end": end, "marker": m, "onset": onset}
        )

    for i, (start, end) in enumerate(turns):
        if i not in claimed:
            rows.append(
                {"turn": i, "turn_start": start, "turn_end": end, "marker": None, "onset": None}
            )

    first_marker = markers[0] if markers else float("inf")
    unprompted = [o for o in onsets if o < first_marker]
    return rows, unprompted


def _stats(v: List[float]) -> Dict[str, float]:
    if not v:
        return {}
    s = sorted(v)
    return {
        "n": len(s),
        "mean": round(st.mean(s), 3),
        "sd": round(st.pstdev(s), 3) if len(s) > 1 else 0.0,
        "min": round(s[0], 3),
        "p50": round(st.median(s), 3),
        "max": round(s[-1], 3),
    }


def analyse(data_dir: Path, provider: str) -> Dict:
    files = sorted(data_dir.glob(f"*/result_{provider}.json"))
    rows: List[Dict] = []
    missing_asr = 0
    n_unprompted = 0
    n_examples_unprompted = 0
    n_turns = 0
    for f in files:
        d = json.loads(f.read_text())
        chunks = d.get("input_asr_chunks") or []
        if not chunks:
            missing_asr += 1
            continue
        onsets = d.get("agent_speech_onsets_s") or []
        markers = d.get("user_speech_end_s") or []
        turns = user_turns(chunks)
        n_turns += len(turns)
        paired_rows, unprompted = pair_onsets(turns, onsets, markers)
        n_unprompted += len(unprompted)
        n_examples_unprompted += 1 if unprompted else 0
        for r in paired_rows:
            end, marker, onset = r["turn_end"], r["marker"], r["onset"]
            rows.append(
                {
                    "example": f.parent.name,
                    **r,
                    "turn_dur": round(end - r["turn_start"], 2) if end is not None else None,
                    "signed_latency": (
                        round(onset - end, 3) if onset is not None and end is not None else None
                    ),
                    # How far the server's end-of-speech decision sat from the acoustic end of
                    # the turn. Negative = it called the turn over while the user was talking.
                    "marker_delta": (
                        round(marker - end, 3) if marker is not None and end is not None else None
                    ),
                }
            )

    answered = [r for r in rows if r["signed_latency"] is not None]
    lat = [r["signed_latency"] for r in answered]
    neg = [x for x in lat if x < 0]
    pos = [x for x in lat if x >= 0]
    deltas = [r["marker_delta"] for r in rows if r["marker_delta"] is not None]

    # A row is one (server end-of-speech marker -> response) pair, and the server can fire
    # several inside a single acoustic user turn -- it segments more finely than the >2 s gap
    # rule does. So responses must be counted separately from turns, and the turns carrying
    # more than one marker are worth naming: over-segmentation is the mechanism that produces
    # the barge-ins below.
    per_turn: Dict[Tuple[str, int], int] = {}
    for r in rows:
        if r["turn"] is not None and r["marker"] is not None:
            per_turn[(r["example"], r["turn"])] = per_turn.get((r["example"], r["turn"]), 0) + 1
    turns_answered = len(per_turn)
    turns_multi = sum(1 for v in per_turn.values() if v > 1)

    return {
        "provider": provider,
        "files": len(files),
        "files_missing_asr": missing_asr,
        "unprompted_openings": n_unprompted,
        "examples_with_unprompted_opening": n_examples_unprompted,
        "user_turns": n_turns,
        "turns_answered": turns_answered,
        "turns_unanswered": n_turns - turns_answered,
        "turns_multi_marker": turns_multi,
        "responses": len(answered),
        "turns_paired": len(answered),
        "signed_all": _stats(lat),
        "barge_ins": _stats(neg),
        "clean_responses": _stats(pos),
        "barge_in_rate_turns": round(len(neg) / len(answered), 4) if answered else None,
        "barge_in_rate_examples": round(
            len({r["example"] for r in answered if r["signed_latency"] < 0})
            / max(len({r["example"] for r in answered}), 1),
            4,
        ),
        "server_marker_delta": _stats(deltas),
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [analyse(args.data_dir, p) for p in args.provider]

    for r in results:
        if r["files_missing_asr"]:
            print(
                f"WARNING {r['provider']}: {r['files_missing_asr']}/{r['files']} files have no "
                f"input_asr_chunks -- run scripts/fdb_v3_asr_input.py --provider {r['provider']} first."
            )

    w = 26
    print()
    print("=" * (w + 22 * len(results)))
    print("SIGNED PER-TURN TURN-TAKING LATENCY")
    print("=" * (w + 22 * len(results)))
    print(f"turn rule: >{TURN_GAP_S}s silence splits user turns (the benchmark's own rule)")
    print("negative = agent began speaking before the user's turn ended (barge-in)")
    print()

    def row(label: str, fn):
        print(f"{label:<{w}}" + "".join(f"{fn(r):>22}" for r in results))

    row("provider", lambda r: r["provider"])
    row("result files", lambda r: r["files"])
    row("acoustic user turns", lambda r: r["user_turns"])
    row("turns answered", lambda r: r["turns_answered"])
    row("turns unanswered", lambda r: r["turns_unanswered"])
    row("agent responses", lambda r: r["responses"])
    row("turns cut into >1 response", lambda r: r["turns_multi_marker"])
    row("unprompted openings", lambda r: r["unprompted_openings"])
    row("  ...in n examples", lambda r: r["examples_with_unprompted_opening"])
    print()
    row("signed mean (s)", lambda r: f"{r['signed_all'].get('mean', float('nan')):+.2f}")
    row("signed median (s)", lambda r: f"{r['signed_all'].get('p50', float('nan')):+.2f}")
    row("signed min / max", lambda r: f"{r['signed_all'].get('min', 0):+.1f} / {r['signed_all'].get('max', 0):+.1f}")
    print()
    row("BARGE-IN rate (turns)", lambda r: f"{100 * (r['barge_in_rate_turns'] or 0):.1f} %")
    row("barge-in rate (examples)", lambda r: f"{100 * (r['barge_in_rate_examples'] or 0):.1f} %")
    row("barge-in count", lambda r: r["barge_ins"].get("n", 0))
    row("barge-in median depth", lambda r: f"{r['barge_ins'].get('p50', float('nan')):+.2f}s")
    row("barge-in worst", lambda r: f"{r['barge_ins'].get('min', float('nan')):+.2f}s")
    print()
    row("clean-response count", lambda r: r["clean_responses"].get("n", 0))
    row("clean-response median", lambda r: f"{r['clean_responses'].get('p50', float('nan')):+.2f}s")
    row("clean-response mean", lambda r: f"{r['clean_responses'].get('mean', float('nan')):+.2f}s")
    print()
    print("server end-of-speech marker minus acoustic turn end (explains the ~6x anchor gap):")
    row("  delta median", lambda r: f"{r['server_marker_delta'].get('p50', float('nan')):+.2f}s")
    row("  delta mean", lambda r: f"{r['server_marker_delta'].get('mean', float('nan')):+.2f}s")
    row("  delta n", lambda r: r["server_marker_delta"].get("n", 0))
    print()

    for r in results:
        worst = sorted(
            (x for x in r["rows"] if x["signed_latency"] is not None and x["signed_latency"] < 0),
            key=lambda x: x["signed_latency"],
        )[: args.examples]
        if not worst:
            continue
        print(f"deepest barge-ins, {r['provider']}:")
        print(f"  {'example':<44}{'turn':>5}{'span':>16}{'onset':>8}{'signed':>9}")
        for x in worst:
            span = f"{x['turn_start']:.1f}-{x['turn_end']:.1f}"
            print(
                f"  {x['example']:<44}{x['turn']:>5}{span:>16}"
                f"{x['onset']:>8.1f}{x['signed_latency']:>+9.2f}"
            )
        print()

    for r in results:
        dest = args.out_dir / f"{r['provider']}_signed_latency.json"
        dest.write_text(json.dumps(r, indent=2))
        print(f"wrote {dest}")

    print(
        "\nNOTE: not comparable to any published FDB-v3 column -- no published column is "
        "computed this way. Report it beside the benchmark's own number, not instead of it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
