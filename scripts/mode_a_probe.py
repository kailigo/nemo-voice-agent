#!/usr/bin/env python3
"""Localise mode A: is a spelled-out ID lost in the audio, in our encoder, or in the copy?

Mode A (NEMO_FAILURE_MODES.md §3) is the highest-yield failure: the user spells an identifier
and the model writes down something else -- `SI5UKW` -> `SIN-555`, `anya_garcia_5901` ->
`JFK59001`. Whether that is a perception error or a copy error decides what SFT data looks
like, and the two answers point at opposite recipes. It is listed as open question #1 in §7.

READOUT 0, which is what this script implements. Run an INDEPENDENT ASR over the exact audio
the model heard and ask only: **is the ID recoverable from this signal at all?**

  * recoverable  -> the audio is fine, the failure is inside our model, and readouts 1-3
                    (below) are worth the GPU time.
  * unrecoverable -> the ID is not in the 8 kHz telephony signal in the first place. Mode A is
                    substantially a STIMULUS problem, the SFT fix is bandwidth/telephony
                    augmentation rather than more spelled-ID text, and the model is partly
                    exonerated. This is the branch that most changes the story, which is why
                    it is the cheapest one and runs first.

The stimulus is not synthesised. tau-voice already persisted `both.wav` per episode plus
Audacity-style label tracks, so we probe the SAME WAVEFORM that produced `SIN-555` -- no
re-synthesis, no ElevenLabs quota, no stimulus drift. `user_labels.txt` gives the spelling turn
to the tick, so the clip is cut to exactly the utterance and nothing else.

WHAT THIS DOES NOT SHOW. Parakeet is not our encoder. A Parakeet success proves the
information survives the codec; it does NOT prove our 0.6b streaming encoder could have got
it. Only readout 2 separates encoder from LLM:

  readout 1  our model, real domain prompt, unchanged        -> does the failure reproduce?
  readout 2  our model, "repeat the ID back, call no tools"  -> did it PERCEIVE the ID?
  readout 3  our model, ID supplied as text, no audio        -> can it copy into the slot?

Readout 1 is the gate, not a formality: sampling is greedy by default (nemo/provider.py:218),
so it must reproduce `SIN-555` exactly. If it does not, the harness is not the episode and
readouts 2-3 are uninterpretable.

Usage (needs ~3 GB of one GPU; Parakeet is 0.6b, not the 11B):
    python scripts/mode_a_probe.py                    # all four receipts
    python scripts/mode_a_probe.py --episode airline__28
    python scripts/mode_a_probe.py --keep-clips out/  # save the cut wavs for listening
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TAU2 = Path("/fsx/home/kai.li/code/tau-voice-2")
RUN = TAU2 / "data/simulations/stage2_subset_0821"
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"  # same model as scripts/fdb_v3_asr_input.py

# The four mode-A receipts, from NEMO_FAILURE_MODES.md §3. `sent` is what the model actually
# put in the tool argument -- kept here so the report is self-contained and so a future reader
# can see the error SHAPE next to the ASR hypothesis.
#
# `spoken` is an explicit regex for how THIS id sounds, not the output of a general
# spoken-id parser. That is a deliberate choice. The first version of this script tried to
# parse arbitrary speech into an id and got 3 of 4 wrong: it picked airline__3's opening
# sentence over the actual spelling turn (the "I"s and "I'm"s scored as spelled letters), and
# it appended a stray "I" from "I need" to `3RK2T9` and `W5056519`. Every one of those bugs
# is indistinguishable from an ASR failure in the output table, which is the one confusion
# this script exists to avoid. With n=4 an explicit pattern per receipt is both shorter and
# auditable, and it doubles as the segment selector.
#
# Note the ids are spoken in two different styles, which is why one parser could not cover
# them: three are spelled character by character, but airline__3's is dictated as words plus
# digits ("anya underscore garcia underscore five, nine, zero, one").
_D = {  # digit -> the ways an ASR may render it
    "0": r"(?:0|zero|oh|o)", "1": r"(?:1|one)", "2": r"(?:2|two)", "3": r"(?:3|three)",
    "4": r"(?:4|four)", "5": r"(?:5|five)", "6": r"(?:6|six)", "7": r"(?:7|seven)",
    "8": r"(?:8|eight)", "9": r"(?:9|nine)",
}
_SEP = r"[^A-Za-z0-9]{0,4}"


def _spell(chars: str) -> str:
    """Regex for an id spelled out character by character."""
    return _SEP.join(_D.get(c, re.escape(c)) for c in chars)


def _digits(ds: str) -> str:
    return _SEP.join(_D[d] for d in ds)


RECEIPTS = [
    {
        "episode": "airline__28", "expected": "SI5UKW", "sent": "SIN-555",
        "spoken": _spell("SI5UKW"),
    },
    {
        "episode": "airline__3", "expected": "anya_garcia_5901", "sent": "JFK59001",
        # Dictated as words, not spelled: "anya underscore garcia underscore five, nine, zero, one"
        "spoken": r"anya" + _SEP + r"(?:underscore)?" + _SEP + r"garcia" + _SEP
                  + r"(?:underscore)?" + _SEP + _digits("5901"),
    },
    {
        "episode": "airline__40", "expected": "3RK2T9", "sent": "3RGRK2T",
        "spoken": _spell("3RK2T9"),
    },
    {
        "episode": "retail__78", "expected": "#W5056519", "sent": "WDL500019",
        # The `#` is not spoken; "order number W, five, zero, ..." carries it.
        "spoken": _spell("W5056519"),
    },
]


def find_audio_dir(episode: str) -> Path | None:
    hits = sorted(RUN.glob(f"{episode}/artifacts/task_*/sim_*/audio/both.wav"))
    return hits[0].parent if hits else None


def parse_labels(path: Path):
    """Audacity label track: start\tend\ttext."""
    out = []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                out.append((float(parts[0]), float(parts[1]), parts[2]))
            except ValueError:
                continue
    return out


def spoken_match(text: str, spoken: str):
    """Does `text` contain this id's spoken form? Returns the matched substring or None."""
    m = re.search(spoken, text, flags=re.IGNORECASE)
    return m.group(0) if m else None


def normalise_id(s: str) -> str:
    """Compare ids without punctuation or case -- `#W5056519` vs `W5056519` is not the defect."""
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extract_user_clip(both_wav: Path, start: float, end: float, out_wav: Path, pad: float = 0.3):
    """Cut [start-pad, end+pad] of the USER channel, mono 16 kHz for Parakeet.

    Which channel is the user is DETECTED, not assumed: compare per-channel energy inside the
    user's labelled span against the assistant's. Assuming channel 0 and being wrong would
    transcribe the agent and quietly invert the whole conclusion.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(str(both_wav), always_2d=True)
    n_ch = audio.shape[1]

    if n_ch == 1:
        user_ch = 0
    else:
        asst = parse_labels(both_wav.parent / "assistant_labels.txt")
        u0, u1 = int(start * rate), int(end * rate)
        user_rms = [float(np.sqrt(np.mean(audio[u0:u1, c] ** 2))) for c in range(n_ch)]
        if asst:
            a0, a1 = int(asst[0][0] * rate), int(asst[0][1] * rate)
            asst_rms = [float(np.sqrt(np.mean(audio[a0:a1, c] ** 2))) for c in range(n_ch)]
            # The user channel is the one that is loud in the user span relative to the
            # assistant span.
            ratios = [user_rms[c] / (asst_rms[c] + 1e-9) for c in range(n_ch)]
            user_ch = int(np.argmax(ratios))
        else:
            user_ch = int(np.argmax(user_rms))

    s = max(0, int((start - pad) * rate))
    e = min(len(audio), int((end + pad) * rate))
    clip = audio[s:e, user_ch]

    if rate != 16000:
        from math import gcd
        g = gcd(int(rate), 16000)
        clip = resample_poly(clip, 16000 // g, int(rate) // g)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), clip, 16000)
    return user_ch, rate, len(clip) / 16000.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--episode", action="append", help="Limit to these episodes.")
    ap.add_argument("--keep-clips", type=Path, help="Directory to keep the cut wavs in.")
    ap.add_argument("--json-out", type=Path, help="Write the result table as JSON.")
    args = ap.parse_args()

    receipts = RECEIPTS
    if args.episode:
        want = set(args.episode)
        receipts = [r for r in RECEIPTS if r["episode"] in want]
        if not receipts:
            print(f"no receipts match {sorted(want)}", file=sys.stderr)
            return 2

    # Resolve every stimulus BEFORE loading the model, so a missing artifact fails in seconds
    # rather than after a multi-minute model load.
    plan = []
    for r in receipts:
        d = find_audio_dir(r["episode"])
        if d is None:
            print(f"!! {r['episode']}: no both.wav under {RUN}", file=sys.stderr)
            continue
        labels = parse_labels(d / "user_labels.txt")
        if not labels:
            print(f"!! {r['episode']}: no user labels", file=sys.stderr)
            continue
        # Select the turn by the same pattern used to score the ASR. If the id's spoken form is
        # not in ANY label, the receipt itself is wrong -- refuse rather than probe the wrong
        # audio and report a miss that is really a bookkeeping error.
        hits = [L for L in labels if spoken_match(L[2], r["spoken"])]
        if not hits:
            print(f"!! {r['episode']}: the id's spoken form matches no user label -- check the "
                  f"receipt. Labels: {[L[2][:60] for L in labels]}", file=sys.stderr)
            continue
        if len(hits) > 1:
            print(f"!! {r['episode']}: {len(hits)} labels match; taking the first", file=sys.stderr)
        best = hits[0]
        plan.append({**r, "dir": d, "span": (best[0], best[1]), "said": best[2]})

    if not plan:
        print("nothing to probe", file=sys.stderr)
        return 1

    print(f"probing {len(plan)} receipt(s) with {ASR_MODEL_NAME}\n")

    import tempfile

    tmpdir = args.keep_clips or Path(tempfile.mkdtemp(prefix="mode_a_"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    for p in plan:
        ch, rate, dur = extract_user_clip(
            p["dir"] / "both.wav", p["span"][0], p["span"][1], tmpdir / f"{p['episode']}.wav"
        )
        p.update(clip=tmpdir / f"{p['episode']}.wav", user_channel=ch, src_rate=rate, dur=dur)

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    hyps = model.transcribe([str(p["clip"]) for p in plan])

    rows = []
    for p, h in zip(plan, hyps):
        text = getattr(h, "text", h) if h is not None else ""
        matched = spoken_match(text, p["spoken"])
        exp_n, sent_n = normalise_id(p["expected"]), normalise_id(p["sent"])
        rows.append(
            {
                "episode": p["episode"],
                "expected": p["expected"],
                "model_sent": p["sent"],
                "span": list(p["span"]),
                "src_rate": p["src_rate"],
                "user_channel": p["user_channel"],
                "clip_seconds": round(p["dur"], 2),
                "asr_text": text,
                "asr_matched": matched,
                "asr_recovered": matched is not None,
                # How wrong the MODEL was, for scale next to the ASR verdict.
                "model_edit_distance": edit_distance(sent_n, exp_n),
            }
        )

    print(f"{'episode':<13} {'expected':<18} {'ASR recovered it?':<20} {'model sent':<11} model d")
    print("-" * 82)
    for r in rows:
        verdict = f"YES  {r['asr_matched']!r}" if r["asr_recovered"] else "NO"
        print(
            f"{r['episode']:<13} {r['expected']:<18} {verdict:<20} "
            f"{r['model_sent']:<11} {r['model_edit_distance']}"
        )

    print("\nfull ASR hypotheses -- audit the verdict against these, not the table alone:")
    for r in rows:
        print(f"  {r['episode']:<13} ch{r['user_channel']} {r['src_rate']}Hz "
              f"{r['clip_seconds']}s  {r['asr_text']!r}")

    n_ok = sum(r["asr_recovered"] for r in rows)
    print(f"\nREADOUT 0: independent ASR recovered {n_ok}/{len(rows)} ids exactly.")
    if n_ok == len(rows):
        print("  => The ids ARE in the 8 kHz signal. Mode A is inside our model; readouts 1-3\n"
              "     (encoder vs copy) are worth the GPU time.")
    elif n_ok == 0:
        print("  => NOT recoverable by a competent ASR either. Mode A is substantially a\n"
              "     STIMULUS problem -- the SFT fix is telephony/bandwidth augmentation, and\n"
              "     the model is partly exonerated. Check the hypotheses above before trusting\n"
              "     this: a normaliser bug looks identical to an ASR failure.")
    else:
        print("  => Mixed. Compare per-episode ASR vs model edit distance: where the ASR is\n"
              "     clean and the model is not, the loss is ours.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    if args.keep_clips:
        print(f"clips kept in {tmpdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
