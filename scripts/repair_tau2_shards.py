#!/usr/bin/env python3
"""Repair the tau2 prototype shards so DuplexSTTModel can actually train on them.

The shards in data/tau2_training_samples/ were generated before two format bugs were understood.
Their semantic content is fine -- correct roles, a coherent duplex timeline, strictly alternating
<TOOLCALL>/<TOOL_RESPONSE> pairs -- but they cannot be loaded. This script fixes the container and
the supervision metadata without touching any of that content.

The generator (episodes_to_nemotron_training.py) has been fixed at source, so freshly produced
data needs none of this. This exists because the tau2 simulations these were built from lived in
/tmp and are gone, so repairing is the only way to recover the samples.

WHAT IS WRONG, AND WHY IT IS FATAL RATHER THAN COSMETIC
------------------------------------------------------
1. metadata.json sits inside the shard directory.
   Lhotse infers field names from EVERY entry in the dir (lhotse/shar/readers/lazy.py ~L202):
       fields = set(p.stem.split(".")[0] for p in in_dir.glob("*"))
   so metadata.json becomes a phantom field "metadata" and is then read as JSONL. It is
   pretty-printed, so the first line is "{\n", and json.loads("{\n") raises exactly the error
   seen in training:
       JSONDecodeError: Expecting property name enclosed in double quotes: line 2 column 1 (char 2)
   Fix: move it one level up, out of the shard dir. Nothing but Shar files may live there.

2. Every speech supervision has custom=None.
   s2s_dataset.py gates ALL function-call extraction on supervisions[1] (~L1593):
       if len(cut.supervisions) > 1 and 'function' in cut.supervisions[1].custom:
   supervisions[1] is a speech turn, and lhotse's SupervisionSegment.custom defaults to None, so
   `'function' in None` raises TypeError -- the dataloader dies, it does not degrade quietly.
   The next line then calls sup.custom.get("function") across ALL of supervisions[1:], so patching
   index 1 alone just moves the crash. Every supervision needs the dict.
   Fix: set custom={"function": ""} wherever the key is absent. Empty string is the safe value --
   the text-channel inclusion rule (~L2274) keeps a turn when custom['function'] == '', so speech
   turns continue to train the text channel exactly as before.

3. One shard, so only one GPU can be fed.
   Lhotse hands whole shards to ranks. Re-sharded at one cut per shard, which is the most
   parallelism 3 cuts can support -- still only 3 ranks. 3 conversations is a smoke test, not a
   training set.

Usage:
    python scripts/repair_tau2_shards.py \
        --input  data/tau2_training_samples \
        --output /fsx/home/kai.li/data/voicechat/tau2_fixed
"""

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

# Field names that legitimately belong to a Lhotse Shar directory for this dataset. Anything else
# in the directory (metadata.json, README, .DS_Store, editor swapfiles) becomes a phantom field.
SHAR_FIELDS = ("cuts", "recording", "target_audio")

# Provenance keys the Shar READER injects into cut.custom that the Shar WRITER cannot serialize.
# shard_origin comes back as a pathlib.PosixPath, and json.dumps dies on it with
# "TypeError: Object of type PosixPath is not JSON serializable" partway through writing -- after
# some shards already exist on disk. They carry no training signal, so drop them on round-trip.
READER_INJECTED_KEYS = ("shard_origin", "shar_epoch")


def stage_clean_shar(src: Path, staging: Path):
    """Symlink only the real Shar files into a clean dir so Lhotse can open it.

    We cannot read `src` directly: metadata.json in there is precisely what breaks field
    discovery. Symlinks avoid copying 43 MB of audio.
    """
    staging.mkdir(parents=True, exist_ok=True)
    kept, skipped = [], []
    for p in sorted(src.glob("*")):
        if p.name.split(".")[0] in SHAR_FIELDS:
            (staging / p.name).symlink_to(p.resolve())
            kept.append(p.name)
        else:
            skipped.append(p.name)
    return kept, skipped


def strip_reader_keys(cut):
    """Drop reader-injected provenance from cut.custom so the cut can be re-serialized."""
    custom = cut.custom or {}
    dropped = [k for k in READER_INJECTED_KEYS if k in custom]
    if not dropped:
        return cut, dropped
    cleaned = {k: v for k, v in custom.items() if k not in READER_INJECTED_KEYS}
    return replace(cut, custom=cleaned), dropped


def patch_supervisions(cut):
    """Give every supervision a custom dict containing a 'function' key."""
    new_sups, patched = [], 0
    for sup in cut.supervisions:
        custom = sup.custom
        if custom is None:
            new_sups.append(replace(sup, custom={"function": ""}))
            patched += 1
        elif "function" not in custom:
            merged = dict(custom)
            merged["function"] = ""
            new_sups.append(replace(sup, custom=merged))
            patched += 1
        else:
            new_sups.append(sup)
    return replace(cut, supervisions=new_sups), patched


def validate(cut) -> list:
    """Re-implement the dataset's own preconditions so failures surface here, not in a dataloader.

    Mirrors s2s_dataset.py: the supervisions[1] gate (~L1593), the segment filter (~L1594), and
    the [call, response, call, response, ...] positional pairing assumption (~L473).
    """
    issues = []
    sups = cut.supervisions

    if not sups:
        return ["no supervisions"]
    if sups[0].speaker != "system":
        issues.append(f"supervisions[0].speaker={sups[0].speaker!r}, expected 'system'")

    # The gate. This is the one that used to raise TypeError.
    if len(sups) > 1:
        if sups[1].custom is None:
            issues.append("supervisions[1].custom is None -> TypeError at s2s_dataset.py:1593")
        elif "function" not in sups[1].custom:
            issues.append("supervisions[1].custom lacks 'function' -> FC data silently dropped")

    # The filter walks all of supervisions[1:] calling .custom.get(...).
    for i, s in enumerate(sups[1:], start=1):
        if s.custom is None:
            issues.append(f"supervisions[{i}].custom is None -> AttributeError at s2s_dataset.py:1596")
            break

    fc = [s for s in sups[1:] if ((s.custom or {}).get("function") or "").strip() != ""]
    if not fc:
        issues.append("no function-calling supervisions")
    if len(fc) % 2 != 0:
        issues.append(f"{len(fc)} FC segments is odd; pairing assumes call/response pairs")
    for i, s in enumerate(fc):
        text = s.custom["function"]
        want_call = i % 2 == 0
        is_call = "<TOOLCALL>" in text
        is_resp = "TOOL_RESPONSE" in text or "TOOLRESPONSE" in text
        if want_call and not is_call:
            issues.append(f"FC segment {i} should be a <TOOLCALL> (even index) but is not")
        if not want_call and not is_resp:
            issues.append(f"FC segment {i} should be a <TOOL_RESPONSE> (odd index) but is not")

    # Turns that carry both text and a function call lose their text entirely (~L2275).
    both = [s.id for s in sups if (s.custom or {}).get("function") and (s.text or "").strip()]
    if both:
        issues.append(f"{len(both)} supervision(s) carry BOTH text and function; text is dropped: {both[:3]}")

    if cut.recording is None:
        issues.append("missing recording")
    elif cut.recording.sampling_rate != 16000:
        issues.append(f"user recording sr={cut.recording.sampling_rate}, expected 16000")

    target = (cut.custom or {}).get("target_audio")
    if target is None:
        issues.append("missing custom['target_audio']")
    elif target.sampling_rate != 22050:
        issues.append(f"target_audio sr={target.sampling_rate}, expected 22050")

    if not (cut.custom or {}).get("s2s_duplex_function_calling"):
        issues.append("custom['s2s_duplex_function_calling'] not truthy -> FC path not taken")

    sp = (cut.custom or {}).get("system_prompt") or ""
    if "<AVAILABLE_TOOLS>" not in sp:
        issues.append("system_prompt missing <AVAILABLE_TOOLS>")

    # Supervisions must start inside the cut, else compute_num_frames lands out of range (~L2276).
    for s in sups:
        if s.start < 0 or s.start > cut.duration:
            issues.append(f"supervision {s.id} start={s.start} outside [0, {cut.duration}]")
            break

    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Existing shar dir (may contain metadata.json)")
    ap.add_argument("--output", required=True, help="Output dir; shards go in <output>/shards")
    ap.add_argument("--shard_size", type=int, default=1, help="Cuts per shard (default 1)")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing output dir")
    args = ap.parse_args()

    from lhotse import CutSet

    src = Path(args.input).resolve()
    # Accept either layout: the shard dir itself, or a parent holding shards/ (what the generator
    # emits, and what data/tau2_training_samples/ now uses).
    if (src / "shards").is_dir():
        src = src / "shards"
    out = Path(args.output).resolve()
    out_shards = out / "shards"

    print("=" * 78)
    print("REPAIR TAU2 SHARDS")
    print("=" * 78)
    print(f"  input : {src}")
    print(f"  output: {out}")

    if out.exists():
        if not args.force:
            sys.exit(f"ABORT: {out} exists. Pass --force to overwrite.")
        shutil.rmtree(out)
    out_shards.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td) / "clean_shar"
        kept, skipped = stage_clean_shar(src, staging)
        print(f"\n  Shar files staged : {kept}")
        print(f"  EXCLUDED (would become phantom Lhotse fields): {skipped or 'none'}")

        print("\n  Reading cuts ...")
        cuts = list(CutSet.from_shar(in_dir=str(staging)))
        print(f"  Read {len(cuts)} cuts")

        fixed, total_patched, failed = [], 0, 0
        for cut in cuts:
            before = validate(cut)
            new_cut, n = patch_supervisions(cut)
            new_cut, dropped = strip_reader_keys(new_cut)
            total_patched += n
            after = validate(new_cut)
            status = "OK" if not after else "STILL BROKEN"
            print(f"\n  cut {cut.id}  dur={cut.duration:.1f}s  sups={len(cut.supervisions)}")
            print(f"    patched {n} supervision(s) with custom={{'function': ''}}")
            print(f"    dropped reader-injected custom keys: {dropped or 'none'}")
            print(f"    issues before: {len(before)}")
            for m in before:
                print(f"      - {m}")
            print(f"    issues after : {len(after)}  [{status}]")
            for m in after:
                print(f"      - {m}")
            if after:
                failed += 1
            fixed.append(new_cut)

        if failed:
            sys.exit(f"\nABORT: {failed} cut(s) still invalid after repair; not writing output.")

        print(f"\n  Writing {len(fixed)} cuts at {args.shard_size} cut(s)/shard ...")
        CutSet.from_cuts(fixed).to_shar(
            output_dir=str(out_shards),
            fields={"recording": "wav", "target_audio": "wav"},
            shard_size=args.shard_size,
        )

    n_shards = len(list(out_shards.glob("cuts.*.jsonl.gz")))
    print(f"  Wrote {n_shards} shard(s) -> {out_shards}")

    # metadata.json goes OUTSIDE the shard dir. Putting it back inside recreates bug #1.
    meta = {}
    for cand in (src / "metadata.json", src.parent / "metadata.json"):
        if cand.exists():
            meta = json.loads(cand.read_text())
            break
    meta.update({
        "num_cuts": len(fixed),
        "num_shards": n_shards,
        "shar_path": str(out_shards),
        "repaired_by": "scripts/repair_tau2_shards.py",
        "repairs": [
            "moved metadata.json out of the shard dir (phantom Lhotse field -> JSONDecodeError)",
            "added custom={'function': ''} to supervisions lacking the key (TypeError at s2s_dataset.py:1593)",
            f"re-sharded from 1 shard to {n_shards} ({args.shard_size} cut(s)/shard)",
        ],
        "max_data_parallel_ranks": n_shards,
    })
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  Wrote metadata -> {out/'metadata.json'} (outside the shard dir, deliberately)")

    # Read back through the real Lhotse path and re-assert. A repair that cannot be re-read is
    # not a repair.
    print("\n  Verifying by re-reading the written shards ...")
    roundtrip = list(CutSet.from_shar(in_dir=str(out_shards)))
    bad = 0
    for cut in roundtrip:
        issues = validate(cut)
        fc = len([s for s in cut.supervisions[1:] if ((s.custom or {}).get("function") or "").strip()])
        print(f"    {cut.id}: {len(cut.supervisions)} sups, {fc} FC segments, "
              f"{'clean' if not issues else 'ISSUES: ' + '; '.join(issues)}")
        bad += bool(issues)
        # Prove the audio actually decodes and both channels span the conversation.
        ua = cut.load_audio()
        ta = cut.load_custom("target_audio")
        print(f"      user audio {ua.shape} @16k = {ua.shape[-1]/16000:.1f}s | "
              f"agent audio {ta.shape} @22050 = {ta.shape[-1]/22050:.1f}s | cut {cut.duration:.1f}s")

    if bad:
        sys.exit(f"\nFAIL: {bad} cut(s) invalid after round-trip.")

    print("\n" + "=" * 78)
    print(f"SUCCESS: {len(roundtrip)} cuts, {n_shards} shards, usable on up to {n_shards} GPU(s).")
    print("=" * 78)
    print("\nTrain with:")
    print(f"  data.train_ds.input_cfg.0.shar_path={out_shards}")
    print(f"  data.validation_ds.datasets.val_set_0.shar_path={out_shards}")


if __name__ == "__main__":
    main()
