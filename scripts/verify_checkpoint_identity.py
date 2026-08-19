#!/usr/bin/env python3
"""Is the checkpoint we evaluate the same weights as the released one?

Every number we compare against the NVIDIA-NemotronLabs-VoiceChat-11B model card assumes we
are running that model. What we actually load is
``/fsx/home/kai.li/data/voicechat/stt_extracted_lora``, produced locally on 2026-08-13 by
remapping the release into the layout our DuplexSTT training config expects (see the
key-rename trap in FINETUNING_11B.md). The two files are not byte-identical -- 45.05 GB
versus 44.38 GB -- and "we remapped it" is a claim, not evidence.

So: compare tensor names, shapes and dtypes from the safetensors headers (no data read),
then compare actual values on a sample of shared tensors. A rename leaves values untouched;
any training, merging or dtype round-trip does not. This answers a narrow question well --
"are the shared weights the same numbers" -- and deliberately does not attempt to prove the
*architecture* wired around them is equivalent.

Reads slices, not whole tensors, so it costs seconds and no GPU.

Usage:
    python scripts/verify_checkpoint_identity.py
    python scripts/verify_checkpoint_identity.py --sample-per-group 5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

RELEASED = Path("/fsx/home/kai.li/data/voicechat/voicechat-11b/model.safetensors")
OURS = Path("/fsx/home/kai.li/data/voicechat/stt_extracted_lora/model.safetensors")

# Coarse buckets, so the sample covers the whole model rather than 10 adjacent layers.
GROUPS = (
    ("llm", re.compile(r"(^|\.)llm\.|backbone|nemotron", re.I)),
    ("perception/asr", re.compile(r"perception|encoder|asr|rnnt", re.I)),
    ("tts/audio", re.compile(r"tts|audio|speech_generation|codec", re.I)),
    ("heads/embed", re.compile(r"head|embed", re.I)),
)


def header(path: Path):
    """{name: (shape, dtype)} for every tensor, without reading tensor data."""
    from safetensors import safe_open

    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            sl = f.get_slice(key)
            out[key] = (tuple(sl.get_shape()), sl.get_dtype())
    return out


def normalise(name: str) -> str:
    """Map either checkpoint's naming scheme onto a common one.

    Measured on the two files (2026-08-19), not guessed:

      released  stt_model.embed_tokens.weight              ours  embed_tokens.weight
      released  stt_model.llm.layers.0.mixer.A_log         ours  llm.base_model.model.layers.0.mixer.A_log
      released  stt_model.llm.layers.1.mixer.down_proj.weight
                                                           ours  llm.base_model.model.layers.1.mixer.down_proj.base_layer.weight

    So the release namespaces the understanding half under ``stt_model.`` and ours does not;
    ours carries PEFT's ``base_model.model.`` wrapper inside ``llm.``; and every module LoRA
    targeted gains a ``base_layer.`` level. All three are pure renames -- ``base_layer`` is
    where PEFT parks the frozen original when it wraps a module. Anything that does not
    reduce to a common name is reported as unmatched rather than force-fitted.
    """
    if name.startswith("stt_model."):
        name = name[len("stt_model."):]
    return name.replace("base_model.model.", "").replace(".base_layer.", ".")


def group_of(name: str) -> str:
    for label, pattern in GROUPS:
        if pattern.search(name):
            return label
    return "other"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--released", type=Path, default=RELEASED)
    p.add_argument("--ours", type=Path, default=OURS)
    p.add_argument("--sample-per-group", type=int, default=3)
    p.add_argument("--rows", type=int, default=4, help="Leading rows to compare per tensor.")
    args = p.parse_args()

    import torch
    from safetensors import safe_open

    print(f"released : {args.released}  ({args.released.stat().st_size / 1e9:.2f} GB)")
    print(f"ours     : {args.ours}  ({args.ours.stat().st_size / 1e9:.2f} GB)\n")

    rel_raw, ours_raw = header(args.released), header(args.ours)
    # normalised name -> original name, so slices can still be fetched by the real key.
    rel_map = {normalise(k): k for k in rel_raw}
    ours_map = {normalise(k): k for k in ours_raw}
    rel = {n: rel_raw[k] for n, k in rel_map.items()}
    ours = {n: ours_raw[k] for n, k in ours_map.items()}

    shared = sorted(set(rel) & set(ours))
    print(f"tensors: released {len(rel_raw)}, ours {len(ours_raw)}, "
          f"matched after normalising names: {len(shared)}")

    only_rel = sorted(set(rel) - set(ours))
    only_ours = sorted(set(ours) - set(rel))
    for label, names in (("only in released", only_rel), ("only in ours", only_ours)):
        src = rel if names is only_rel else ours
        print(f"  {label}: {len(names)}")
        # Prefix histogram, because 600 unmatched tensors are 600 lines of one story.
        prefixes: dict[str, int] = {}
        for name in names:
            prefixes[name.split(".")[0]] = prefixes.get(name.split(".")[0], 0) + 1
        for prefix, n in sorted(prefixes.items(), key=lambda kv: -kv[1])[:8]:
            example = next(x for x in names if x.startswith(prefix))
            print(f"     {prefix}.*  x{n}   e.g. {example} {src[example][0]}")

    mismatched = [k for k in shared if rel[k] != ours[k]]
    print(f"\nshared tensors with a different shape or dtype: {len(mismatched)}")
    for k in mismatched[:10]:
        print(f"   {k}: released {rel[k]} vs ours {ours[k]}")

    # Value comparison on a sample spread across the model.
    buckets: dict[str, list[str]] = {}
    for k in shared:
        if rel[k] == ours[k]:
            buckets.setdefault(group_of(k), []).append(k)

    print(f"\nvalue check ({args.rows} leading rows per tensor):")
    verdicts = []
    with safe_open(str(args.released), framework="pt", device="cpu") as fr, \
         safe_open(str(args.ours), framework="pt", device="cpu") as fo:
        for group in list(dict(GROUPS)) + ["other"]:
            keys = buckets.get(group, [])
            if not keys:
                print(f"  [{group}] no shared tensors")
                continue
            # Spread the sample across the bucket instead of taking the first N, which
            # would all be layer 0.
            step = max(1, len(keys) // args.sample_per_group)
            for key in keys[::step][: args.sample_per_group]:
                a = fr.get_slice(rel_map[key])[: args.rows]
                b = fo.get_slice(ours_map[key])[: args.rows]
                a32, b32 = a.to(torch.float32), b.to(torch.float32)
                exact = bool(torch.equal(a, b))
                diff = float((a32 - b32).abs().max()) if a32.numel() else 0.0
                scale = float(a32.abs().max()) if a32.numel() else 0.0
                verdicts.append(exact or diff == 0.0)
                mark = "IDENTICAL" if exact else f"DIFFERS max|d|={diff:.3g} (max|w|={scale:.3g})"
                print(f"  [{group}] {key} {tuple(a.shape)}: {mark}")

    # Tensors we have and the release does not are the ones that could be *untrained*. If a
    # randomly-initialised tensor feeds the LLM's input fusion, every output is degraded for
    # a reason no prompt fix can reach -- so stats and a copy-check against same-shaped
    # released tensors, rather than an assumption either way.
    if only_ours:
        print("\nours-only tensors (could be random init -- checking):")
        with safe_open(str(args.released), framework="pt", device="cpu") as fr, \
             safe_open(str(args.ours), framework="pt", device="cpu") as fo:
            for name in only_ours:
                if name.startswith("llm."):
                    continue  # LoRA plumbing, covered by the matched set
                t = fo.get_slice(ours_map[name])[: args.rows].to(torch.float32)
                line = (f"  {name} {ours[name][0]}: mean={t.mean():+.4g} std={t.std():.4g} "
                        f"max|w|={t.abs().max():.4g}")
                twins = [n for n in shared if rel[n][0] == ours[name][0]]
                for twin in twins:
                    ref = fr.get_slice(rel_map[twin])[: args.rows].to(torch.float32)
                    if torch.equal(ref, t):
                        line += f"  == copy of released {twin}"
                        break
                print(line)

    same = sum(verdicts)
    print(f"\n{same}/{len(verdicts)} sampled tensors bit-identical.")
    if not verdicts:
        print("VERDICT: nothing was compared -- no tensor matched by name, so `normalise` "
              "does not cover these two naming schemes. Fix the mapping; a zero-sample run "
              "is not evidence of anything.")
    elif same == len(verdicts) and not mismatched:
        print("VERDICT: the matched weights are the released weights, renamed. Comparing our "
              "results to the model card is comparing the same model's weights -- for the "
              "tensors that exist in both. Read the 'only in' lists above for what does not.")
    else:
        print("VERDICT: the checkpoints differ in value or geometry. Any comparison to the "
              "model card needs to say so -- see the per-tensor lines above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
