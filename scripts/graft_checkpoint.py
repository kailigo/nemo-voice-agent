#!/usr/bin/env python3
"""Assemble a servable full VoiceChat checkpoint from a trained STT half.

Why this is needed
------------------
Training produces only the understanding half. `stt_extracted_lora` holds 984 tensors in the
DuplexSTT layout; the released checkpoint holds 1632 in the `stt_model.*` / `tts_model.*`
layout that the NIM container's `generate-triton-repo` consumes. Nothing in the training path
emits the speech half, because nothing in the training path supervises it: `target_audio` is
"saved for debugging" only (`duplex_stt_model.py:651`), never a loss target.

So a trained model cannot be served until its tensors are put back into the released layout
with the untrained speech half reattached. That is this script.

Why reattaching an untrained decoder is safe
-------------------------------------------
The trunk and the TTS module are coupled through *discrete subword ids only*
(`nemotron_voicechat.py:697-712`): `current_subword_id = inference_state["gen_text"][:, t]` is
handed to `tts_model.infer_codes_one_step(...)`. No hidden states and no LLM latents cross the
boundary, and the TTS module carries its own text front-end (`tts_model.embed_subword`, a
byte-level encoder over a (257, 1152) embedding plus `special_flags` / `is_continuation` tables
keyed on the trunk's 131072-token vocab) and its own `audio_codec`. It never reads
`stt_model.embed_tokens`.

Consequence: a frozen decoder vocodes whatever *text* the trained trunk emits, at released
voice quality, provided the vocab stays at 131072. The script asserts that rather than
assuming it -- a vocab resize is the one edit that would silently produce garbled speech.

The four operations
-------------------
1. **Merge LoRA.** r=32, alpha=64 -> W = base_layer + 2.0 * B @ A, then the `base_layer` level
   collapses away. `stt_extracted_lora` has 66 `base_layer` keys and zero adapters (it is the
   *input* to LoRA training), so on that file this step is a verified no-op -- which is exactly
   what makes it usable as a correctness test of everything else.
2. **Rename** onto the released scheme: `stt_model.` prefix, drop PEFT's `base_model.model.`
   wrapper. This is `verify_checkpoint_identity.normalise` run backwards, and it is imported
   from there rather than restated, so the two cannot drift.
3. **Drop the two dead heads.** `asr_head.weight` and `embed_asr_tokens.weight` are bit-identical
   copies of `lm_head` / `embed_tokens` seeded by `remap_checkpoint_for_lora.py
   --warm-start-asr-head` against a future `predict_user_text: true`. Both configs set that
   false, so `duplex_stt_model.py:274` never builds the modules and the tensors are never
   loaded or trained. They have no counterpart in the released layout and are 4.7 GB of fp32
   nothing.
4. **Append the untrained halves** from the released checkpoint: 635 `tts_model.*` (already
   extracted to `stt_extracted_tts`) and the 15 `stt_model.rnnt_decoder` / `rnnt_joint` tensors
   our extraction dropped. The RNNT branch does user-side transcription against its own 1024-token
   tokenizer (`config.json:489-493`), entirely disjoint from the trunk vocab, so SFT cannot have
   affected it and copying it across is purely additive.

The self-test
-------------
Run with `--verify-against-released` on an *untrained* STT half and the output must be
bit-identical to the released checkpoint. If it is, the graft is proven by construction: same
bytes and same keys as a file the container already serves, so servability needs no GPU to
establish. Once SFT lands, the only thing that has changed is the tensor values.

Usage:
    # prove the mechanism, no GPU, no trained checkpoint needed
    python scripts/graft_checkpoint.py --verify-against-released --dry-run

    # write a servable checkpoint from a trained half
    python scripts/graft_checkpoint.py --stt <sft_out>/model.safetensors --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from verify_checkpoint_identity import normalise  # noqa: E402  (single source of the rename)

VOICECHAT = Path("/fsx/home/kai.li/data/voicechat")
RELEASED = VOICECHAT / "voicechat-11b" / "model.safetensors"
STT = VOICECHAT / "stt_extracted_lora" / "model.safetensors"
TTS = VOICECHAT / "stt_extracted_tts" / "model.safetensors"

# Never instantiated (`predict_user_text: false` in both configs), no released counterpart.
DEAD_HEADS = ("asr_head.weight", "embed_asr_tokens.weight")
# Present in the release, dropped by our extraction, untouched by SFT.
FROM_RELEASED = ("stt_model.rnnt_decoder.", "stt_model.rnnt_joint.")
LORA_A = re.compile(r"^(?P<mod>.+)\.lora_A(?:\.(?P<adapter>[^.]+))?\.weight$")


def read_keys(path: Path) -> dict:
    """{name: (shape, dtype)} from the header alone -- no tensor data, so this costs seconds."""
    from safetensors import safe_open

    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            sl = f.get_slice(key)
            out[key] = (tuple(sl.get_shape()), sl.get_dtype())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stt", type=Path, default=STT, help="Trained understanding half.")
    p.add_argument("--tts", type=Path, default=TTS, help="Untrained speech half (635 tensors).")
    p.add_argument("--released", type=Path, default=RELEASED,
                   help="Source of the RNNT branch, and the reference for --verify-against-released.")
    p.add_argument("--out", type=Path, default=None, help="Output directory.")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve and check the key mapping without reading or writing tensors.")
    p.add_argument("--verify-against-released", action="store_true",
                   help="Assert the result is bit-identical to --released. Only meaningful on an "
                        "untrained STT half, where it proves the mechanism. With no --out this "
                        "streams tensor by tensor, so it costs no disk and ~1 tensor of RAM.")
    args = p.parse_args()
    if not args.dry_run and args.out is None and not args.verify_against_released:
        p.error("--out is required unless --dry-run or --verify-against-released")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    rel_keys = read_keys(args.released)
    stt_keys = read_keys(args.stt)
    tts_keys = read_keys(args.tts)
    print(f"stt      : {args.stt}  ({len(stt_keys)} tensors)")
    print(f"tts      : {args.tts}  ({len(tts_keys)} tensors)")
    print(f"released : {args.released}  ({len(rel_keys)} tensors)\n")

    # -- 1/2/3: plan the STT half's key mapping from the header alone --------------------
    # `normalise` already strips `stt_model.`, `base_model.model.` and `.base_layer.`, so
    # composing it with the prefix is the whole rename -- including the post-merge collapse.
    planned, dead, adapters = {}, [], {}
    for key in stt_keys:
        m = LORA_A.match(key)
        if m:
            # Resolve the triple now, from the header, so a malformed adapter set fails during
            # planning rather than 40 GB into a materialise.
            mod, adapter = m.group("mod"), m.group("adapter")
            suffix = f".{adapter}.weight" if adapter else ".weight"
            b_key, base_key = f"{mod}.lora_B{suffix}", f"{mod}.base_layer.weight"
            for needed, what in ((b_key, "lora_B"), (base_key, "frozen base")):
                if needed not in stt_keys:
                    raise RuntimeError(f"{key} has no {what} at {needed}; refusing to guess.")
            adapters[base_key] = (key, b_key)
            continue
        if ".lora_B" in key:
            continue  # consumed with its lora_A
        base = normalise(key)
        if base in DEAD_HEADS:
            dead.append(key)
            continue
        planned[f"stt_model.{base}"] = key
    print(f"  renamed {len(planned)} STT tensors onto the released scheme")
    print(f"  dropped {len(dead)} dead head(s): {[normalise(k) for k in dead]}")
    print(f"  LoRA adapter pairs to merge: {len(adapters)} "
          f"(of {sum('.base_layer.' in k for k in stt_keys)} wrapped modules)")

    from_released = sorted(k for k in rel_keys if k.startswith(FROM_RELEASED))
    tts_plan = {f"tts_model.{k}": k for k in tts_keys}
    print(f"  appending {len(tts_plan)} tts_model.* and {len(from_released)} RNNT tensor(s) "
          f"from the released checkpoint")

    # -- the mapping must land exactly on the released key set --------------------------
    result_keys = set(planned) | set(tts_plan) | set(from_released)
    missing = sorted(set(rel_keys) - result_keys)
    extra = sorted(result_keys - set(rel_keys))
    print(f"\nkey set: {len(result_keys)} vs released {len(rel_keys)}; "
          f"missing {len(missing)}, unexpected {len(extra)}")
    for label, names in (("missing", missing), ("unexpected", extra)):
        for name in names[:8]:
            print(f"   {label}: {name}")
        if len(names) > 8:
            print(f"   ... and {len(names) - 8} more {label}")
    if missing or extra:
        print("\nVERDICT: the mapping does not reproduce the released key set. A checkpoint "
              "written now would be missing modules or carry ones the server does not load, "
              "and `generate-triton-repo` is not the place to find that out.")
        return 1

    # The vocab assertion: the TTS front-end's flag tables are sized on the trunk vocab, so a
    # resize would desynchronise them and garble every utterance.
    vocab = stt_keys[planned["stt_model.embed_tokens.weight"]][0][0]
    rel_vocab = rel_keys["stt_model.embed_tokens.weight"][0][0]
    flags = rel_keys["tts_model.tts_model.embed_subword.bos_eos_emb.special_flags"][0][0]
    ok = vocab == rel_vocab == flags
    print(f"\nvocab: trunk {vocab}, released {rel_vocab}, TTS special_flags {flags} "
          f"-> {'consistent' if ok else 'MISMATCH'}")
    if not ok:
        print("VERDICT: the trunk vocab no longer matches the TTS front-end's flag tables. "
              "The grafted model would speak garbage. Do not serve it.")
        return 1

    print("\nmapping resolves cleanly to the released layout.")
    if args.dry_run:
        print("(--dry-run: no tensors read or written)")
        return 0

    # -- materialise, one tensor at a time -----------------------------------------------
    # A generator rather than a dict so `--verify-against-released` costs ~1 tensor of RAM and
    # no disk. The write path collects it, because `save_file` needs the whole mapping -- but
    # the self-test, which is the thing we run repeatedly, does not.
    def produce():
        scaling = args.lora_alpha / args.lora_r
        merged = 0
        with safe_open(str(args.stt), framework="pt", device="cpu") as f:
            for out_key, stt_key in planned.items():
                t = f.get_tensor(stt_key)
                if stt_key in adapters:
                    a_key, b_key = adapters[stt_key]
                    a = f.get_tensor(a_key).to(torch.float32)
                    b = f.get_tensor(b_key).to(torch.float32)
                    delta = (b @ a) * scaling
                    if delta.shape != t.shape:
                        raise RuntimeError(
                            f"{stt_key}: (B @ A) is {tuple(delta.shape)} but base is "
                            f"{tuple(t.shape)}. Wrong r/alpha, or a transposed convention -- "
                            f"do not silently broadcast."
                        )
                    t = (t.to(torch.float32) + delta).to(t.dtype)
                    merged += 1
                yield out_key, t
        if merged != len(adapters):
            raise RuntimeError(f"merged {merged} adapters but planned {len(adapters)}")
        if adapters:
            print(f"  merged {merged} adapter pair(s) at scaling={scaling:g} "
                  f"(r={args.lora_r}, alpha={args.lora_alpha})", flush=True)
        with safe_open(str(args.tts), framework="pt", device="cpu") as f:
            for k in tts_keys:
                yield f"tts_model.{k}", f.get_tensor(k)
        with safe_open(str(args.released), framework="pt", device="cpu") as f:
            for k in from_released:
                yield k, f.get_tensor(k)

    if args.verify_against_released:
        print("\nverifying against the release, streaming tensor by tensor...", flush=True)
        differing, seen = [], 0
        with safe_open(str(args.released), framework="pt", device="cpu") as f:
            for key, tensor in produce():
                seen += 1
                if not torch.equal(f.get_tensor(key), tensor):
                    differing.append(key)
                del tensor
        if differing:
            print(f"{len(differing)} of {seen} tensors differ, e.g.:")
            for k in differing[:10]:
                print(f"   {k}")
            print("\nVERDICT: not bit-identical. On an untrained STT half that means the graft "
                  "itself is lossy -- fix it before trusting it with real SFT deltas. On a "
                  "trained half, differences are expected and this flag is meaningless.")
            return 1
        print(f"VERDICT: all {seen} tensors bit-identical to the release. The graft is proven "
              f"by construction -- it reproduces a file the container already serves, so "
              f"servability needs no GPU to establish.")
        if args.out is None:
            return 0

    print("\nmaterialising (this holds the whole checkpoint in RAM to write it)...", flush=True)
    out = dict(produce())
    if set(out) != set(rel_keys):
        raise RuntimeError(
            f"post-materialisation key set drifted from the plan: "
            f"{len(set(rel_keys) - set(out))} missing, {len(set(out) - set(rel_keys))} extra. "
            f"The merge or the rename did something the header-only plan did not predict."
        )
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "model.safetensors"
    print(f"writing {dest} ...", flush=True)
    save_file(out, str(dest), metadata={"format": "pt"})
    print(f"  {dest.stat().st_size / 1e9:.2f} GB, {len(out)} tensors")

    # The server needs more than the tensors: the released config.json describes the wiring
    # (including `_rnnt_merge_info`) and `rnnt_tokenizer/` is the RNNT branch's own 1024-token
    # vocabulary, which lives outside the safetensors entirely.
    src = args.released.parent
    for name in ("config.json", "rnnt_tokenizer"):
        s, d = src / name, args.out / name
        if not s.exists():
            print(f"  WARNING: {s} absent; the server may refuse to load without it")
            continue
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
        print(f"  copied {name}")
    (args.out / "graft.json").write_text(json.dumps({
        "stt": str(args.stt), "tts": str(args.tts), "released": str(args.released),
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "dropped_dead_heads": [normalise(k) for k in dead],
        "rnnt_from_released": from_released,
        "tensors": len(out),
    }, indent=2))
    print("  wrote graft.json (provenance -- which halves, which merge, what was dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
