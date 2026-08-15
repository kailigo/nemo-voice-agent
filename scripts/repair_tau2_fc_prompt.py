#!/usr/bin/env python3
"""Retrofit the canonical tool schema and single-encoded tool responses onto existing shards.

WHY THIS EXISTS
---------------
Two defects were baked into the prototype shards:

1. The <AVAILABLE_TOOLS> block was reconstructed from the tool calls the teacher happened to
   make, so retail advertises 5 tools instead of 16, with no descriptions and every optional
   parameter marked required. That teaches "call whatever is in the prompt" rather than tool
   selection, which is precisely what the cross-domain arm measures.

2. Tool-response payloads are JSON-encoded twice -- a JSON string wrapped in {"content": ...}
   and re-encoded, escaping every quote for no added information.

Both are fixed at source in episodes_to_nemotron_training.py, so newly collected data needs
none of this. This script exists because the raw tau2 episodes these shards were built from
are gone (no data/simulations, and a filesystem-wide search for both.wav found nothing), so
the 3 existing cuts cannot be regenerated -- only repaired.

WHY IT IS SAFE TO EDIT THE SHARDS IN PLACE
------------------------------------------
Everything this touches is plain text inside cuts.*.jsonl.gz:
  - custom["system_prompt"] and supervisions[0]["text"], which are byte-identical copies
  - supervisions[*]["custom"]["function"], for the <TOOL_RESPONSE> segments only

Timings, speakers, durations and both audio tars (recording.*.tar, target_audio.*.tar) are
never read or written, so the duplex timeline is untouched by construction. We deliberately do
NOT round-trip through lhotse: the Shar reader injects shard_origin as a PosixPath that the
writer cannot serialize (see repair_tau2_shards.py), so a JSON-level rewrite is both safer and
faster. <TOOLCALL> segments are also left alone -- they are what the model generates, so their
surface form is a learned target.

WHAT IT REPORTS
---------------
Per cut, the FC prompt-token total before and after, against max_fc_total_tokens. The repair
makes the system prompt *bigger* (the honest 16-tool schema is not free), so it is expected to
push these cuts further over budget even with the response savings -- that is the point of
printing it. Use the numbers to size the max_fc_total_tokens raise.

USAGE
-----
  python scripts/repair_tau2_fc_prompt.py \
      --shards /fsx/home/kai.li/data/voicechat/tau2_fixed/shards \
      --tool_schemas /fsx/home/kai.li/code/tau-voice-2/data/tool_schemas.json \
      --domain retail \
      --output /fsx/home/kai.li/data/voicechat/tau2_canonical

Add --in_place to overwrite --shards instead of writing a new tree. Default is a copy, because
these 3 cuts are unregenerable.
"""

import argparse
import glob
import gzip
import json
import os
import re
import shutil
import sys

TOOLS_BLOCK = re.compile(r"\n*<AVAILABLE_TOOLS>.*?</AVAILABLE_TOOLS>", re.S)
RESPONSE = re.compile(r"^<TOOL_RESPONSE>(.*)</TOOL_RESPONSE>$", re.S)


def format_system_prompt_with_tools(policy: str, tool_schemas: list) -> str:
    """Byte-identical to episodes_to_nemotron_training.py::format_system_prompt_with_tools."""
    tools_json = json.dumps(tool_schemas, separators=(",", ":"))
    return f"{policy}\n\n<AVAILABLE_TOOLS>{tools_json}</AVAILABLE_TOOLS>"


def swap_tool_block(system_prompt: str, tool_schemas: list) -> str:
    """Replace the <AVAILABLE_TOOLS> block, keeping the cut's own policy text verbatim.

    The policy may have been overridden per-run, so we reuse what is in the cut rather than
    the sidecar's copy.
    """
    policy = TOOLS_BLOCK.sub("", system_prompt)
    return format_system_prompt_with_tools(policy, tool_schemas)


def unnest_response(seg: str) -> str:
    """Re-encode a <TOOL_RESPONSE> segment with a single-encoded, compact payload.

    Mirrors episodes_to_nemotron_training.py::format_tool_response: only dict/list payloads
    are un-nested, so a bare "123" does not silently become an int. Returns the input
    unchanged if it is not a well-formed response segment.
    """
    m = RESPONSE.match(seg.strip())
    if not m:
        return seg
    try:
        obj = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return seg
    if not isinstance(obj, list):
        return seg
    changed = False
    for entry in obj:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, (dict, list)):
                entry["content"] = parsed
                changed = True
    if not changed:
        # Still re-emit compactly: harmless, and keeps every segment consistently encoded.
        pass
    return f"<TOOL_RESPONSE>{json.dumps(obj, separators=(',', ':'))}</TOOL_RESPONSE>"


def fc_total_tokens(cut: dict, n_ids) -> int:
    """Replicates _get_fc_cut_total_prompt_tokens (s2s_dataset.py ~L1294).

    Note the fall-through to sup["text"]: with custom={"function": ""} on speech turns the
    empty string is falsy, so the whole transcript counts toward the FC budget too.
    """
    sup0 = cut["supervisions"][0]
    total = 0
    if sup0.get("speaker") == "system" and sup0.get("text"):
        total += 1 + len(n_ids(sup0["text"])) + 1
    segs = []
    for sup in cut["supervisions"][1:]:
        custom = sup.get("custom") or {}
        seg = (custom.get("function") or sup.get("text") or "").strip()
        if seg:
            segs.append(seg)
    for idx, seg in enumerate(segs):
        total += len(n_ids(seg)) + (2 if idx % 2 == 0 else 1)
    return total


def repair_cut(cut: dict, tool_schemas: list) -> dict:
    """Return the cut with a canonical tool block and single-encoded responses."""
    old_sp = cut["supervisions"][0]["text"]
    new_sp = swap_tool_block(old_sp, tool_schemas)
    cut["supervisions"][0]["text"] = new_sp
    # custom["system_prompt"] is a byte-identical copy; the dataset reads one or the other
    # depending on the code path, so both must move together.
    if "system_prompt" in cut.get("custom", {}):
        cut["custom"]["system_prompt"] = new_sp
    for sup in cut["supervisions"]:
        custom = sup.get("custom") or {}
        fn = custom.get("function")
        if fn and fn.lstrip().startswith("<TOOL_RESPONSE>"):
            custom["function"] = unnest_response(fn)
    return cut


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True, help="Shar directory containing cuts.*.jsonl.gz")
    p.add_argument("--tool_schemas", required=True, help="Sidecar from export_tool_schemas.py")
    p.add_argument("--domain", required=True, help="Domain of the cuts in --shards")
    p.add_argument("--output", help="Write a repaired copy here (default unless --in_place)")
    p.add_argument("--in_place", action="store_true", help="Overwrite --shards")
    p.add_argument("--budget", type=int, default=8000, help="max_fc_total_tokens to check against")
    p.add_argument("--dry_run", action="store_true", help="Report only, write nothing")
    args = p.parse_args()

    if not args.in_place and not args.output and not args.dry_run:
        p.error("give --output, or --in_place, or --dry_run")

    with open(args.tool_schemas) as f:
        schemas = json.load(f)
    if args.domain not in schemas:
        p.error(f"domain {args.domain!r} not in sidecar; have: {', '.join(sorted(schemas))}")
    tools = schemas[args.domain]["tools"]
    print(f"Repairing with the canonical {args.domain} schema: {len(tools)} tools")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2", local_files_only=True
    )
    n_ids = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731

    # Stage the destination first so a failure never leaves a half-written tree in place.
    dest = args.shards if args.in_place else args.output
    if not args.dry_run and not args.in_place:
        os.makedirs(dest, exist_ok=True)
        for src in sorted(glob.glob(os.path.join(args.shards, "*"))):
            name = os.path.basename(src)
            if name.startswith("cuts."):
                continue  # rewritten below
            # Audio is never modified. Hardlink when possible to avoid copying 43 MB.
            dst = os.path.join(dest, name)
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        print(f"Linked audio tars into {dest}")

    shard_files = sorted(glob.glob(os.path.join(args.shards, "cuts.*.jsonl.gz")))
    if not shard_files:
        p.error(f"no cuts.*.jsonl.gz in {args.shards}")

    print(f"\n{'cut':16s} {'before':>8s} {'after':>8s} {'delta':>7s} {'headroom':>9s}  verdict")
    n_over = 0
    for shard in shard_files:
        with gzip.open(shard, "rt") as fh:
            cuts = [json.loads(line) for line in fh if line.strip()]

        out_lines = []
        for cut in cuts:
            before = fc_total_tokens(cut, n_ids)
            cut = repair_cut(cut, tools)
            after = fc_total_tokens(cut, n_ids)
            over = after > args.budget
            n_over += over
            print(
                f"{cut['id']:16s} {before:8d} {after:8d} {after - before:+7d} "
                f"{args.budget - after:9d}  {'OVER BUDGET' if over else 'OK'}"
            )
            out_lines.append(json.dumps(cut))

        if not args.dry_run:
            out_path = os.path.join(dest, os.path.basename(shard))
            tmp = out_path + ".tmp"
            with gzip.open(tmp, "wt") as fh:
                for line in out_lines:
                    fh.write(line + "\n")
            os.replace(tmp, out_path)

    if args.dry_run:
        print("\n--dry_run: nothing written.")
    else:
        print(f"\nWrote repaired shards to {dest}")

    if n_over:
        print(
            f"\n{n_over} cut(s) exceed max_fc_total_tokens={args.budget} and would be DROPPED\n"
            "silently (s2s_dataset.py substitutes a 1-second placeholder batch). Raise\n"
            "max_fc_total_tokens in the training config and turn fc_log on."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
