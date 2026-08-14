#!/usr/bin/env python3
"""
Measure the FC prompt-token cost of training cuts against `max_fc_total_tokens`.

WHY THIS EXISTS
---------------
`max_fc_total_tokens` (8000 in our config) does not truncate -- it **drops the whole
cut** (s2s_dataset.py ~L1294-1303, then a 1-second placeholder batch). The drop is
logged only when `fc_log` is on, and it is biased toward the longest, most
tool-call-rich, most expensive-to-collect trajectories.

It also scales with the domain: telecom's policy alone is 23,318 chars against
retail's 6,699. So cuts get dropped **non-uniformly by domain, silently** -- a whole
domain can be nearly absent from training while the run looks healthy.

And fixing the truncated tool schemas (advertising all 16 retail tools with real
descriptions instead of the 5 the teacher happened to call) makes the prompt bigger,
so the schema fix and this budget interact. That is what this script quantifies.

WHAT COUNTS -- replicated from `_get_fc_cut_total_prompt_tokens`
---------------------------------------------------------------
  total  = 1 + tokens(system_prompt) + 1          # BOS + prompt + EOS
         + sum over segments of supervisions[1:]:
               tokens(seg) + 2   if even index    # <SOTC>/<EOTC>
               tokens(seg) + 1   if odd index     # <EOTR>

Two traps worth knowing, both faithfully reproduced here:

1. `seg_text = (custom.get("function") or sup.text or "").strip()`. With
   `custom={"function": ""}` on speech turns -- which is REQUIRED to avoid a
   TypeError crash -- the empty string is falsy, so it falls through to `sup.text`.
   **The entire conversation transcript therefore counts toward the FC budget**, not
   just the tool-call strings. Long conversations consume the budget too.

2. The even/odd parity assumes segments strictly alternate call, response, call...
   Our generator appends all speech supervisions first and the FC pairs after, so the
   parity of the FC segments depends on how many speech turns preceded them. This
   shifts the count by ~1 token per segment. Harmless for the budget decision, but it
   means the +2/+1 special-token bookkeeping is not meaningful per-segment here.

USAGE
-----
  python scripts/measure_fc_token_budget.py \
      --shards /fsx/home/kai.li/data/voicechat/tau2_fixed/shards \
      --tool_schemas /fsx/home/kai.li/code/tau-voice-2/data/tool_schemas.json

Either argument may be omitted: with only --shards it reports the current cost of
existing cuts; with only --tool_schemas it reports the per-domain system-prompt cost.
"""

import argparse
import glob
import gzip
import json
import os
import re
import sys

MAX_FC_TOTAL_TOKENS = 8000  # examples/speechlm2/conf/finetune/s2s_duplex_stt_11b.yaml:122
TOKENIZER = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    # NeMo's AutoTokenizer.text_to_ids is encode without special tokens.
    return lambda s: tok.encode(s, add_special_tokens=False)


def format_system_prompt_with_tools(policy: str, tool_schemas: list) -> str:
    """Byte-identical to episodes_to_nemotron_training.py::format_system_prompt_with_tools."""
    tools_json = json.dumps(tool_schemas, separators=(",", ":"))
    return f"{policy}\n\n<AVAILABLE_TOOLS>{tools_json}</AVAILABLE_TOOLS>"


def fc_total_tokens(system_prompt: str, segments: list, n_ids) -> int:
    """Replicates _get_fc_cut_total_prompt_tokens (s2s_dataset.py)."""
    total = 0
    if system_prompt:
        total += 1 + len(n_ids(system_prompt)) + 1
    for idx, seg in enumerate(segments):
        total += len(n_ids(seg))
        total += 2 if idx % 2 == 0 else 1
    return total


def cut_segments(cut: dict) -> list:
    """The segment list the dataset would build from supervisions[1:]."""
    segs = []
    for sup in cut["supervisions"][1:]:
        custom = sup.get("custom") or {}
        seg = (custom.get("function") or sup.get("text") or "").strip()
        if seg:
            segs.append(seg)
    return segs


def read_cuts(shard_dir: str) -> list:
    cuts = []
    for f in sorted(glob.glob(os.path.join(shard_dir, "cuts.*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                cuts.append(json.loads(line))
    return cuts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", help="Directory of Lhotse Shar cuts.*.jsonl.gz")
    p.add_argument("--tool_schemas", help="JSON sidecar from tau-voice-2 export_tool_schemas.py")
    p.add_argument("--budget", type=int, default=MAX_FC_TOTAL_TOKENS)
    p.add_argument("--domain", default="retail", help="Domain of the cuts in --shards")
    args = p.parse_args()

    if not args.shards and not args.tool_schemas:
        p.error("give at least one of --shards / --tool_schemas")

    n_ids = load_tokenizer()
    schemas = {}
    if args.tool_schemas:
        with open(args.tool_schemas) as f:
            schemas = json.load(f)

    # ---------------------------------------------------------------- existing cuts
    transcript_costs = []
    if args.shards:
        cuts = read_cuts(args.shards)
        print(f"\n=== Existing cuts ({len(cuts)}) vs budget {args.budget} ===")
        print(f"{'cut':16s} {'sys':>7s} {'segs':>7s} {'total':>7s} {'headroom':>9s}  verdict")
        for c in cuts:
            sp = c["supervisions"][0]["text"] if c["supervisions"][0].get("speaker") == "system" else ""
            segs = cut_segments(c)
            total = fc_total_tokens(sp, segs, n_ids)
            sys_tok = 1 + len(n_ids(sp)) + 1 if sp else 0
            seg_tok = total - sys_tok
            transcript_costs.append(seg_tok)
            ok = "OK" if total <= args.budget else "DROPPED"
            print(
                f"{c['id']:16s} {sys_tok:7d} {seg_tok:7d} {total:7d} "
                f"{args.budget - total:9d}  {ok}"
            )

        # What happens if we substitute the full, correct schema for this domain?
        if args.domain in schemas:
            dom = schemas[args.domain]
            print(
                f"\n=== Same cuts with the FULL {args.domain} schema "
                f"({dom['n_tools']} tools, real descriptions) ==="
            )
            print(f"{'cut':16s} {'sys':>7s} {'segs':>7s} {'total':>7s} {'headroom':>9s}  verdict")
            for c in cuts:
                sp_old = c["supervisions"][0]["text"]
                # Reuse the cut's own policy text, swap only the tool block.
                policy = re.sub(r"\n*<AVAILABLE_TOOLS>.*?</AVAILABLE_TOOLS>", "", sp_old, flags=re.S)
                sp_new = format_system_prompt_with_tools(policy, dom["tools"])
                segs = cut_segments(c)
                total = fc_total_tokens(sp_new, segs, n_ids)
                sys_tok = 1 + len(n_ids(sp_new)) + 1
                ok = "OK" if total <= args.budget else "DROPPED"
                print(
                    f"{c['id']:16s} {sys_tok:7d} {total - sys_tok:7d} {total:7d} "
                    f"{args.budget - total:9d}  {ok}"
                )

    # ------------------------------------------------------------ per-domain budget
    if schemas:
        typical = max(transcript_costs) if transcript_costs else 0
        print(f"\n=== Per-domain system-prompt cost (full schemas) vs budget {args.budget} ===")
        if typical:
            print(
                f"(transcript allowance = {typical} tokens, the largest of the "
                f"{len(transcript_costs)} measured cuts)"
            )
        print(f"{'domain':20s} {'tools':>5s} {'policy':>7s} {'schema':>7s} {'sys':>7s} {'left':>7s}  verdict")
        rows = []
        for name, dom in sorted(schemas.items()):
            sp = format_system_prompt_with_tools(dom["policy"], dom["tools"])
            sys_tok = 1 + len(n_ids(sp)) + 1
            pol_tok = len(n_ids(dom["policy"]))
            sch_tok = sys_tok - pol_tok - 2
            left = args.budget - sys_tok - typical
            rows.append((left, name, dom, pol_tok, sch_tok, sys_tok))
        for left, name, dom, pol_tok, sch_tok, sys_tok in sorted(rows):
            verdict = "OK" if left >= 0 else "OVER BUDGET"
            print(
                f"{name:20s} {dom['n_tools']:5d} {pol_tok:7d} {sch_tok:7d} "
                f"{sys_tok:7d} {left:7d}  {verdict}"
            )
        over = [r for r in rows if r[0] < 0]
        if over:
            print(
                f"\n{len(over)} domain(s) cannot fit a typical conversation: "
                f"{', '.join(r[1] for r in over)}"
            )
            print(
                "Every FC cut in those domains would be silently dropped. Fix by\n"
                "compressing tool responses, trimming the policy, or raising\n"
                "max_fc_total_tokens -- and enable fc_log either way."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
