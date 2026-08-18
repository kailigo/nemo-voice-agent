#!/usr/bin/env python
"""CPU unit test for StreamingFCSession._parse_call -- no model, no GPU, no shards.

This exists because the parser had a bug that only a GPU run surfaced: the regex was
anchored to end-of-string, so the trained `<TOOLCALL>[...]</TOOLCALL>` format never
matched and every call decoded to name="" with empty arguments. That is a total failure of
tool calling, and nothing in the pipeline would have raised -- it would have looked like a
model that picks tools badly. A 30-line CPU test is the right place to catch it.

Run: python scripts/test_parse_call.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_parser():
    """Import _parse_call without importing NeMo (which needs torch + 90 s of imports).

    The function is a @staticmethod with no self/model dependency, so the module's two
    regexes plus the function body are all it needs. Exec'ing the source keeps this test
    honest -- it tests the shipped code, not a copy -- while staying import-free.
    """
    src = (REPO / "nemo/collections/speechlm2/models/streaming_fc_session.py").read_text()

    ns: dict = {"re": re}
    for name in ("_TOOLCALL_BLOCK", "_TOOLCALL_TAGS"):
        m = re.search(rf"^{name} = re\.compile\(.*?\)$", src, re.M)
        assert m, f"could not find {name} in the module source"
        exec(m.group(0), ns)

    m = re.search(r"^    @staticmethod\n    def _parse_call.*?(?=\n    def )", src, re.S | re.M)
    assert m, "could not find _parse_call in the module source"
    body = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in m.group(0).splitlines())
    body = body.replace("@staticmethod\n", "", 1)
    ns.setdefault("json", __import__("json"))
    ns.setdefault("Optional", object)
    exec(body, ns)
    return ns["_parse_call"]


CASES = [
    # (label, raw, expected_name, expected_args_subset)
    (
        "trained format, both tags",
        '<TOOLCALL>[{"name": "find_user_id_by_name_zip", "arguments": '
        '{"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}}]</TOOLCALL>',
        "find_user_id_by_name_zip",
        {"zip": "19122", "first_name": "Yusuf"},
    ),
    (
        "no closing tag (truncated generation)",
        '<TOOLCALL>[{"name": "get_order_details", "arguments": {"order_id": "#W123"}}]',
        "get_order_details",
        {"order_id": "#W123"},
    ),
    (
        "no tags at all",
        '[{"name": "list_all_product_types", "arguments": {}}]',
        "list_all_product_types",
        {},
    ),
    (
        "bare object rather than a list",
        '<TOOLCALL>{"name": "transfer_to_human_agents", "arguments": {"summary": "x"}}</TOOLCALL>',
        "transfer_to_human_agents",
        {"summary": "x"},
    ),
    (
        "nested array in arguments -- must not terminate the match early",
        '<TOOLCALL>[{"name": "modify_pending_order_items", "arguments": '
        '{"item_ids": ["1", "2"], "new_item_ids": ["3", "4"]}}]</TOOLCALL>',
        "modify_pending_order_items",
        {"item_ids": ["1", "2"], "new_item_ids": ["3", "4"]},
    ),
    (
        "whitespace and newlines inside the block",
        '<TOOLCALL>\n  [{"name": "get_user_details",\n   "arguments": {"user_id": "u_1"}}]\n</TOOLCALL>',
        "get_user_details",
        {"user_id": "u_1"},
    ),
    (
        "`parameters` instead of `arguments`",
        '<TOOLCALL>[{"name": "calculate", "parameters": {"expression": "1+1"}}]</TOOLCALL>',
        "calculate",
        {"expression": "1+1"},
    ),
]

# Inputs that MUST return None -- a wrong name is worse than a detected failure, because
# _close_call turns None into an explicit warning the caller can act on.
REJECT = [
    ("empty", ""),
    ("prose, not a call", "Retail Cancel Transfer"),
    ("truncated mid-JSON", '<TOOLCALL>[{"name": "get_order_det'),
    ("valid JSON but no name field", '<TOOLCALL>[{"arguments": {"a": 1}}]</TOOLCALL>'),
    ("empty list", "<TOOLCALL>[]</TOOLCALL>"),
]


def main() -> int:
    parse = load_parser()
    failures = []

    for label, raw, want_name, want_args in CASES:
        got = parse(raw)
        if got is None:
            failures.append(f"{label}: returned None, expected {want_name!r}")
            continue
        name, args = got
        if name != want_name:
            failures.append(f"{label}: name={name!r}, expected {want_name!r}")
        missing = {k: v for k, v in want_args.items() if args.get(k) != v}
        if missing:
            failures.append(f"{label}: arguments missing/wrong {missing!r} (got {args!r})")
        if not failures or failures[-1].split(":")[0] != label:
            print(f"  ok   {label} -> {name}")

    for label, raw in REJECT:
        got = parse(raw)
        if got is not None:
            failures.append(f"reject/{label}: expected None, got {got!r}")
        else:
            print(f"  ok   reject: {label}")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        print(f"\n{len(failures)} failure(s)")
        return 1
    print(f"all {len(CASES) + len(REJECT)} cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
