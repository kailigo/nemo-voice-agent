#!/usr/bin/env python3
"""
Readout 3 (NEMO_FAILURE_MODES.md §7 / TAU_VOICE_SFT_RL_PROGRAM.md §13): with the id supplied
as literal text and NO audio at all, can the model copy it verbatim into a tool-call
argument?

This is the rung readouts 1-2 never reached. Readout 1 (real audio, unchanged domain prompt)
reproduces mode A but is confounded -- a wrong copy could be an ASR/perception error (the
model never correctly heard "S, I, 5, U, K, W") or a copy-into-slot error (it heard correctly
but wrote the wrong thing down anyway). Readout 2 ("repeat back what you heard") was supposed
to separate them but landed on a near-miss, not a clean answer.

Readout 3 removes the audio/ASR path from the experiment entirely. Two earlier attempts at this
were confounded and are recorded in `spell_out()`'s docstring below -- both tried to inject the
id via a natural-language instruction appended to the system prompt, and both failed for reasons
unrelated to copy-fidelity (the model correctly rejected a comma-spelled id as malformed; then,
with a compact id, it ignored the injected instruction entirely and fell into a generic
Mode-H-style refusal, because a hand-written instruction glued onto the system prompt is not a
format the model was ever trained to treat as a live directive to act).

This version instead embeds a FAKE PRECEDING EXCHANGE in the model's own native
<TOOLCALL>/<TOOL_RESPONSE> format (exactly how episodes_to_nemotron_training.py represents a
completed lookup) with the correct id already used once, then asks the model to repeat that same
call with the same argument. This tests in-context copying from the model's own prior turn using
its actual trained format, not audio, not ASR, not an invented prompt convention.

  * still wrong here      -> copy-into-slot is broken independent of speech; fix is targeted
                              synthetic SFT drills (spell an id, copy it into a tool call),
                              not an ASR/encoder change.
  * correct here          -> the LLM's copy mechanism is fine; readout 1's error must be
                              upstream, in the speech encoder (nemotron-speech-streaming-en-0.6b).
                              Fix targets the encoder, not the LLM.

Usage:
  python scripts/mode_a_readout3.py --episode airline__28
  python scripts/mode_a_readout3.py   # all four receipts
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mode_a_probe import RECEIPTS, normalise_id, edit_distance  # noqa: E402
from mode_a_readout1 import (  # noqa: E402
    TOOL_SCHEMAS_PATH, PRETRAINED_S2S_MODEL,
    format_system_prompt_with_tools, domain_of, find_episode_policy,
    parse_toolcall, find_any_id_like_value,
)

# (lookup_tool, lookup_arg, instruction, companion_exchange_or_None). Third attempt asked the
# model to take a genuine next action (cancel) instead of repeating the lookup -- this worked
# cleanly for retail__78 (correctly copied the id into cancel_pending_order) but not for the
# airline receipts, which kept asking for a user_id. Airline policy requires BOTH user_id and
# reservation_id verified before acting on a reservation; the fake exchange only supplied one.
# `companion_exchange` prepends a second fake lookup (placeholder value, we don't care what it
# is) so both required fields are satisfied in-context before the cancel instruction, testing
# whether that was really the blocker.
LOOKUP_TOOL = {
    "airline__28": ("get_reservation_details", "reservation_id",
                     "The customer would now like to cancel this reservation. Cancel it now.",
                     ("get_user_details", "user_id", "placeholder_user_0000")),
    "airline__3": ("get_user_details", "user_id",
                    "Please repeat the exact same tool call again, with the exact same argument "
                    "value, to reconfirm.",
                    None),
    "airline__40": ("get_reservation_details", "reservation_id",
                     "The customer would now like to cancel this reservation. Cancel it now.",
                     ("get_user_details", "user_id", "placeholder_user_0000")),
    "retail__78": ("get_order_details", "order_id",
                    "The customer would now like to cancel this order. Cancel it now, using "
                    "\"customer request\" as the reason.",
                    None),
}

COMPANION_EXCHANGE = (
    "<TOOLCALL>[{{\"name\":\"{tool}\",\"arguments\":{{\"{arg}\":\"{value}\"}}}}]</TOOLCALL>\n"
    "<TOOL_RESPONSE>[{{\"status\":\"success\"}}]</TOOL_RESPONSE>\n"
)

READOUT3_EXCHANGE = (
    "\n\n{companion}<TOOLCALL>[{{\"name\":\"{lookup_tool}\",\"arguments\":{{\"{lookup_arg}\":\"{value}\"}}}}]"
    "</TOOLCALL>\n<TOOL_RESPONSE>[{{\"status\":\"success\"}}]</TOOL_RESPONSE>\n\n{instruction}"
)


def spell_out(id_str: str) -> str:
    """Return the id unchanged, in its already-compact form (not comma-spelled).

    Two earlier attempts spelled the id out comma-by-comma ("S, I, 5, U, K, W") or appended a
    natural-language instruction with the compact id -- see the module docstring for why both
    were confounded (0/4 attempted a call either way, for reasons unrelated to copy-fidelity).
    This version doesn't call spell_out() on the way into the prompt at all; kept only so the
    results JSON still records the exact string tested, unchanged, for comparability with the
    earlier two JSON files."""
    return id_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", action="append")
    ap.add_argument("--ckpt-path", default=None, help="Default: base (stt_extracted_lora), no SFT")
    ap.add_argument("--silence-seconds", type=float, default=15.0)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    receipts = RECEIPTS
    if args.episode:
        want = set(args.episode)
        receipts = [r for r in RECEIPTS if r["episode"] in want]

    tool_schemas = json.loads(TOOL_SCHEMAS_PATH.read_text())

    plan = []
    for r in receipts:
        domain = domain_of(r["episode"])
        if domain not in tool_schemas:
            print(f"!! {r['episode']}: domain '{domain}' not in tool_schemas.json", file=sys.stderr)
            continue
        if r["episode"] not in LOOKUP_TOOL:
            print(f"!! {r['episode']}: no LOOKUP_TOOL mapping", file=sys.stderr)
            continue
        lookup_tool, lookup_arg, instruction, companion = LOOKUP_TOOL[r["episode"]]
        policy = find_episode_policy(r["episode"])
        base_prompt = format_system_prompt_with_tools(policy, tool_schemas[domain]["tools"])
        companion_str = ""
        if companion is not None:
            c_tool, c_arg, c_value = companion
            companion_str = COMPANION_EXCHANGE.format(tool=c_tool, arg=c_arg, value=c_value)
        system_prompt = base_prompt + READOUT3_EXCHANGE.format(
            companion=companion_str, lookup_tool=lookup_tool, lookup_arg=lookup_arg,
            instruction=instruction, value=r["expected"],
        )
        plan.append({**r, "system_prompt": system_prompt})

    if not plan:
        print("nothing to probe", file=sys.stderr)
        sys.exit(1)

    print(f"[readout3] building model (ckpt={args.ckpt_path or 'base, no SFT'})...", flush=True)
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    from hydra import initialize_config_dir, compose
    from nemo.collections.speechlm2 import DuplexSTTModel

    with initialize_config_dir(config_dir=str(REPO / "examples/speechlm2/conf/finetune"), version_base=None):
        cfg = compose(config_name="s2s_duplex_stt_11b")
    OmegaConf.update(cfg, "model.pretrained_s2s_model", PRETRAINED_S2S_MODEL)
    OmegaConf.update(cfg, "model.debug_fc", False, force_add=True)
    model_cfg = OmegaConf.to_container(cfg, resolve=True)
    model = DuplexSTTModel(model_cfg)

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        print(f"[readout3] loaded ckpt {args.ckpt_path}", flush=True)

    device = torch.device("cuda")
    model = model.to(dtype=torch.bfloat16, device=device).eval()

    results = []
    for p in plan:
        silence = np.zeros(int(args.silence_seconds * 16000), dtype="float32")
        input_signal = torch.tensor(silence, dtype=torch.float32, device=device).unsqueeze(0)
        input_signal_lens = torch.tensor([input_signal.shape[1]], device=device)

        tok = model.tokenizer
        prompt_ids = [tok.bos] + tok.text_to_ids(p["system_prompt"]) + [tok.eos]
        prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_token_lens = torch.tensor([len(prompt_ids)], device=device)

        with torch.no_grad():
            out = model.offline_inference(
                input_signal, input_signal_lens,
                prompt_tokens=prompt_tokens, prompt_token_lens=prompt_token_lens,
            )

        func_tokens = [t for t in out["tokens_function"][0].tolist() if t != model.text_pad_id]
        func_text = model.tokenizer.ids_to_text(func_tokens)
        text_tokens = [t for t in out["tokens_text"][0].tolist() if t != model.text_pad_id]
        agent_text = model.tokenizer.ids_to_text(text_tokens)

        predicted_call = parse_toolcall(func_text)
        candidate_values = find_any_id_like_value(predicted_call.get("arguments", {})) if predicted_call else []

        exp_n = normalise_id(p["expected"])
        matched_expected = any(normalise_id(v) == exp_n for v in candidate_values)
        best_edit = min((edit_distance(normalise_id(v), exp_n) for v in candidate_values), default=None)

        results.append({
            "episode": p["episode"],
            "expected": p["expected"],
            "spelled_in_prompt": spell_out(p["expected"]),
            "predicted_call": predicted_call,
            "candidate_values": candidate_values,
            "got_it_right": matched_expected,
            "best_edit_distance_to_expected": best_edit,
            "func_text_full": func_text,
            "agent_text_preview": agent_text[:300],
        })
        print(f"  {p['episode']:<13} expected={p['expected']!r:20s} "
              f"got_right={matched_expected} candidates={candidate_values}", flush=True)

    n_correct = sum(r["got_it_right"] for r in results)
    print(f"\n[readout3] {n_correct}/{len(results)} correctly copied the id from clean text "
          f"(no audio at all).")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"[readout3] wrote {args.output_json}")


if __name__ == "__main__":
    main()
