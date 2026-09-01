#!/usr/bin/env python3
"""
Readout 1 (NEMO_FAILURE_MODES.md §7): does mode A reproduce when OUR model, not the
container, processes the exact real τ-voice stimulus with the unchanged domain prompt?

This is "the gate, not a formality" (mode_a_probe.py docstring): sampling is greedy, so if
our harness is a faithful stand-in for whatever produced SIN-555/JFK59001/etc., it MUST
reproduce the same error under the same input. If it doesn't reproduce, readouts 2/3
(encoder-vs-copy) are uninterpretable -- our harness would not be testing the same thing.

Stimulus: the same real `both.wav` + Audacity label tracks used by readout 0
(scripts/mode_a_probe.py), reused here via its RECEIPTS/find_audio_dir/parse_labels. Not a
new recording, no ElevenLabs, no synthesis. The user-channel audio is truncated to the end
of the id-spelling turn (+ trailing silence for the model to act in) and fed through
DuplexSTTModel.offline_inference() with no forced function-call positions -- free
generation, same mechanism as scripts/eval_first_call_argument_accuracy.py.

System prompt: reconstructed byte-identically to how episodes_to_nemotron_training.py
built it for training -- policy text from the episode's own recorded simulation JSON
(`policy` field) + <AVAILABLE_TOOLS> from tau-voice-2/data/tool_schemas.json, via the same
format_system_prompt_with_tools() function (reimplemented verbatim here, not imported --
episodes_to_nemotron_training.py has heavier deps this script doesn't need).

Usage:
  python scripts/mode_a_readout1.py --episode airline__28
  python scripts/mode_a_readout1.py   # all four receipts
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mode_a_probe import (  # noqa: E402  (reused as-is, not reimplemented)
    RECEIPTS, find_audio_dir, parse_labels, spoken_match, normalise_id, edit_distance,
)

TAU2 = Path("/fsx/home/kai.li/code/tau-voice-2")
TOOL_SCHEMAS_PATH = TAU2 / "data/tool_schemas.json"
PRETRAINED_S2S_MODEL = "/fsx/home/kai.li/data/voicechat/stt_extracted_lora"


def format_system_prompt_with_tools(policy: str, tool_schemas: list) -> str:
    """Byte-identical to episodes_to_nemotron_training.py::format_system_prompt_with_tools."""
    tools_json = json.dumps(tool_schemas, separators=(",", ":"))
    return f"{policy}\n\n<AVAILABLE_TOOLS>{tools_json}</AVAILABLE_TOOLS>"


def domain_of(episode: str) -> str:
    return episode.split("__")[0]


def find_episode_policy(episode: str) -> str:
    sims = sorted((TAU2 / "data/simulations/stage2_subset_0821" / episode / "simulations").glob("*.json"))
    if not sims:
        raise FileNotFoundError(f"no simulation json for {episode}")
    d = json.loads(sims[0].read_text())
    policy = d.get("policy")
    if not policy:
        raise ValueError(f"{sims[0]} has no 'policy' field")
    return policy


def extract_user_audio_prefix(both_wav: Path, end: float, pad: float = 8.0):
    """User-channel audio from 0 to `end`, resampled to 16kHz, plus `pad` seconds of trailing
    silence. Channel is detected the same way as extract_user_clip (RMS ratio vs. the
    assistant's own labelled span), not assumed -- see mode_a_probe.py's own comment on why."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(str(both_wav), always_2d=True)
    n_ch = audio.shape[1]

    if n_ch == 1:
        user_ch = 0
    else:
        asst = parse_labels(both_wav.parent / "assistant_labels.txt")
        e0 = int(end * rate)
        # Use a short window right before `end` as the "user span" for channel detection.
        u0 = max(0, e0 - int(2.0 * rate))
        user_rms = [float(np.sqrt(np.mean(audio[u0:e0, c] ** 2))) for c in range(n_ch)]
        if asst:
            a0, a1 = int(asst[0][0] * rate), int(asst[0][1] * rate)
            asst_rms = [float(np.sqrt(np.mean(audio[a0:a1, c] ** 2))) for c in range(n_ch)]
            ratios = [user_rms[c] / (asst_rms[c] + 1e-9) for c in range(n_ch)]
            user_ch = int(np.argmax(ratios))
        else:
            user_ch = int(np.argmax(user_rms))

    e = min(len(audio), int(end * rate))
    clip = audio[:e, user_ch].astype("float32")

    if rate != 16000:
        from math import gcd
        g = gcd(int(rate), 16000)
        clip = resample_poly(clip, 16000 // g, int(rate) // g).astype("float32")

    pad_samples = np.zeros(int(pad * 16000), dtype="float32")
    return np.concatenate([clip, pad_samples]), user_ch, rate


def parse_toolcall(text: str):
    m = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", text, re.S)
    if not m:
        return None
    try:
        calls = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return calls[0] if calls else None


def find_any_id_like_value(obj):
    """Walk a predicted call's arguments and return every string value -- we don't know
    which argument slot the id landed in a priori (that IS one of the things being tested:
    mode B, wrong slot, is a live possibility here too)."""
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(find_any_id_like_value(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(find_any_id_like_value(v))
    elif isinstance(obj, str):
        out.append(obj)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", action="append")
    ap.add_argument("--ckpt-path", default=None, help="Default: base (stt_extracted_lora), no SFT")
    ap.add_argument("--pad-seconds", type=float, default=8.0)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    receipts = RECEIPTS
    if args.episode:
        want = set(args.episode)
        receipts = [r for r in RECEIPTS if r["episode"] in want]

    tool_schemas = json.loads(TOOL_SCHEMAS_PATH.read_text())

    plan = []
    for r in receipts:
        d = find_audio_dir(r["episode"])
        if d is None:
            print(f"!! {r['episode']}: no both.wav found", file=sys.stderr)
            continue
        labels = parse_labels(d / "user_labels.txt")
        hits = [L for L in labels if spoken_match(L[2], r["spoken"])]
        if not hits:
            print(f"!! {r['episode']}: id's spoken form matches no user label", file=sys.stderr)
            continue
        best = hits[0]
        domain = domain_of(r["episode"])
        if domain not in tool_schemas:
            print(f"!! {r['episode']}: domain '{domain}' not in tool_schemas.json", file=sys.stderr)
            continue
        policy = find_episode_policy(r["episode"])
        system_prompt = format_system_prompt_with_tools(policy, tool_schemas[domain]["tools"])
        plan.append({**r, "dir": d, "end": best[1], "said": best[2], "system_prompt": system_prompt})

    if not plan:
        print("nothing to probe", file=sys.stderr)
        sys.exit(1)

    print(f"[readout1] building model (ckpt={args.ckpt_path or 'base, no SFT'})...", flush=True)
    import torch
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
        print(f"[readout1] loaded ckpt {args.ckpt_path}", flush=True)

    device = torch.device("cuda")
    model = model.to(dtype=torch.bfloat16, device=device).eval()

    results = []
    for p in plan:
        clip, user_ch, src_rate = extract_user_audio_prefix(
            p["dir"] / "both.wav", p["end"], pad=args.pad_seconds
        )
        input_signal = torch.tensor(clip, dtype=torch.float32, device=device).unsqueeze(0)
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

        exp_n, sent_n = normalise_id(p["expected"]), normalise_id(p["sent"])
        matched_expected = any(normalise_id(v) == exp_n for v in candidate_values)
        matched_sent = any(normalise_id(v) == sent_n for v in candidate_values)
        best_edit = min((edit_distance(normalise_id(v), exp_n) for v in candidate_values), default=None)

        results.append({
            "episode": p["episode"],
            "expected": p["expected"],
            "original_model_sent": p["sent"],
            "user_channel": user_ch,
            "src_rate": src_rate,
            "predicted_call": predicted_call,
            "candidate_values": candidate_values,
            "reproduced_original_error": matched_sent,
            "got_it_right": matched_expected,
            "best_edit_distance_to_expected": best_edit,
            "func_text_full": func_text,
            "agent_text_preview": agent_text[:300],
        })
        print(f"  {p['episode']:<13} expected={p['expected']!r:20s} "
              f"got_right={matched_expected} reproduced_original_error={matched_sent} "
              f"candidates={candidate_values}", flush=True)

    n_reproduced = sum(r["reproduced_original_error"] for r in results)
    n_correct = sum(r["got_it_right"] for r in results)
    print(f"\n[readout1] {n_reproduced}/{len(results)} reproduced the original error; "
          f"{n_correct}/{len(results)} got the id right.")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"[readout1] wrote {args.output_json}")


if __name__ == "__main__":
    main()
