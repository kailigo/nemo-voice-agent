#!/usr/bin/env python3
"""
Readout: does mode A reproduce with CLEAN (non-telephony, non-degraded) audio of the same
spelled-out ids? (TAU_VOICE_SFT_RL_PROGRAM.md §13; supersedes the readout-3 text-injection
attempts, which were confounded by inventing a prompt convention the model never trained on --
see mode_a_readout3.py's docstring history.)

This is a direct test of the perception hypothesis instead of an indirect one: real τ-voice
audio is 8kHz telephony-compressed with channel effects (see [[tau-voice-telephony-and-tick-grid]]
memory); if the model gets the id right when the SAME spelled content is delivered as clean
16kHz audio but wrong on the real degraded recording, the failure is audio-quality/channel-driven,
not a fundamental encoder weakness -- different fix again (robustness/augmentation, not encoder
fine-tuning). If it's still wrong on clean audio, that's real evidence of a genuine perception
weakness independent of degradation.

Synthesizes clean audio via Gemini TTS (tau-voice-2's tts_gemini, already integrated this
session) of each receipt's id spelled out letter-by-letter / digit-by-digit, comma-separated,
matching tau2's own voice-guideline dictation convention -- NOT reusing each receipt's original
real dictation style (airline__3 was originally dictated as whole words + digits; here every
receipt is fully spelled character-by-character for methodological uniformity across the set).

Runs BOTH probes on the same clean audio for direct comparison against the existing real-audio
results:
  readout 1 style: real domain policy + tools, free-running -- does it call the right tool with
                    the right id? (real audio result: reproduces the original wrong-copy error)
  readout 2 style: "repeat back what you heard, no tool call" -- did it perceive the id?
                    (real audio result: near-miss, not a clean match)

Usage:
  GOOGLE_APPLICATION_CREDENTIALS=/fsx/home/kai.li/code/test/gcp_credentials.json \
    python scripts/mode_a_readout_clean_audio.py --episode airline__28
  python scripts/mode_a_readout_clean_audio.py   # all four receipts
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/fsx/home/kai.li/code/tau-voice-2/src")

from mode_a_probe import RECEIPTS, normalise_id, edit_distance  # noqa: E402
from mode_a_readout1 import (  # noqa: E402
    TOOL_SCHEMAS_PATH, PRETRAINED_S2S_MODEL,
    format_system_prompt_with_tools, domain_of, find_episode_policy,
    parse_toolcall, find_any_id_like_value,
)
from mode_a_readout2 import READOUT2_PROMPT  # noqa: E402

_DIGIT_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def spell_for_speech(id_str: str) -> str:
    """Render an id as it should be SPOKEN, letter/digit by letter/digit, comma-separated --
    matches data/tau2/user_simulator/simulation_guidelines_voice.md's own convention
    ("Letters: 'J, O, H, N' NOT 'J O H N'"; digits as words). '#' is dropped, never spoken
    (mode_a_probe.py's own note: retail__78's '#' "is not spoken")."""
    tokens = []
    for ch in id_str:
        if ch == "#":
            continue
        elif ch == "_":
            tokens.append("underscore")
        elif ch.isdigit():
            tokens.append(_DIGIT_WORD[ch])
        else:
            tokens.append(ch.upper())
    return ", ".join(tokens)


def synth_clean_audio(text: str, sample_rate_out: int = 16000):
    """Synthesize clean (non-telephony) audio via Gemini TTS, return float32 [-1,1] array."""
    import numpy as np
    from tau2.voice.utils.gemini_utils import tts_gemini

    audio = tts_gemini(text=text)
    pcm = np.frombuffer(audio.data, dtype="<i2").astype("float32") / 32768.0
    src_rate = audio.format.sample_rate
    if src_rate != sample_rate_out:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(src_rate), sample_rate_out)
        pcm = resample_poly(pcm, sample_rate_out // g, int(src_rate) // g).astype("float32")
    return pcm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", action="append")
    ap.add_argument("--ckpt-path", default=None, help="Default: base (stt_extracted_lora), no SFT")
    ap.add_argument("--pad-seconds", type=float, default=10.0)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    receipts = RECEIPTS
    if args.episode:
        want = set(args.episode)
        receipts = [r for r in RECEIPTS if r["episode"] in want]

    tool_schemas = json.loads(TOOL_SCHEMAS_PATH.read_text())

    print("[clean-audio] synthesizing clean audio via Gemini TTS...", flush=True)
    import numpy as np

    plan = []
    for r in receipts:
        domain = domain_of(r["episode"])
        if domain not in tool_schemas:
            print(f"!! {r['episode']}: domain '{domain}' not in tool_schemas.json", file=sys.stderr)
            continue
        spoken_text = spell_for_speech(r["expected"])
        clip = synth_clean_audio(spoken_text)
        pad = np.zeros(int(args.pad_seconds * 16000), dtype="float32")
        clip_padded = np.concatenate([clip, pad])
        policy = find_episode_policy(r["episode"])
        system_prompt_readout1 = format_system_prompt_with_tools(policy, tool_schemas[domain]["tools"])
        plan.append({
            **r, "spoken_text": spoken_text, "clip": clip_padded,
            "system_prompt_readout1": system_prompt_readout1,
        })
        print(f"  {r['episode']:<13} spoken as: {spoken_text!r} ({len(clip)/16000:.1f}s)", flush=True)

    if not plan:
        print("nothing to probe", file=sys.stderr)
        sys.exit(1)

    print(f"\n[clean-audio] building model (ckpt={args.ckpt_path or 'base, no SFT'})...", flush=True)
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
        print(f"[clean-audio] loaded ckpt {args.ckpt_path}", flush=True)

    device = torch.device("cuda")
    model = model.to(dtype=torch.bfloat16, device=device).eval()

    def run_inference(clip, prompt_text):
        input_signal = torch.tensor(clip, dtype=torch.float32, device=device).unsqueeze(0)
        input_signal_lens = torch.tensor([input_signal.shape[1]], device=device)
        tok = model.tokenizer
        prompt_ids = [tok.bos] + tok.text_to_ids(prompt_text) + [tok.eos]
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
        return func_text, agent_text

    results = []
    for p in plan:
        exp_n = normalise_id(p["expected"])

        # Readout-1 style: real domain policy + tools, free-running tool call.
        func_text_1, agent_text_1 = run_inference(p["clip"], p["system_prompt_readout1"])
        predicted_call = parse_toolcall(func_text_1)
        candidate_values = find_any_id_like_value(predicted_call.get("arguments", {})) if predicted_call else []
        matched_expected_1 = any(normalise_id(v) == exp_n for v in candidate_values)
        best_edit_1 = min((edit_distance(normalise_id(v), exp_n) for v in candidate_values), default=None)

        # Readout-2 style: "repeat back what you heard, no tool call".
        _, agent_text_2 = run_inference(p["clip"], READOUT2_PROMPT)
        contains_expected_2 = exp_n in normalise_id(agent_text_2)

        results.append({
            "episode": p["episode"],
            "expected": p["expected"],
            "spoken_text": p["spoken_text"],
            "readout1_predicted_call": predicted_call,
            "readout1_candidate_values": candidate_values,
            "readout1_got_it_right": matched_expected_1,
            "readout1_best_edit_distance": best_edit_1,
            "readout1_func_text_full": func_text_1,
            "readout1_agent_text_preview": agent_text_1[:300],
            "readout2_contains_expected_id": contains_expected_2,
            "readout2_agent_text": agent_text_2[:300],
        })
        print(f"  {p['episode']:<13} expected={p['expected']!r:20s} "
              f"readout1_right={matched_expected_1} readout1_candidates={candidate_values} "
              f"readout2_contains_id={contains_expected_2}", flush=True)

    n_r1 = sum(r["readout1_got_it_right"] for r in results)
    n_r2 = sum(r["readout2_contains_expected_id"] for r in results)
    print(f"\n[clean-audio] readout1-style: {n_r1}/{len(results)} correct with CLEAN audio "
          f"(real degraded audio: reproduces the wrong-copy error, 0/4 correct per readout1.py)")
    print(f"[clean-audio] readout2-style: {n_r2}/{len(results)} correctly repeated the id with "
          f"CLEAN audio (real degraded audio: near-miss, 0/4 exact match per readout2.py)")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"[clean-audio] wrote {args.output_json}")


if __name__ == "__main__":
    main()
