#!/usr/bin/env python3
"""
Readout 1, but with the model's OWN native voice instead of an unrelated external TTS.

2026-09-02 finding: every script in this whole project (all mode_a_readout*.py, the
entire live tau2 batch) built a bare DuplexSTTModel and, for anything needing audible
speech, bolted on an unrelated TTS (SilenceTTS/NeMoTTS/Gemini). The checkpoint we
finetune (stt_extracted_lora, source: "extracted_from_nemotron_voicechat_11b") was
deliberately stripped of the tts_model/DuplexEARTTS weights the *original* release
checkpoint has (635 tts_model.* keys in voicechat-11b/model.safetensors, 0 in ours).
NemotronVoiceChat (nemotron_voicechat.py) is the wrapper that holds both:
self.stt_model = DuplexSTTModel(...), self.tts_model = DuplexEARTTS(...).

This script: build NemotronVoiceChat from the full voicechat-11b checkpoint (real
tts_model), then swap in OUR finetuned DuplexSTTModel (built exactly the way
mode_a_readout1.py does, so the finetuning is genuinely applied) for model.stt_model.
Downstream code (model.offline_inference, model.stt_model.tokenizer, etc.) only ever
references model.stt_model generically, so the swap is a plain Python attribute
assignment -- no checkpoint key-renaming needed.

Verifies two things at once: (1) does the finetuned understanding still work correctly
after the swap (tool-call output should match what bare DuplexSTTModel gives), and
(2) does genuine, non-silent audio come out of model.tts_model this time.

Usage:
  python scripts/mode_a_readout_voicechat.py --episode airline__28
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mode_a_probe import RECEIPTS, find_audio_dir, parse_labels, spoken_match, normalise_id, edit_distance  # noqa: E402
from mode_a_readout1 import (  # noqa: E402
    TOOL_SCHEMAS_PATH, PRETRAINED_S2S_MODEL,
    format_system_prompt_with_tools, domain_of, find_episode_policy,
    extract_user_audio_prefix, parse_toolcall, find_any_id_like_value,
)

FULL_CHECKPOINT_DIR = "/fsx/home/kai.li/data/voicechat/voicechat-11b"


def build_merged_finetuned_state_dict(ckpt_path):
    """Build our finetuned DuplexSTTModel (same config/LoRA-install/checkpoint-load path
    as mode_a_readout1.py), then MERGE the LoRA deltas into the base weights
    (model.llm.merge_and_unload(), standard PEFT API -- LoRA is installed on just
    model.llm, per maybe_install_lora). After merging there is no PEFT wrapper left:
    plain nn.Module, plain key names (llm.layers.0... not
    llm.base_model.model.layers.0...), matching NemotronVoiceChat's own (unwrapped)
    stt_model exactly. Returns a plain state_dict on CPU, no model instance kept
    around -- the caller loads it directly into the already-correctly-constructed
    model.stt_model, so structure/tokenizer/tts-coupling stay identical to the
    proven-working control run; only the weight *values* change."""
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

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[voicechat] loaded finetuned ckpt {ckpt_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    if hasattr(model.llm, "merge_and_unload"):
        print("[voicechat] merging LoRA deltas into base weights (model.llm.merge_and_unload())", flush=True)
        model.llm = model.llm.merge_and_unload()
    else:
        print("[voicechat] model.llm has no merge_and_unload -- LoRA was not installed, nothing to merge", flush=True)

    return {k: v.cpu() for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", action="append")
    ap.add_argument("--ckpt-path", default=None, help="Our finetuned .ckpt. Default: base (stt_extracted_lora), no SFT")
    ap.add_argument("--pad-seconds", type=float, default=8.0)
    ap.add_argument("--output-dir", default="logs/mode_a_readout_voicechat_out")
    ap.add_argument("--no-swap", action="store_true",
                     help="Control test: skip swapping in our finetuned stt_model, use "
                          "NemotronVoiceChat's own unmodified (base, release) stt_model. "
                          "Isolates whether near-silent audio is caused by the swap or "
                          "something else in this pipeline.")
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

    print(f"[voicechat] building NemotronVoiceChat from {FULL_CHECKPOINT_DIR} (real tts_model)...", flush=True)
    import torch
    from nemo.collections.speechlm2.inference.utils.offline_voicechat import build_model

    device = torch.device("cuda")
    model = build_model(FULL_CHECKPOINT_DIR, device=device)
    print("[voicechat] base NemotronVoiceChat ready (stt_model is still the UNFINETUNED base here)", flush=True)

    if args.no_swap:
        print("[voicechat] --no-swap: using NemotronVoiceChat's own unmodified stt_model "
              "(control test, no finetuning, no merge)", flush=True)
    else:
        merged_sd = build_merged_finetuned_state_dict(args.ckpt_path)
        missing, unexpected = model.stt_model.load_state_dict(merged_sd, strict=False)
        print(f"[voicechat] loaded merged finetuned weights directly into NemotronVoiceChat's own "
              f"stt_model (ckpt={args.ckpt_path or 'base, no SFT'}); missing={len(missing)}, "
              f"unexpected={len(unexpected)}; structure/tts-coupling unchanged from the working control run", flush=True)

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for p in plan:
        clip, user_ch, src_rate = extract_user_audio_prefix(
            p["dir"] / "both.wav", p["end"], pad=args.pad_seconds
        )
        input_signal = torch.tensor(clip, dtype=torch.float32, device=device).unsqueeze(0)
        input_signal_lens = torch.tensor([input_signal.shape[1]], device=device)

        tok = model.stt_model.tokenizer
        prompt_ids = [tok.bos] + tok.text_to_ids(p["system_prompt"]) + [tok.eos]
        prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_token_lens = torch.tensor([len(prompt_ids)], device=device)

        with torch.no_grad():
            out = model.offline_inference(
                input_signal=input_signal, input_signal_lens=input_signal_lens,
                prompt_tokens=prompt_tokens, prompt_token_lens=prompt_token_lens,
                decode_audio=True,
            )

        func_tokens = [t for t in out["tokens_function"][0].tolist() if t != model.stt_model.text_pad_id]
        func_text = model.stt_model.tokenizer.ids_to_text(func_tokens)
        text_tokens = [t for t in out["tokens_text"][0].tolist() if t != model.stt_model.text_pad_id]
        agent_text = model.stt_model.tokenizer.ids_to_text(text_tokens)

        predicted_call = parse_toolcall(func_text)
        candidate_values = find_any_id_like_value(predicted_call.get("arguments", {})) if predicted_call else []
        exp_n = normalise_id(p["expected"])
        matched_expected = any(normalise_id(v) == exp_n for v in candidate_values)

        audio = out.get("audio")
        audio_stats = None
        audio_path = None
        wav = None
        if audio is not None:
            wav = audio[0].detach().float().cpu().numpy()
            import numpy as np
            audio_stats = {
                "n_samples": int(wav.shape[0]),
                "max_abs": float(np.abs(wav).max()) if wav.size else 0.0,
                "rms": float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0,
            }

        results.append({
            "episode": p["episode"], "expected": p["expected"],
            "predicted_call": predicted_call, "candidate_values": candidate_values,
            "got_it_right": matched_expected,
            "func_text_full": func_text, "agent_text_preview": agent_text[:300],
            "audio_stats": audio_stats, "audio_path": audio_path,
        })
        print(f"  {p['episode']:<13} expected={p['expected']!r:20s} got_right={matched_expected} "
              f"candidates={candidate_values} audio={audio_stats}", flush=True)

        if wav is not None and wav.size:
            import soundfile as sf
            audio_path = str(out_dir / f"{p['episode']}_agent_voice.wav")
            sf.write(audio_path, wav, int(model.tts_model.target_sample_rate))
            results[-1]["audio_path"] = audio_path
            print(f"    wrote {audio_path}", flush=True)

    n_correct = sum(r["got_it_right"] for r in results)
    n_audible = sum(1 for r in results if r["audio_stats"] and r["audio_stats"]["rms"] > 1e-4)
    print(f"\n[voicechat] {n_correct}/{len(results)} correct tool-call id; "
          f"{n_audible}/{len(results)} produced genuinely audible (non-silent) audio")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[voicechat] wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
