#!/usr/bin/env python3
"""Feed a training cut's audio through SFT-500 (or base) via the streaming inference
path, and at every LM position log:
  - text head's argmax
  - function head's argmax
  - SOTC's rank in the function head's logits
  - function-head logit gap between SOTC and PAD

Purpose: test the "two heads decouple at free-running" claim. If SOTC's rank stays
huge (>>5) across the whole run, the function head never seriously considered
firing. If SOTC is often top-2..5 close-miss, the head recognized the moment but
lost the argmax — a different bug with a different fix.

Bypasses the tau2 harness entirely: no user simulator, no tool server, no
ElevenLabs. Just feed the cut's own 16 kHz recording (matches training's audio
distribution) and observe the head outputs.

Usage:
  python scripts/probe_head_coupling.py --ckpt-path logs/sft_train8_0826/exp/checkpoints/step-500.ckpt
  python scripts/probe_head_coupling.py --no-ckpt   # base weights, for A/B
  python scripts/probe_head_coupling.py --ckpt-path .../step-500.ckpt --telephony-band
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

PRETRAINED = "/fsx/home/kai.li/data/voicechat/stt_extracted_lora"
CUT_SHARD = "/fsx/home/kai.li/code/nemo-voice-agent/data/tau2_training_samples/mix_full86/train"
DEFAULT_CUT_ID = "11_bd8c4d4e"


def load_cut(cut_id):
    from lhotse import CutSet
    cuts = CutSet.from_shar(in_dir=CUT_SHARD).to_eager()
    for c in cuts:
        if c.id == cut_id:
            return c
    raise SystemExit(f"cut {cut_id} not found")


def apply_telephony(audio, sr):
    """Bandpass 300-3400Hz + downsample to 8kHz + upsample back to sr — the
    approximate transformation the live-batch audio undergoes before reaching
    the model at 16kHz."""
    import scipy.signal as sg
    b, a = sg.butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="bandpass")
    audio = sg.filtfilt(b, a, audio).astype(np.float32)
    audio = sg.resample_poly(audio, 8000, sr).astype(np.float32)
    audio = sg.resample_poly(audio, sr, 8000).astype(np.float32)
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", default=None,
                    help="Path to SFT checkpoint .ckpt; omit for base weights only.")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="Explicit: use base weights, no SFT. (Also implied if --ckpt-path omitted.)")
    ap.add_argument("--telephony-band", action="store_true",
                    help="Filter the training audio to 300-3400Hz + 8kHz roundtrip before feeding.")
    ap.add_argument("--cut-id", default=DEFAULT_CUT_ID)
    ap.add_argument("--limit-seconds", type=float, default=180.0,
                    help="Cap audio length to save GPU time; training cuts run 96-513s.")
    ap.add_argument("--chunk-ms", type=int, default=200)
    args = ap.parse_args()

    print(f"[probe] building DuplexSTTModel from pretrained={PRETRAINED}", flush=True)
    with initialize_config_dir(config_dir=str(REPO / "examples/speechlm2/conf/finetune"), version_base=None):
        cfg = compose(config_name="s2s_duplex_stt_11b")
    OmegaConf.update(cfg, "model.pretrained_s2s_model", PRETRAINED)
    OmegaConf.update(cfg, "model.debug_fc", False, force_add=True)
    model_cfg = OmegaConf.to_container(cfg, resolve=True)

    from nemo.collections.speechlm2 import DuplexSTTModel
    model = DuplexSTTModel(model_cfg)

    if args.ckpt_path and not args.no_ckpt:
        print(f"[probe] loading SFT ckpt {args.ckpt_path}", flush=True)
        ck = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        miss, unex = model.load_state_dict(sd, strict=False)
        print(f"[probe]   loaded (missing={len(miss)}, unexpected={len(unex)})", flush=True)
    else:
        print("[probe] using base weights only (no SFT ckpt)", flush=True)

    device = torch.device("cuda")
    model = model.to(dtype=torch.bfloat16, device=device).eval()

    sotc_id, eotc_id, eotr_id = model._get_function_call_special_tokens()
    text_pad_id = model.text_pad_id
    print(f"[probe] SOTC={sotc_id}  EOTC={eotc_id}  EOTR={eotr_id}  TEXT_PAD={text_pad_id}", flush=True)

    cut = load_cut(args.cut_id)
    audio = cut.recording.load_audio()[0].astype(np.float32)
    sr = cut.recording.sampling_rate
    system_prompt = cut.custom["system_prompt"]
    print(f"[probe] cut={cut.id}  dur={audio.shape[0]/sr:.1f}s  sr={sr}Hz  prompt={len(system_prompt)}chars", flush=True)

    if args.telephony_band:
        audio = apply_telephony(audio, sr)
        print("[probe] applied telephony bandpass + 8kHz roundtrip", flush=True)

    if args.limit_seconds and audio.shape[0] > args.limit_seconds * sr:
        audio = audio[: int(args.limit_seconds * sr)]
        print(f"[probe] capped audio to {args.limit_seconds}s", flush=True)

    # Build prompt tokens
    tok = model.tokenizer
    prompt_ids = [tok.bos] + tok.text_to_ids(system_prompt) + [tok.eos]
    prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_token_lens = torch.tensor([len(prompt_ids)], device=device)
    print(f"[probe] prompt token count = {len(prompt_ids)}", flush=True)

    # Hook _step_inference to record logits at each generation step (skip prompt positions).
    logit_capture = []
    orig_step = model._step_inference

    def step_and_capture(t, state, ans, force_bos_positions):
        ans_out = orig_step(t, state, ans, force_bos_positions)
        # Skip if this is a prompt position (nothing generated there)
        is_prompt = state["is_prompt_position_mask"][0, t].item()
        if is_prompt or ans_out is None or "function_logits" not in ans_out:
            return ans_out
        fn_l = ans_out["function_logits"][:, -1, :].float().cpu()[0]  # (V,)
        tx_l = ans_out["text_logits"][:, -1, :].float().cpu()[0]
        fn_sorted = torch.argsort(fn_l, descending=True)
        sotc_rank = int((fn_sorted == sotc_id).nonzero()[0].item())
        logit_capture.append({
            "t": t,
            "text_top1": int(tx_l.argmax().item()),
            "text_top1_is_pad": int(tx_l.argmax().item()) == text_pad_id,
            "fn_top1": int(fn_l.argmax().item()),
            "sotc_rank": sotc_rank,
            "sotc_logit": float(fn_l[sotc_id]),
            "pad_logit": float(fn_l[text_pad_id]),
            "sotc_minus_pad": float(fn_l[sotc_id] - fn_l[text_pad_id]),
        })
        return ans_out

    model._step_inference = step_and_capture

    from nemo.collections.speechlm2.models.streaming_fc_session import StreamingFCSession
    session = StreamingFCSession(model, prompt_tokens, prompt_token_lens,
                                 max_audio_seconds=max(300.0, args.limit_seconds + 5))
    session.start()

    chunk_samples = int(args.chunk_ms / 1000.0 * sr)
    n_chunks = audio.shape[0] // chunk_samples
    print(f"[probe] feeding {n_chunks} × {args.chunk_ms}ms chunks (chunk_samples={chunk_samples})", flush=True)

    n_tool_calls = 0
    for i in range(n_chunks):
        chunk = audio[i * chunk_samples:(i + 1) * chunk_samples]
        samples = torch.from_numpy(chunk.copy())
        try:
            out = session.push_audio(samples)
        except Exception as e:
            print(f"[probe] push_audio error at chunk {i}: {e}", flush=True)
            break
        if out.tool_calls:
            for tc in out.tool_calls:
                print(f"[probe]  TOOL CALL at LM position {tc.lm_position}: name={tc.name!r} args={tc.arguments}", flush=True)
                n_tool_calls += 1
        if i > 0 and i % 200 == 0:
            print(f"[probe]  progress: {i}/{n_chunks} chunks, {len(logit_capture)} positions captured, {n_tool_calls} calls fired", flush=True)

    # ---- Report ----
    n_pos = len(logit_capture)
    ranks = np.array([c["sotc_rank"] for c in logit_capture])
    n_sotc_top1 = int((ranks == 0).sum())
    n_sotc_top5 = int((ranks < 5).sum())
    n_sotc_top20 = int((ranks < 20).sum())
    n_sotc_top100 = int((ranks < 100).sum())
    n_text_pad = sum(1 for c in logit_capture if c["text_top1_is_pad"])

    print(f"\n[probe] ============ REPORT ============")
    print(f"[probe] Free-run LM positions logged: {n_pos}")
    print(f"[probe] Tool calls actually fired:    {n_tool_calls}")
    print(f"[probe] Text head → PAD at {n_text_pad}/{n_pos} ({100*n_text_pad/n_pos:.1f}%) positions")
    print(f"[probe] Function-head SOTC rank distribution:")
    print(f"[probe]   top-1  : {n_sotc_top1}/{n_pos}  ({100*n_sotc_top1/n_pos:.2f}%)")
    print(f"[probe]   top-5  : {n_sotc_top5}/{n_pos}  ({100*n_sotc_top5/n_pos:.2f}%)")
    print(f"[probe]   top-20 : {n_sotc_top20}/{n_pos} ({100*n_sotc_top20/n_pos:.2f}%)")
    print(f"[probe]   top-100: {n_sotc_top100}/{n_pos} ({100*n_sotc_top100/n_pos:.2f}%)")

    gaps = np.array([c["sotc_minus_pad"] for c in logit_capture])
    print(f"[probe] SOTC minus PAD logit (positive = SOTC winning over PAD):")
    print(f"[probe]   min={gaps.min():.2f}  p50={np.median(gaps):.2f}  p95={np.percentile(gaps,95):.2f}  max={gaps.max():.2f}")

    # Top 20 "closest-miss" positions where SOTC was highest-ranked but still not top-1
    close = sorted([c for c in logit_capture if 0 < c["sotc_rank"] < 50],
                   key=lambda c: c["sotc_rank"])
    print(f"\n[probe] Top-20 closest-miss positions (SOTC rank ≥ 1 but small):")
    for c in close[:20]:
        print(f"  t={c['t']:>5}  text_top1={c['text_top1']:>6} (pad={c['text_top1_is_pad']})  "
              f"fn_top1={c['fn_top1']:>6}  sotc_rank={c['sotc_rank']:>3}  "
              f"sotc_logit={c['sotc_logit']:6.2f}  gap_to_pad={c['sotc_minus_pad']:6.2f}")

    # Also: any positions where text=PAD AND SOTC was in top-5 (this is the head-alignment moment)
    aligned = [c for c in logit_capture if c["text_top1_is_pad"] and c["sotc_rank"] < 5]
    print(f"\n[probe] Aligned moments (text→PAD AND SOTC in function top-5): {len(aligned)}")
    for c in aligned[:20]:
        print(f"  t={c['t']:>5}  fn_top1={c['fn_top1']:>6}  sotc_rank={c['sotc_rank']}  sotc_logit={c['sotc_logit']:.2f}  gap_to_pad={c['sotc_minus_pad']:.2f}")


if __name__ == "__main__":
    main()
