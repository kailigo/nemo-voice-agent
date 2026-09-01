#!/usr/bin/env python3
"""
Readout 2 (NEMO_FAILURE_MODES.md §7): with the domain prompt replaced by a short
"repeat the id back, call no tools" instruction, does the model correctly perceive the
spelled-out id? Separates encoder-perception from copy-into-argument-slot.

Also serves as a check on whether mode H's refusal (found while running readout 1) is
specific to the long real domain-policy prompt, or something broader -- this prompt is
short, non-domain, and explicitly tells the model not to call tools at all.

Reuses the exact same stimulus (real both.wav + labels) as readout 0/1.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mode_a_probe import RECEIPTS, find_audio_dir, parse_labels, spoken_match, normalise_id, edit_distance  # noqa: E402
from mode_a_readout1 import extract_user_audio_prefix, PRETRAINED_S2S_MODEL  # noqa: E402

READOUT2_PROMPT = (
    "You are a transcription checkpoint. The user will say an identifier out loud. "
    "Do not call any tool. Simply repeat back, in text, exactly what you heard them say."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", action="append")
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--pad-seconds", type=float, default=8.0)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    receipts = RECEIPTS
    if args.episode:
        want = set(args.episode)
        receipts = [r for r in RECEIPTS if r["episode"] in want]

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
        plan.append({**r, "dir": d, "end": best[1], "said": best[2]})

    if not plan:
        print("nothing to probe", file=sys.stderr)
        sys.exit(1)

    print(f"[readout2] building model (ckpt={args.ckpt_path or 'base, no SFT'})...", flush=True)
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
        print(f"[readout2] loaded ckpt {args.ckpt_path}", flush=True)

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
        prompt_ids = [tok.bos] + tok.text_to_ids(READOUT2_PROMPT) + [tok.eos]
        prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_token_lens = torch.tensor([len(prompt_ids)], device=device)

        with torch.no_grad():
            out = model.offline_inference(
                input_signal, input_signal_lens,
                prompt_tokens=prompt_tokens, prompt_token_lens=prompt_token_lens,
            )

        text_tokens = [t for t in out["tokens_text"][0].tolist() if t != model.text_pad_id]
        agent_text = model.tokenizer.ids_to_text(text_tokens)

        exp_n = normalise_id(p["expected"])
        got_it = exp_n in normalise_id(agent_text)
        refused = "unable to assist" in agent_text.lower() or "i do not have access" in agent_text.lower()

        results.append({
            "episode": p["episode"],
            "expected": p["expected"],
            "agent_text": agent_text,
            "contains_expected_id": got_it,
            "mode_h_refusal": refused,
        })
        print(f"  {p['episode']:<13} expected={p['expected']!r:20s} contains_expected={got_it} "
              f"mode_h_refusal={refused}", flush=True)
        print(f"    agent_text: {agent_text[:200]!r}", flush=True)

    n_got = sum(r["contains_expected_id"] for r in results)
    n_refused = sum(r["mode_h_refusal"] for r in results)
    print(f"\n[readout2] {n_got}/{len(results)} correctly repeated the id; "
          f"{n_refused}/{len(results)} still showed a mode-H-style refusal.")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"[readout2] wrote {args.output_json}")


if __name__ == "__main__":
    main()
