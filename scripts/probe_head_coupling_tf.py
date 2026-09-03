#!/usr/bin/env python3
"""Teacher-forced counterpart to probe_head_coupling.py.

Runs the model in *training-style* forward pass: ground-truth text-channel and
function-channel tokens are fed as inputs, and we measure what the function
head predicts at each labeled SOTC position. This is the decisive test:

- If SOTC is top-1 at all N scripted positions under teacher-forcing → the
  training genuinely put SOTC-firing into the head; the free-running gap is
  the model conditioning on its own generated prior context instead of the
  ground-truth prior.
- If SOTC is NOT top-1 at every position even under teacher-forcing → the
  training metrics (`function_sotc_acc=1.0`) are misleading and the head is
  not as well-fit as claimed.

Usage:
  python scripts/probe_head_coupling_tf.py --ckpt-path .../step-500.ckpt
  python scripts/probe_head_coupling_tf.py --no-ckpt
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

PRETRAINED = "/fsx/home/kai.li/data/voicechat/stt_extracted_lora"
CUT_SHARD = "/fsx/home/kai.li/code/nemo-voice-agent/data/tau2_training_samples/mix_full86/train"
DEFAULT_CUT_ID = "11_bd8c4d4e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--cut-id", default=DEFAULT_CUT_ID)
    args = ap.parse_args()

    print(f"[tf-probe] building model", flush=True)
    with initialize_config_dir(config_dir=str(REPO / "examples/speechlm2/conf/finetune"), version_base=None):
        cfg = compose(config_name="s2s_duplex_stt_11b")
    OmegaConf.update(cfg, "model.pretrained_s2s_model", PRETRAINED)
    OmegaConf.update(cfg, "model.debug_fc", False, force_add=True)

    # Reuse the eval-forward path already validated by eval_forward_id_copy.py
    from eval_forward import build_model_and_dataset, load_ckpt_into_model
    from lhotse import CutSet

    val_dir = CUT_SHARD  # use train shard as our "eval" set to grab a specific cut
    model, dataset = build_model_and_dataset(cfg, val_dir)

    if args.ckpt_path and not args.no_ckpt:
        print(f"[tf-probe] loading SFT ckpt {args.ckpt_path}", flush=True)
        info = load_ckpt_into_model(model, args.ckpt_path, strict=False)
        print(f"[tf-probe]   loaded (missing={info['missing']}, unexpected={info['unexpected']})", flush=True)
    else:
        print("[tf-probe] using base weights only", flush=True)

    device = torch.device("cuda")
    model = model.to(dtype=torch.bfloat16, device=device).eval()

    sotc_id, eotc_id, eotr_id = model._get_function_call_special_tokens()
    text_pad_id = model.text_pad_id
    print(f"[tf-probe] SOTC={sotc_id} EOTC={eotc_id} EOTR={eotr_id} PAD={text_pad_id}", flush=True)

    # Find the target cut
    cuts = CutSet.from_shar(in_dir=val_dir).to_eager()
    target = None
    for c in cuts:
        if c.id == args.cut_id:
            target = c; break
    if target is None:
        raise SystemExit(f"cut {args.cut_id} not found")
    print(f"[tf-probe] cut={target.id}  dur={target.duration:.1f}s", flush=True)

    # Build a one-cut batch through the same pipeline training uses
    single = CutSet.from_cuts([target])
    batch = dataset[single]

    def to_dev(x, device):
        if torch.is_tensor(x): return x.to(device, non_blocking=True)
        if isinstance(x, dict): return {k: to_dev(v, device) for k, v in x.items()}
        if isinstance(x, list): return [to_dev(v, device) for v in x]
        return x

    batch = to_dev(batch, device)

    with torch.no_grad():
        inputs = model.prepare_inputs(batch["audio_data"], include_asr_loss=False)
        out = model(inputs["input_embeds"], compute_asr=inputs["compute_asr"])

    fn_logits = out["function_logits"]  # (B, T, V)
    fn_labels = inputs["function_labels"]  # (B, T)
    tx_logits = out["text_logits"]  # (B, T, V)

    B, T, V = fn_logits.shape
    print(f"[tf-probe] logits shape: B={B} T={T} V={V}", flush=True)

    # Positions where the label is SOTC (i.e. the scripted tool-call start positions)
    sotc_positions = (fn_labels[0] == sotc_id).nonzero(as_tuple=True)[0].tolist()
    eotc_positions = (fn_labels[0] == eotc_id).nonzero(as_tuple=True)[0].tolist()
    print(f"[tf-probe] cut has {len(sotc_positions)} scripted SOTC positions, "
          f"{len(eotc_positions)} EOTC positions", flush=True)

    # At each scripted SOTC position, what does the function head actually predict?
    print(f"\n[tf-probe] Function head at each scripted SOTC position (t = position in expanded sequence):")
    print(f"{'t':>7}  {'label':>6}  {'pred':>6}  {'sotc_rank':>10}  {'sotc_logit':>11}  {'pad_logit':>10}  {'sotc-pad':>10}")
    n_correct = 0
    for pos in sotc_positions:
        fn_l = fn_logits[0, pos].float()
        fn_sorted = torch.argsort(fn_l, descending=True)
        sotc_rank = int((fn_sorted == sotc_id).nonzero()[0].item())
        pred = int(fn_l.argmax().item())
        label = int(fn_labels[0, pos].item())
        is_correct = pred == sotc_id
        if is_correct: n_correct += 1
        print(f"{pos:>7d}  {label:>6d}  {pred:>6d}  {sotc_rank:>10d}  "
              f"{float(fn_l[sotc_id]):>11.3f}  {float(fn_l[text_pad_id]):>10.3f}  "
              f"{float(fn_l[sotc_id] - fn_l[text_pad_id]):>10.3f}  "
              f"{'OK' if is_correct else '**MISS**'}")
    if sotc_positions:
        print(f"\n[tf-probe] Teacher-forced SOTC prediction accuracy: {n_correct}/{len(sotc_positions)} "
              f"({100*n_correct/len(sotc_positions):.1f}%)")

    # Also: for a small sample of non-SOTC positions with label=PAD, is SOTC in top-5 there?
    # This tests whether the head is trigger-happy at non-tool-call positions.
    pad_positions = (fn_labels[0] == text_pad_id).nonzero(as_tuple=True)[0].tolist()
    import random
    random.seed(0)
    sample = random.sample(pad_positions, min(15, len(pad_positions)))
    print(f"\n[tf-probe] Function head at 15 random non-SOTC (label=PAD) positions:")
    print(f"{'t':>7}  {'pred':>6}  {'sotc_rank':>10}  {'sotc_logit':>11}  {'sotc-pad':>10}")
    for pos in sample:
        fn_l = fn_logits[0, pos].float()
        fn_sorted = torch.argsort(fn_l, descending=True)
        sotc_rank = int((fn_sorted == sotc_id).nonzero()[0].item())
        pred = int(fn_l.argmax().item())
        print(f"{pos:>7d}  {pred:>6d}  {sotc_rank:>10d}  "
              f"{float(fn_l[sotc_id]):>11.3f}  "
              f"{float(fn_l[sotc_id] - fn_l[text_pad_id]):>10.3f}")


if __name__ == "__main__":
    main()
