#!/usr/bin/env python3
"""
Value-level check: does the training setup actually hold a working pretrained model?

verify_checkpoint_load.py proves tensors land in the right slots by name and shape. It
cannot prove they are the right NUMBERS. This script closes that gap the only way that
really counts: build the model exactly as training will (same finetune yaml, same
pretrained_s2s_model, LoRA installed), feed it real speech, and read the text channel.

Intelligible, on-topic English means the ~10.1B pretrained params are live. Random weights
produce token salad (e.g. "annte ابت ابت wrestlingérable"), which is unmistakable.

Note this is a duplex agent, not a transcriber: with predict_user_text=false the text
channel emits the ASSISTANT'S REPLY to the spoken input, not a transcript of it.

Usage:
  python scripts/smoke_test_inference.py \
      --checkpoint /fsx/home/kai.li/data/voicechat/stt_extracted_lora \
      --wav examples/speechlm2/sample_audio/sample_general.wav

Compare against the untrained baseline to see what failure looks like:
  python scripts/smoke_test_inference.py --checkpoint ... --wav ... --random-weights
"""

import argparse
import os
import re
import sys

import torch
from omegaconf import OmegaConf


def build_fc_system_prompt():
    """Render the tool-declaring system prompt from the NeMo repo's own FC assets.

    A plain system prompt cannot exercise the function head: with no <AVAILABLE_TOOLS>
    section the model correctly declines ("I cannot generate random numbers"), which looks
    like a broken function_head but is right behaviour. We reuse template.jinja and the
    DEFAULT_TOOLS/DEFAULT_SYSTEM_MESSAGE from offline_voicechat_fc_infer.py rather than
    copying them, so this stays in step with the released model's documented setup.
    """
    import importlib.util

    import nemo
    from nemo.collections.speechlm2.inference.utils.offline_voicechat import render_fc_system_prompt

    # nemo is installed editable from the Speech clone, so its parent is the repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(nemo.__file__)))
    fc_script = os.path.join(repo_root, "examples", "speechlm2", "offline_voicechat_fc_infer.py")
    template = os.path.join(repo_root, "examples", "speechlm2", "function_calling", "template.jinja")
    for path in (fc_script, template):
        if not os.path.exists(path):
            raise FileNotFoundError(f"FC asset not found: {path}")

    spec = importlib.util.spec_from_file_location("_fc_infer", fc_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # main() is guarded by __name__, so nothing runs
    tool_names = [t["function"]["name"] for t in mod.DEFAULT_TOOLS]
    print(f"  FC system prompt: declaring tools {tool_names}")
    return render_fc_system_prompt(template, mod.DEFAULT_SYSTEM_MESSAGE, mod.DEFAULT_TOOLS)


def report_function_calls(model, result):
    """Decode the function channel, mirroring pass 1 of run_fc_offline_inference()."""
    import json

    func_tokens = result.get("tokens_function_pred", result.get("tokens_function"))
    if func_tokens is None:
        print("  no function-channel tokens in the result")
        return 0

    positions = model._extract_function_call_positions(
        func_tokens, result.get("tokens_len"), result.get("tokens_text")
    )
    n = 0
    for b_info in positions:
        for call in b_info.get("function_calls", []):
            n += 1
            raw = call["call_text"]
            print(f"  TOOL CALL at step {call['start_pos']}: {raw}")
            clean = raw.replace("<SPECIAL_20>", "").replace("<SPECIAL_21>", "").strip()
            if "<TOOLCALL>" in clean:
                clean = clean.split("<TOOLCALL>")[1].split("</TOOLCALL>")[0].strip()
            try:
                calls = json.loads(clean) if clean.startswith("[") else [json.loads(clean)]
                for tc in calls:
                    print(f"      parsed: name={tc.get('name')!r} arguments={tc.get('arguments')}")
            except json.JSONDecodeError as e:
                print(f"      [WARN] tool call is not valid JSON ({e}): {clean!r}")
    return n


def spot_check_weights(model, checkpoint_dir, n_per_group=2):
    """Compare a sample of loaded parameters against the checkpoint file, numerically.

    The key-set audit proves names and shapes agree; the loader's own log line ("Loaded N
    tensors") proves it copied something. Neither proves the VALUES in memory match the file.
    This does, on a sample spread across the components that matter.
    """
    from safetensors import safe_open

    sd = dict(model.named_parameters())
    picked, groups = [], {}
    with safe_open(os.path.join(checkpoint_dir, "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key not in sd:
                continue
            # Group by component so we sample the LLM, perception and the heads alike.
            g = "llm" if key.startswith("llm.") else key.split(".")[0]
            if len(groups.setdefault(g, [])) < n_per_group:
                groups[g].append(key)
                picked.append(key)

        print(f"\n  Numeric spot check of {len(picked)} tensors against the checkpoint file:")
        worst = 0.0
        for key in picked:
            ref = f.get_tensor(key)
            got = sd[key].detach().to(device="cpu", dtype=ref.dtype)
            diff = (ref - got).abs().max().item() if ref.numel() else 0.0
            worst = max(worst, diff)
            print(f"      {'OK ' if diff == 0 else 'DIFF'}  max|Δ|={diff:<12.3g} {key}")
    return worst

# Same prompt as examples/speechlm2/offline_voicechat_infer.py, so output is comparable to
# the released model's documented behaviour.
DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)

SOURCE_SR = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Remapped STT checkpoint directory")
    parser.add_argument("--wav", required=True, help="Input wav (any sample rate, mono or stereo)")
    parser.add_argument("--config", default="conf/finetune/s2s_duplex_stt_11b.yaml")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy")
    parser.add_argument(
        "--pad-seconds",
        type=float,
        default=2.0,
        help="Trailing silence appended to the input. The agent replies along the audio "
        "timeline, so it needs frames after the user stops speaking to answer in.",
    )
    parser.add_argument(
        "--function-calling",
        action="store_true",
        help="Use the tool-declaring FC system prompt and decode the function channel. "
        "Without this the model has no tools to call and will (correctly) decline.",
    )
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="Skip loading the checkpoint. Use this once to see what a broken load looks like.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)

    import torchaudio

    config_path = os.path.join(project_root, "examples", "speechlm2", args.config)
    if not os.path.exists(config_path):
        config_path = args.config
    cfg = OmegaConf.load(config_path)

    OmegaConf.update(cfg, "model.pretrained_s2s_model", None if args.random_weights else args.checkpoint)
    OmegaConf.update(cfg, "data.train_ds.input_cfg", [{"type": "lhotse_shar", "shar_path": "/tmp/dummy"}])
    OmegaConf.update(cfg, "data.validation_ds.datasets.val_set_0.shar_path", "/tmp/dummy_val")
    OmegaConf.resolve(cfg)

    from nemo.collections.speechlm2.models import DuplexSTTModel

    print("=" * 70)
    print(f"Building DuplexSTTModel  (weights: {'RANDOM' if args.random_weights else args.checkpoint})")
    print("=" * 70)
    model = DuplexSTTModel(OmegaConf.to_container(cfg, resolve=True))

    if not args.random_weights:
        # Do this before .to(dtype), while the parameters still hold exactly what was copied
        # out of the file. Casting to bf16 afterwards is lossy and would blur the comparison.
        worst = spot_check_weights(model, args.checkpoint)
        if worst == 0.0:
            print("      -> exact match: the checkpoint values are in the model.")
        else:
            print(f"      -> [FAIL] largest deviation {worst:.3g}; the load did not take effect.")

    model = model.to(device=args.device, dtype=getattr(torch, args.dtype)).eval()

    wav_path = args.wav
    if not os.path.isabs(wav_path) and not os.path.exists(wav_path):
        wav_path = os.path.join(project_root, args.wav)
    wav, sr = torchaudio.load(wav_path)
    if sr != SOURCE_SR:
        wav = torchaudio.functional.resample(wav, sr, SOURCE_SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if args.pad_seconds > 0:
        wav = torch.cat([wav, torch.zeros(int(args.pad_seconds * SOURCE_SR), dtype=wav.dtype)])

    input_signal = wav.unsqueeze(0).to(device=args.device, dtype=getattr(torch, args.dtype))
    input_signal_lens = torch.tensor([wav.shape[0]], device=args.device)

    system_prompt = build_fc_system_prompt() if args.function_calling else args.system_prompt

    tokenizer = model.tokenizer
    prompt_ids = [tokenizer.bos_id] + tokenizer.text_to_ids(system_prompt) + [tokenizer.eos_id]
    prompt_tokens = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
    prompt_token_lens = torch.tensor([len(prompt_ids)], dtype=torch.long, device=args.device)

    print(f"\nInput : {wav_path}")
    print(f"        {wav.shape[0] / SOURCE_SR:.2f}s @ {SOURCE_SR} Hz "
          f"(incl. {args.pad_seconds}s trailing silence)")
    print(f"Prompt: {len(prompt_ids)} tokens")
    print("\nRunning offline_inference (decode_audio=False, no audio codec in this config)...")

    result = model.offline_inference(
        input_signal=input_signal,
        input_signal_lens=input_signal_lens,
        prompt_tokens=prompt_tokens,
        prompt_token_lens=prompt_token_lens,
        decode_audio=False,
        temperature=args.temperature,
    )

    text = result.get("text", [""])[0]
    print("\n" + "=" * 70)
    print("AGENT TEXT CHANNEL")
    print("=" * 70)
    print(text if text.strip() else "<empty>")

    n_calls = 0
    if args.function_calling:
        print("\n" + "-" * 70)
        print("FUNCTION CHANNEL")
        print("-" * 70)
        n_calls = report_function_calls(model, result)

    print("\n" + "=" * 70)
    # Fluent English is NOT evidence of a good load. cfg.pretrained_llm is fetched from HF
    # (duplex_stt_model.py:228) before our checkpoint is applied, so a model that loaded
    # nothing from the checkpoint still writes fluent prose — it just ignores the audio and
    # free-associates on the system prompt. The discriminating signal is duplex behaviour:
    # turn-taking timestamps, which exist only in the VoiceChat-finetuned weights.
    turn_taking = re.findall(r"<\$[\d.]+\$>|<\|[\d.]+\|>", text)
    print(f"  chars: {len(text)}   turn-taking tokens: {len(turn_taking)}")
    if not text.strip():
        print("  [FAIL] Empty output. Try a longer --pad-seconds, or check the load.")
    elif not turn_taking:
        print("  [FAIL] No turn-taking timestamps. The model is not exhibiting duplex "
              "behaviour, which means the VoiceChat-specific weights did not load — it is "
              "running as the base LLM and ignoring the audio.")
    else:
        print("  [OK] Duplex turn-taking timestamps present, so the VoiceChat weights are live.")
        print("       Now read the text above: it should be spoken-style and should respond to "
              "what was actually said, not to the system prompt.")

    if args.function_calling:
        if n_calls:
            print(f"  [OK] function_head emitted {n_calls} tool call(s) on the function channel.")
        else:
            print("  [FAIL] No tool call emitted. With tools declared and a request that matches "
                  "one, function_head should fire — check it received checkpoint weights.")


if __name__ == "__main__":
    main()
