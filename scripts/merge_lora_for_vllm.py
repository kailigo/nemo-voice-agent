#!/usr/bin/env python
"""
Merge our LoRA-SFT'd .ckpt's LLM backbone into plain weights and save an
HF-native Nemotron-H checkpoint dir that vanilla vLLM can load directly.

Scope: LLM backbone only (llm/embed_tokens/lm_head). Drops perception and
function_head -- this is for testing whether vLLM's mature, correctly-cached
Mamba-hybrid serving reproduces mode H on long generations, independent of
any duplex/audio/function-calling machinery. Not a servable duplex model.

Key rename: our DuplexSTTModel.llm *is* HF's `backbone` submodule (embeddings
extracted out, see duplex_stt_model.py ~L242-247), so the only rename needed
is the top-level prefix `llm.` -> `backbone.`. Internal names (A_log, q_proj,
etc.) are left untouched -- vLLM's own hf_to_vllm_mapper
(orig_to_new_prefix={"backbone": "model"}, orig_to_new_substr={"A_log": "A", ...})
expects to find them in that original HF-native form and does the rest of the
renaming itself at load time.

Usage:
  python scripts/merge_lora_for_vllm.py \
      --config-path examples/speechlm2/conf/finetune \
      --config-name s2s_duplex_stt_11b \
      --pretrained-s2s-model /fsx/home/kai.li/data/voicechat/stt_extracted_lora \
      --ckpt-path logs/sft_train8_0826/exp/checkpoints/step-500.ckpt \
      --output-dir /fsx/home/kai.li/data/voicechat/sft_step500_backbone_vllm
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from safetensors.torch import save_file


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--pretrained-s2s-model", required=True)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg_dir = str((REPO / args.config_path).resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=args.config_name)
    OmegaConf.update(cfg, "model.pretrained_s2s_model", args.pretrained_s2s_model)

    from nemo.collections.speechlm2 import DuplexSTTModel

    print("[merge] building DuplexSTTModel...", flush=True)
    t0 = time.time()
    model_cfg = OmegaConf.to_container(cfg, resolve=True)
    model = DuplexSTTModel(model_cfg)
    print(f"[merge] built in {time.time()-t0:.1f}s", flush=True)

    print(f"[merge] loading ckpt: {args.ckpt_path}", flush=True)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[merge] loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    if missing:
        print(f"  missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  unexpected (first 5): {unexpected[:5]}")

    print("[merge] merging LoRA into base weights (model.llm.merge_and_unload())...", flush=True)
    t0 = time.time()
    model.llm = model.llm.merge_and_unload()
    print(f"[merge] merged in {time.time()-t0:.1f}s", flush=True)

    lora_keys_left = [k for k in model.llm.state_dict().keys() if "lora" in k.lower()]
    assert not lora_keys_left, f"LoRA keys survived merge: {lora_keys_left[:5]}"

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[merge] assembling backbone-only state_dict...", flush=True)
    merged = {}
    for k, v in model.llm.state_dict().items():
        merged[f"backbone.{k}"] = v.contiguous().to(torch.bfloat16)
    # The true original HF key is backbone.embeddings.weight (DuplexSTTModel extracts
    # it out as a sibling attribute and deletes it from the backbone -- see
    # duplex_stt_model.py ~L242-247 -- so it must be put back to load correctly).
    # vLLM's own hf_to_vllm_mapper renames the substring "embeddings"->"embed_tokens" at
    # load time, so this name works for both vanilla HF .from_pretrained() AND vLLM;
    # saving it pre-renamed as "embed_tokens" instead works for vLLM by coincidence
    # (vLLM's internal attribute happens to be named embed_tokens) but silently leaves
    # plain HF's backbone.embeddings.weight randomly initialized -- verified by loading
    # this checkpoint both ways.
    for k, v in model.embed_tokens.state_dict().items():
        merged[f"backbone.embeddings.{k}"] = v.contiguous().to(torch.bfloat16)
    for k, v in model.lm_head.state_dict().items():
        merged[f"lm_head.{k}"] = v.contiguous().to(torch.bfloat16)

    dst_safetensors = out / "model.safetensors"
    print(f"[merge] writing {dst_safetensors} ({len(merged)} tensors)...", flush=True)
    save_file(merged, str(dst_safetensors))

    # Config + tokenizer straight from the base pretrained_llm HF cache -- architecture
    # is unchanged, only weight values differ, and we dropped nothing the base config
    # doesn't already describe (perception/function_head aren't part of this config).
    from transformers import AutoConfig, AutoTokenizer

    pretrained_llm = cfg.model.pretrained_llm
    print(f"[merge] copying config+tokenizer from {pretrained_llm}", flush=True)
    base_config = AutoConfig.from_pretrained(pretrained_llm, trust_remote_code=True)
    base_config.save_pretrained(str(out))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_llm, trust_remote_code=True)
    tokenizer.save_pretrained(str(out))

    print(f"[merge] done. Output: {out}", flush=True)
    print(f"[merge] safetensors size: {dst_safetensors.stat().st_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
