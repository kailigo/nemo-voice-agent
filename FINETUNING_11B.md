# Nemotron VoiceChat 11B — GPU Machine Handoff

**Created**: 2026-08-13  
**Purpose**: Everything needed to continue the finetuning work on a GPU machine.

---

## TL;DR

We want to finetune NVIDIA's Nemotron VoiceChat 11B (STT component) on tau2-bench domain data so the voice agent handles banking, healthcare, hotels, etc. better. All code, configs, and scripts are ready. The GPU machine needs to: clone the repo, download the model, extract weights, prepare data, and launch training.

---

## Source Repo

One repo, one branch. `nemotron-labs-voicechat` is a fork of NVIDIA's NeMo Speech
`nemotron-labs-voicechat` branch (it contains upstream commit `097dfe9`), so upstream's
`nemo/` and `examples/` sit in the same tree as our finetuning scripts and config:

```bash
git clone https://github.com/kailigo/nemo-voice-agent.git
cd nemo-voice-agent          # nemotron-labs-voicechat is the default branch
pip install -e . --no-deps --no-build-isolation
```

There used to be a second directory (`code/nemotron-voicechat`, a shallow clone of upstream)
with absolute symlinks pointing back here. It was redundant — this branch is a strict superset
— and the symlinks were version-controlled nowhere, so it has been removed. If you see a
reference to it anywhere, it means that text predates the consolidation.

Our custom scripts live in `scripts/` alongside upstream's. They are:

| Script | Purpose |
|--------|---------|
| `scripts/extract_stt_checkpoint.py` | Split combined 11B checkpoint → STT-only weights |
| `scripts/remap_checkpoint_for_lora.py` | **Mandatory.** Rename extracted keys to their LoRA-wrapped form (see Step 2b) |
| `scripts/tau2_to_conversations.py` | Convert tau2-bench tasks → conversation JSON |
| `scripts/prepare_lhotse_data.py` | Conversation JSON → Lhotse Shar training format |
| `scripts/run_finetune.sh` | One-command training launch |
| `scripts/verify_training_setup.py` | Config/model sanity check + dummy forward pass (**needs a GPU**) |
| `scripts/verify_checkpoint_load.py` | Audit which model params actually receive checkpoint weights |
| `scripts/smoke_test_inference.py` | Real-audio inference against the training config; proves the weights are live |

Training config: `examples/speechlm2/conf/finetune/s2s_duplex_stt_11b.yaml`

### Actual paths on this machine

Commands below use generic `/data/...` paths. On the current 8× H200 box they are:

| Generic | Actual |
|---------|--------|
| repo clone | `/fsx/home/kai.li/code/nemo-voice-agent` (branch `nemotron-labs-voicechat`) |
| conda env | `/fsx/home/kai.li/miniforge3/envs/voicechat` (`conda activate voicechat`) |
| `/data/checkpoints/voicechat-11b` | `/fsx/home/kai.li/data/voicechat/voicechat-11b` |
| `/data/checkpoints/stt_extracted_lora` | `/fsx/home/kai.li/data/voicechat/stt_extracted_lora` ← **use this one** |
| `/data/training/...` | not created yet — no training data exists |

---

## Architecture (what we're training)

```
NemotronVoiceChat 11B = DuplexSTTModel (10.10B) + DuplexEARTTS (1.00B)
                         ^^^^^^^^^^^^^^^^
                         This is what we finetune
```

Params measured from the released checkpoint, not estimated. As instantiated from our config
(`predict_user_text: false`) the STT model is **10.13B total / 2.41B trainable / 7.71B frozen**.

**DuplexSTTModel** components:
- **Speech Encoder**: Fast Conformer streaming, 0.61B params (`nvidia/nemotron-speech-streaming-en-0.6b`)
- **Modality Adapter**: `IdentityConnector` — **zero params**. The encoder already emits d_model=1024, so there is nothing to bridge. (Earlier drafts of this doc claimed a 2-layer Conformer, ~5M params; that is the 1.1B recipe, not this one.)
- **LLM**: Nemotron Nano V2 9B — hybrid Mamba/Transformer, 7.75B here (`nvidia/NVIDIA-Nemotron-Nano-9B-v2`)
- **Heads**: three tied-shape 0.587B blocks — `embed_tokens`, `lm_head`, and `function_head` (function-calling channel)

**Training approach**: LoRA on the LLM (rank=32, alpha=64), plus **fully trained** speech encoder,
`embed_tokens`, `lm_head` and `function_head`. This is not "LoRA-only": `freeze_params` matches
only `^audio_codec\..+$` and `^perception\.preprocessor\..+$`, so everything else outside the
LoRA-wrapped LLM is trainable. That is 2.41B trainable params, ~4x what a LoRA-only reading
suggests — budget optimizer state accordingly. TTS stays frozen/separate.

---

## Step-by-Step Execution Plan

### 0. Environment Setup

**Do not use the unpinned recipe this section used to give.** The authoritative recipe is in the
NeMo Speech branch's own README (`nemotron-labs-voicechat`, "Create the conda environment"), and
it must be followed verbatim: python 3.12, `torch==2.10.0` / `torchvision==0.25.0` /
`torchaudio==2.10.0`, `pip install -e ".[all]"`, uninstall `nvidia-resiliency-ext`, then the
pinned `transformers==4.56.0` / `lhotse==1.32.2` / `torchcodec==0.10.0`, and finally:

```bash
pip install --no-build-isolation --no-deps causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1
```

torch 2.10 is the newest release with **prebuilt** mamba-ssm/causal-conv1d wheels *and* a matching
torchaudio, so this takes ~40 s. Unpinned torch forces an nvcc build from source: 20+ minutes and
frequently broken.

Then apply one fix the branch README omits. `import torchcodec` fails on this box because conda's
ffmpeg/libopenvino need CXXABI_1.3.15+ while the system libstdc++ (Ubuntu 22.04 / gcc 11) only has
1.3.13 and wins the loader search:

```bash
conda install -c conda-forge "ffmpeg=7"
# plus an etc/conda/activate.d hook prepending $CONDA_PREFIX/lib to LD_LIBRARY_PATH
```

Both are already applied in the `voicechat` env on this machine. Verify — this must print
`2.10.0+cu128 True True NVIDIA H200`:

```bash
python -c "import torch, torchcodec; from transformers.utils.import_utils import is_mamba_2_ssm_available as m, is_causal_conv1d_available as c; print(torch.__version__, m(), c(), torch.cuda.get_device_name(0))"
```

```bash
# tau2-bench (for data generation)
pip install -e /path/to/tau2-bench
```

### 1. Download Model (44.4 GB)

```bash
huggingface-cli download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
    --local-dir /data/checkpoints/voicechat-11b
```

The checkpoint is a single `model.safetensors` file (~44.4 GB, FP32). Keys are prefixed `stt_model.*` and `tts_model.*`.

### 2. Extract STT Weights

```bash
python scripts/extract_stt_checkpoint.py \
    --input_dir /data/checkpoints/voicechat-11b \
    --output_dir /data/checkpoints/stt_extracted \
    --also_extract_tts
```

This produces:
- `/data/checkpoints/stt_extracted/model.safetensors` — STT weights with prefix stripped
- `/data/checkpoints/stt_extracted_tts/model.safetensors` — TTS weights (for later)

Expected output: **10.10B** params for STT, **1.00B** for TTS.

### 2b. Remap keys for LoRA (MANDATORY — skipping this silently breaks training)

```bash
python scripts/remap_checkpoint_for_lora.py \
    --src /data/checkpoints/stt_extracted \
    --dst /data/checkpoints/stt_extracted_lora
```

`DuplexSTTModel.__init__` installs LoRA (`duplex_stt_model.py` ~L313) *before* it loads
`pretrained_s2s_model` (~L321). peft has by then renamed every LLM key
(`llm.layers.0…` → `llm.base_model.model.layers.0…`, plus `.base_layer` on LoRA-targeted
projections). The loader copies by exact key match and warns only about checkpoint keys missing
from the model — never about model params that received nothing. Point it at `stt_extracted`
directly and **339 LLM tensors (7.75B params) are skipped**, with one warning line as the symptom.

The failure is nearly invisible: the skipped LLM does not become noise. `cfg.pretrained_llm` was
already loaded from HF at ~L228, so the LLM falls back to base Nemotron-Nano-9B-v2 — fluent
English with none of the VoiceChat duplex / turn-taking / function-calling finetuning. You cannot
catch this by reading generated text.

### 3. Verify Setup (do not skip)

```bash
# 3a. Names and shapes: every non-LoRA model param must receive a checkpoint tensor
python scripts/verify_checkpoint_load.py --checkpoint /data/checkpoints/stt_extracted_lora
# want: "Exact-name matches: N / N matchable" and "MODEL-ONLY: 0 tensors"

# 3b. Config + instantiation + dummy forward pass. NEEDS A GPU (see below).
python scripts/verify_training_setup.py --config conf/finetune/s2s_duplex_stt_11b.yaml

# 3c. Values and behaviour: real audio through the real training config
python scripts/smoke_test_inference.py \
    --checkpoint /data/checkpoints/stt_extracted_lora \
    --wav examples/speechlm2/sample_audio/sample_general.wav
```

**`verify_training_setup.py` requires a GPU** — an earlier version of this doc said "no GPU
needed", which is wrong. Nemotron-H's Mamba mixer calls
`torch.cuda.stream(torch.cuda.default_stream(hidden_states.device))` with no CPU fallback, so the
LLM cannot run on CPU at all. Steps 1, 2 and 4 of that script are CPU-safe; the forward pass is
not. Use `--skip_forward` to leave it out.

For 3c, the pass/fail signal is **turn-taking timestamps** (`<$0.72$>`, `<|2.08|>`) in the output,
not fluency — fluency proves nothing, per the fallback described in 2b. A correct run on
`sample_general.wav` looks like:

```
<$0.72$> <|2.08|> Hi there! How can I help you today? <$8.56$> <|12.16|> The sky is blue. …
```

To exercise the function-calling channel (relevant for tau2), add `--function-calling`, which
declares tools via the repo's own `template.jinja` and decodes the function channel. Without
declared tools the model correctly *declines* to call anything, which looks like a broken
`function_head` but is not.

### 4. Prepare Training Data

#### 4a. Generate conversations from tau2-bench

```bash
python scripts/tau2_to_conversations.py \
    --tau2_dir /path/to/tau2-bench \
    --domains banking,healthcare,hotels,calendar,car_rental,events,housing,media,transit,restaurant \
    --output /data/training/conversations.json \
    --mode template
```

This produces 82 conversations (one per task) in JSON format. Template mode is fast and doesn't need an LLM API.

#### 4b. Synthesize audio + create Lhotse shards

```bash
python scripts/prepare_lhotse_data.py \
    --input /data/training/conversations.json \
    --output_dir /data/training/lhotse \
    --tts_backend edge_tts \
    --shard_size 100
```

**Important caveat**: The Lhotse CutSet creation in `prepare_lhotse_data.py` is a skeleton — it references only the first turn's audio per conversation instead of properly concatenating all turns into a multi-supervision recording. This needs to be fixed before real training. For a smoke test, the dummy backend (`--tts_backend dummy`) will generate random noise audio that passes through the pipeline.

For a proper fix, `create_lhotse_cutset()` should:
1. Concatenate all user turns into one source recording (16 kHz)
2. Concatenate all agent turns into one target recording (22050 Hz)
3. Create supervisions with correct start/duration relative to the concatenated audio

### 5. Launch Training

```bash
bash scripts/run_finetune.sh \
    /data/checkpoints/stt_extracted_lora \
    /data/training/lhotse/shards \
    /data/training/lhotse_val/shards \
    8  # number of GPUs
```

Or manually:

```bash
cd examples/speechlm2

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    s2s_duplex_stt_train.py \
    --config-path conf/finetune \
    --config-name s2s_duplex_stt_11b \
    model.pretrained_s2s_model=/data/checkpoints/stt_extracted_lora \
    data.train_ds.input_cfg.0.shar_path=/data/training/lhotse/shards \
    data.validation_ds.datasets.val_set_0.shar_path=/data/training/lhotse_val/shards \
    trainer.devices=8
```

---

## Training Config Key Settings

File: `examples/speechlm2/conf/finetune/s2s_duplex_stt_11b.yaml`

| Setting | Value | Rationale |
|---------|-------|-----------|
| Strategy | DDP (find_unused_parameters=true) | Mamba layers don't shard cleanly with FSDP |
| Precision | bf16-true | Standard for 11B models |
| LoRA rank | 32, alpha=64 | ~50M trainable LoRA params |
| LoRA targets | q/k/v/o_proj + gate/up/down_proj | All attention + MLP projections |
| LR | 1e-4 | Lower than 1.1B (3e-4) for LoRA |
| Scheduler | Cosine annealing, 200 warmup steps | |
| Batch size | 2 per GPU × 8 GPUs × 4 accum = 64 effective | |
| Max steps | 10,000 | ~2.5 epochs over 250K frames |
| Gradient clip | 1.0 | |
| Audio loss weight | 0 | STT only, no audio generation |
| Text loss weight | 3 | Matches paper |
| Checkpoints | Every 500 steps, top-3 by val_asr_bleu | |
| Output dir | `results/finetune_stt_11b/` | |

**Frozen** (all `freeze_params` matches): audio codec, `perception.preprocessor` (mel-spectrogram)  
**Trainable (2.41B total)**: speech encoder 0.61B, `embed_tokens` 0.587B, `lm_head` 0.587B,
`function_head` 0.587B, LLM via LoRA ~0.05B. There is no modality adapter to train
(`IdentityConnector`, zero params).

---

## Hardware Requirements

| Config | GPUs | Batch | Notes |
|--------|------|-------|-------|
| This machine | 8× H200 143GB | 2/GPU, accum=4 | What the config targets; comfortable |
| Workable | 8× A100/H100 80GB | 1/GPU, accum=8 | Tight — see the optimizer-state note below |

The earlier "Minimum: 4× A100 80GB, LoRA only" row was based on a ~0.65B trainable-param estimate
and understated optimizer memory by roughly 5x. The real figure is **2.41B trainable**, because
`freeze_params` leaves the speech encoder and all three 0.587B head/embedding blocks unfrozen (see
Architecture above). Per rank that is:

- weights: 10.13B in bf16 ≈ 20 GB
- AdamW state for 2.41B params: fp32 m+v ≈ **19 GB** (plus fp32 master weights if used)
- gradients + activations on top

DDP replicates all of this per GPU, so 80 GB is workable but not roomy. If you genuinely want
LoRA-only memory, add `^embed_tokens\..+$`, `^lm_head\..+$`, `^function_head\..+$` and
`^perception\..+$` to `freeze_params` — but note the released 11B was trained with these unfrozen,
so freezing them departs from the recipe.

If DDP OOMs, switch to FSDP by editing the config:
```yaml
trainer:
  strategy:
    _target_: lightning.pytorch.strategies.ModelParallelStrategy
    data_parallel_size: 8
```

---

## Key Code Locations in the NeMo Repo

| What | Path | Key Lines |
|------|------|-----------|
| STT model class | `nemo/collections/speechlm2/models/duplex_stt_model.py` | L179 (class), L231 (Nemotron handling), L321 (checkpoint loading), L2313 (training_step) |
| Combined model | `nemo/collections/speechlm2/models/nemotron_voicechat.py` | L184 (init), L235 (checkpoint loading) |
| LoRA utility | `nemo/collections/speechlm2/parts/lora.py` | `maybe_install_lora()` |
| Pretrained utils | `nemo/collections/speechlm2/parts/pretrained.py` | `load_pretrained_hf()`, `setup_speech_encoder()` |
| Dataset class | `nemo/collections/speechlm2/data/s2s_duplex_dataset.py` | Batch format |
| 1.1B training config | `examples/speechlm2/conf/s2s_duplex_stt.yaml` | Template for our 11B config |
| 11B inference config | `examples/speechlm2/conf/nemotron-labs-voicechat.yaml` | Architecture params |
| Training entry point | `examples/speechlm2/s2s_duplex_stt_train.py` | Launch script |

---

## How the Model Handles Nemotron Nano

In `duplex_stt_model.py` line 231:

```python
if 'Nemotron' in self.cfg.pretrained_llm:
    # Uses 'backbone' instead of 'model'
    # Uses 'embeddings' instead of 'embed_tokens'
    # Uses 'cache_params' instead of 'past_key_values'
```

This is already handled — no code changes needed. The config sets `base_model_name: backbone`, `embed_tokens_name: embeddings`, `cache_key: cache_params`.

---

## Known Issues / Gotchas

1. **`prepare_lhotse_data.py` is a skeleton**: The CutSet creation only uses the first turn's audio. Fix this before real training (see Step 4b above).

2. **`NemotronVoiceChat.training_step()` returns None**: This is the combined model class — don't train via that. Train `DuplexSTTModel` directly (which our config does via `s2s_duplex_stt_train.py`).

3. **Mamba kernel installation**: use the pinned wheels in Step 0. If the fast kernels are absent, transformers falls back to a slower pure-PyTorch mixer — but that fallback is still **CUDA-only**, so it is not a way to run on CPU.

7. **Architectural config keys must match `voicechat-11b/config.json`.** `duplex_*_channel_weight` read like loss weights but are not: `create_fusion_module` (`duplex_stt_model.py` ~L296) multiplies each channel's embedding by its weight before summing into the LLM input, so they are part of the forward pass. NeMo defaults all four to 1.0, but the released checkpoint uses `duplex_function_channel_weight: 2.0`. With 1.0 an FC test still emitted a correct tool call, but two decoder steps late and with the agent text channel collapsed to 19 characters. Same class of key: `predict_user_text`, `use_function_head`, `fuse_method`. Loss weights are ours to tune; these are not.

8. **`predict_user_text` must stay `false`** to match the release. Turning it on builds `asr_head` + `embed_asr_tokens` (1.17B) which the checkpoint has no weights for, and because they are deep-copied from `lm_head` at ~L275 *before* the checkpoint loads, they get the base HF head rather than the VoiceChat-finetuned one. To enable it properly, regenerate with `remap_checkpoint_for_lora.py --warm-start-asr-head`.

4. **HuggingFace `trust_remote_code=True`**: Required for Nemotron Nano. The model loading code already sets this.

5. **Checkpoint size**: Combined checkpoint is 44.4 GB (FP32). After extraction, STT is ~38 GB (FP32) → ~19 GB in bf16 at runtime.

6. **`find_unused_parameters: true`**: Required for DDP because the Mamba/Transformer hybrid has conditional paths that may leave some params unused in certain forward passes.

---

## Smoke Test Sequence

Before a full training run, verify incrementally:

Steps 1-2 below are already green on this machine; steps 3-4 are blocked only on training data.

```bash
# 1. Checkpoint audit + config + dummy forward pass (see Step 3 above for all three gates)
python scripts/verify_checkpoint_load.py --checkpoint /data/checkpoints/stt_extracted_lora
python scripts/verify_training_setup.py

# 2. Real audio, including the function-calling channel
python scripts/smoke_test_inference.py --checkpoint /data/checkpoints/stt_extracted_lora \
    --wav examples/speechlm2/sample_audio/sample_fc.wav --function-calling

# 3. Overfit on 10 examples (loss should approach 0)
torchrun --nproc_per_node=1 s2s_duplex_stt_train.py \
    --config-path conf/finetune --config-name s2s_duplex_stt_11b \
    model.pretrained_s2s_model=/data/checkpoints/stt_extracted_lora \
    data.train_ds.input_cfg.0.shar_path=/data/training/lhotse/shards \
    data.validation_ds.datasets.val_set_0.shar_path=/data/training/lhotse/shards \
    trainer.devices=1 \
    trainer.max_steps=100 \
    trainer.limit_train_batches=10

# 4. Multi-GPU check (short run)
torchrun --nproc_per_node=4 s2s_duplex_stt_train.py \
    --config-path conf/finetune --config-name s2s_duplex_stt_11b \
    model.pretrained_s2s_model=/data/checkpoints/stt_extracted_lora \
    data.train_ds.input_cfg.0.shar_path=/data/training/lhotse/shards \
    data.validation_ds.datasets.val_set_0.shar_path=/data/training/lhotse/shards \
    trainer.devices=4 \
    trainer.max_steps=50
```

---

## tau2-bench Context

tau2-bench has 10 domains (82 total tasks): banking, healthcare, hotels, calendar, car_rental, events, housing, media, transit, restaurant.

The tau2-bench repo is at: `/path/to/tau2-bench` (adjust on GPU machine).  
Source: `https://github.com/<org>/tau2-bench` (or wherever it's hosted).

The conversation converter (`tau2_to_conversations.py`) imports from tau2-bench's registry to load tasks and generate oracle-guided conversations. It needs tau2-bench installed or on `sys.path`.

---

## Expected Outcome

After training, we get a finetuned `DuplexSTTModel` that:
- Better understands domain-specific speech (banking terms, medical terminology, etc.)
- Correctly executes tool calls in voice conversations
- Maintains turn-taking and duplex behavior

The finetuned LoRA weights can be merged back or loaded alongside the base model. Evaluation is on the tau_voice benchmark (action accuracy + NL quality).

---

## File Transfer Checklist

These files need to exist on the GPU machine:

```
nemo-voice-agent/                            # fork of NeMo Speech (branch nemotron-labs-voicechat)
├── scripts/
│   ├── extract_stt_checkpoint.py            # OUR SCRIPT
│   ├── remap_checkpoint_for_lora.py         # OUR SCRIPT (mandatory, see Step 2b)
│   ├── tau2_to_conversations.py             # OUR SCRIPT
│   ├── prepare_lhotse_data.py               # OUR SCRIPT
│   ├── run_finetune.sh                      # OUR SCRIPT
│   ├── verify_training_setup.py             # OUR SCRIPT
│   ├── verify_checkpoint_load.py            # OUR SCRIPT
│   └── smoke_test_inference.py              # OUR SCRIPT
├── examples/speechlm2/conf/finetune/
│   └── s2s_duplex_stt_11b.yaml             # OUR CONFIG
└── (rest is from the git clone)

tau2-bench/                                  # Needed for data generation
└── src/tau2/...
```

If the GPU machine can't access this Mac, copy the 9 custom files manually (8 scripts + 1 config). Everything else comes from `git clone`.
