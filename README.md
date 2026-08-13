# Nemotron VoiceChat 11B — GPU Machine Handoff

**Created**: 2026-08-13  
**Purpose**: Everything needed to continue the finetuning work on a GPU machine.

---

## TL;DR

We want to finetune NVIDIA's Nemotron VoiceChat 11B (STT component) on tau2-bench domain data so the voice agent handles banking, healthcare, hotels, etc. better. All code, configs, and scripts are ready. The GPU machine needs to: clone the repo, download the model, extract weights, prepare data, and launch training.

---

## Source Repo

```bash
git clone https://github.com/NVIDIA-NeMo/Speech.git --branch nemotron-labs-voicechat nemotron-voicechat
cd nemotron-voicechat
```

Our custom scripts live in `scripts/` within this clone. They are:

| Script | Purpose |
|--------|---------|
| `scripts/extract_stt_checkpoint.py` | Split combined 11B checkpoint → STT-only weights |
| `scripts/tau2_to_conversations.py` | Convert tau2-bench tasks → conversation JSON |
| `scripts/prepare_lhotse_data.py` | Conversation JSON → Lhotse Shar training format |
| `scripts/run_finetune.sh` | One-command training launch |
| `scripts/verify_training_setup.py` | Dry-run config/model sanity check (no GPU needed) |

Training config: `examples/speechlm2/conf/finetune/s2s_duplex_stt_11b.yaml`

---

## Architecture (what we're training)

```
NemotronVoiceChat 11B = DuplexSTTModel (~9.6B) + DuplexEARTTS (~1.5B)
                         ^^^^^^^^^^^^^^^^
                         This is what we finetune
```

**DuplexSTTModel** components:
- **Speech Encoder**: Fast Conformer streaming, 0.6B params (`nvidia/nemotron-speech-streaming-en-0.6b`)
- **Modality Adapter**: 2-layer Conformer, ~5M params (bridges encoder → LLM)
- **LLM**: Nemotron Nano V2 9B — hybrid Mamba/Transformer (`nvidia/NVIDIA-Nemotron-Nano-9B-v2`)
- **Text head**: Standard lm_head for next-token prediction

**Training approach**: LoRA on the LLM (rank=32, alpha=64) + full training of speech encoder & modality adapter. TTS stays frozen/separate.

---

## Step-by-Step Execution Plan

### 0. Environment Setup

```bash
# Clone repo
git clone https://github.com/NVIDIA-NeMo/Speech.git --branch nemotron-labs-voicechat nemotron-voicechat
cd nemotron-voicechat

# Install NeMo (editable)
pip install -e .

# Key dependencies
pip install safetensors lhotse soundfile torchaudio peft edge-tts
pip install lightning omegaconf hydra-core

# For Nemotron Nano (Mamba support)
pip install mamba-ssm causal-conv1d  # requires CUDA

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

Expected output: ~9.6B params for STT, ~1.5B for TTS.

### 3. Verify Setup (optional but recommended)

```bash
python scripts/verify_training_setup.py --config conf/finetune/s2s_duplex_stt_11b.yaml
```

This checks config resolution, model instantiation, and forward pass with dummy data. Catches issues before committing to a full run.

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
    /data/checkpoints/stt_extracted \
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
    model.pretrained_s2s_model=/data/checkpoints/stt_extracted \
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

**Frozen**: Preprocessor (mel-spectrogram), audio codec  
**Trainable**: Speech encoder (0.6B), modality adapter (~5M), LLM via LoRA (~50M)

---

## Hardware Requirements

| Config | GPUs | Batch | Notes |
|--------|------|-------|-------|
| Minimum | 4× A100 80GB | 1/GPU, accum=8 | LoRA only, gradient checkpointing |
| Recommended | 8× A100 80GB | 2/GPU, accum=4 | Full encoder + LoRA |
| Fast | 8× H100 80GB | 2-4/GPU | 2-3x faster |

Memory estimate: ~40GB per GPU (11B in bf16 + activations + LoRA adapters). A100 80GB should fit batch_size=2 comfortably.

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

3. **Mamba kernel installation**: `pip install mamba-ssm causal-conv1d` requires CUDA toolkit. If it fails, the model falls back to a pure-PyTorch implementation (slower but works).

4. **HuggingFace `trust_remote_code=True`**: Required for Nemotron Nano. The model loading code already sets this.

5. **Checkpoint size**: Combined checkpoint is 44.4 GB (FP32). After extraction, STT is ~38 GB (FP32) → ~19 GB in bf16 at runtime.

6. **`find_unused_parameters: true`**: Required for DDP because the Mamba/Transformer hybrid has conditional paths that may leave some params unused in certain forward passes.

---

## Smoke Test Sequence

Before a full training run, verify incrementally:

```bash
# 1. Config loads correctly
python scripts/verify_training_setup.py --skip_model

# 2. Model instantiates (with pretrained weights)
python scripts/verify_training_setup.py

# 3. Overfit on 10 examples (loss should approach 0)
torchrun --nproc_per_node=1 s2s_duplex_stt_train.py \
    --config-path conf/finetune --config-name s2s_duplex_stt_11b \
    model.pretrained_s2s_model=/data/checkpoints/stt_extracted \
    data.train_ds.input_cfg.0.shar_path=/data/training/lhotse/shards \
    data.validation_ds.datasets.val_set_0.shar_path=/data/training/lhotse/shards \
    trainer.devices=1 \
    trainer.max_steps=100 \
    trainer.limit_train_batches=10

# 4. Multi-GPU check (short run)
torchrun --nproc_per_node=4 s2s_duplex_stt_train.py \
    --config-path conf/finetune --config-name s2s_duplex_stt_11b \
    model.pretrained_s2s_model=/data/checkpoints/stt_extracted \
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
nemotron-voicechat/                          # git clone of NeMo Speech
├── scripts/
│   ├── extract_stt_checkpoint.py            # OUR SCRIPT
│   ├── tau2_to_conversations.py             # OUR SCRIPT
│   ├── prepare_lhotse_data.py               # OUR SCRIPT
│   ├── run_finetune.sh                      # OUR SCRIPT
│   └── verify_training_setup.py             # OUR SCRIPT
├── examples/speechlm2/conf/finetune/
│   └── s2s_duplex_stt_11b.yaml             # OUR CONFIG
└── (rest is from the git clone)

tau2-bench/                                  # Needed for data generation
└── src/tau2/...
```

If the GPU machine can't access this Mac, copy the 6 custom files manually (5 scripts + 1 config). Everything else comes from `git clone`.
