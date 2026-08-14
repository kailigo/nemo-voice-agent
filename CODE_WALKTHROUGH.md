# DuplexSTTModel — Code Walkthrough

A guided tour of the training code we finetune: how a conversation becomes a batch, how four
parallel channels become one vector per audio frame, and where every load-bearing assumption is
enforced.

**Provenance.** Every file:line reference below was read out of the source, not inferred, and
verified at commit `906c816b2` (2026-08-14). Line numbers drift; the surrounding function names
and quoted code are the durable anchors. All paths under `nemo/collections/speechlm2/` are
upstream NVIDIA code that we do **not** modify — see [Ours vs upstream](#2-ours-vs-upstream).

Companion docs: [FINETUNING_11B.md](FINETUNING_11B.md) for the operational runbook (env, weights,
launch commands, hardware).

New to this code? Two orienting sections sit at the back rather than the front, so the main tour
stays linear: [§7](#7-the-model-family--duplexsttmodel-vs-duplexs2smodel) places `DuplexSTTModel` in
its family (and explains why `audio_loss_weight: 0` is vestigial), and
[§8](#8-the-released-checkpoint--and-how-the-training-stack-is-reconstructed) explains where every
weight we train actually comes from.

---

## Table of contents

1. [The central mental model](#1-the-central-mental-model)
2. [Ours vs upstream](#2-ours-vs-upstream)
3. [The call chain](#3-the-call-chain)
4. [Layer by layer](#4-layer-by-layer)
   - [4a. `__init__` — assembling the model](#4a-__init__--assembling-a-frankenstein-model)
   - [4b. `perception` — audio to embeddings](#4b-perception--audio-to-llm-dim-embeddings)
   - [4c. `fusion_module` — 4 channels to 1](#4c-fusion_module--4-channels-to-1-vector)
   - [4d. `prepare_labels` — time-shift bookkeeping](#4d-prepare_labels--the-time-shift-bookkeeping)
   - [4e. The function-calling channel](#4e-the-function-calling-channel--the-genuinely-unusual-part)
   - [4f. Losses](#4f-losses)
   - [4g. Data loading](#4g-data-loading)
5. [Where to look for X](#5-where-to-look-for-x)
6. [Things that will bite you](#6-things-that-will-bite-you)
7. [The model family — `DuplexSTTModel` vs `DuplexS2SModel`](#7-the-model-family--duplexsttmodel-vs-duplexs2smodel)
8. [The released checkpoint and how the training stack is reconstructed](#8-the-released-checkpoint--and-how-the-training-stack-is-reconstructed)
   - [8a. One flat state dict, two prefixes](#8a-the-release-is-one-flat-state-dict-with-two-prefixes)
   - [8b. The bolted-on RNNT head](#8b-there-is-a-third-thing-in-there-belonging-to-neither-model-class)
   - [8c. `config.json` is the whole training yaml](#8c-configjson-is-not-a-model-config--it-is-the-whole-training-yaml-recursively-nested)
   - [8d. Reconstruction: three legs](#8d-reconstruction-three-legs-only-one-of-which-is-the-checkpoint)
   - [8e. Two silent failure modes](#8e-two-ways-this-reconstruction-fails-silently)
9. [Glossary](#9-glossary)

---

## 1. The central mental model

This is **not** an autoregressive LM over text tokens. The sequence axis is **time** — audio frames
at 12.5 Hz (`data.frame_length: 0.08`). At every frame the model consumes **one fused embedding**
built from up to four parallel channels, and predicts the **next frame's** token in two or three
channels simultaneously.

```
                 frame:  0    1    2    3     4      5    6  ...   (0.08 s each)
  +-- user audio    -->  [encoder embeddings, 1024 -> 4480 dim]     INPUT only
  |-- agent text    -->  PAD  PAD  BOS  "Hi" "there" EOS  PAD       IN + OUT
  |-- user text/ASR -->  ^   "hel" "lo"  $    PAD    PAD  PAD       IN + OUT (off for us)
  +-- function      -->  PAD  PAD  PAD  SOTC <TOOLCALL>... EOTC     IN + OUT
                            |
                            v  weighted sum -> ONE vector per frame
                        +-----------------------------+
                        |  Nemotron Nano 9B backbone  |
                        +-----------------------------+
                            |
              +-------------+-------------+
              v             v             v
          lm_head      function_head    asr_head        three independent copies
        (agent text)   (tool calls)   (user text)       of the same initial head
```

The enabling trick is `duplex_stt_model.py:245-247`:

```python
self.embed_tokens = getattr(self.llm, embed_tokens_name)
delattr(self.llm, embed_tokens_name)     # the LLM can no longer embed tokens
```

The embedding table is **detached from the LLM and owned by the wrapper**. The LLM is therefore
only ever called with `inputs_embeds=` (`duplex_stt_model.py:485`), never `input_ids`. That is what
makes multi-channel fusion possible at all — and also why the model is 10.13B rather than 9B:
three 0.587B head/embedding blocks hang off the side.

### Parameter budget (measured, `predict_user_text: false`)

| Block | Params | Trainable |
|---|---:|:---:|
| `llm` (Nemotron Nano 9B backbone) | 7.75B | frozen except LoRA |
| LoRA adapters | 36.1M | yes |
| `perception` (Fast Conformer + proj) | 0.61B | yes |
| `embed_tokens` | 0.587B | yes |
| `lm_head` | 0.587B | yes |
| `function_head` | 0.587B | yes |
| **Total** | **10.13B** | **2.41B trainable / 7.71B frozen** |

With `predict_user_text: true`, add `asr_head` + `embed_asr_tokens` → 11.30B / 3.59B trainable.

---

## 2. Ours vs upstream

`git diff --stat vendor/nemotron-labs-voicechat..main` — **18 files, +3568 lines, zero deletions.**
We are purely additive.

| Ours | Role |
|---|---|
| `examples/speechlm2/conf/finetune/s2s_duplex_stt_11b.yaml` | the only config we control; 307 lines, mostly comments documenting traps |
| `scripts/tau2_to_conversations.py` | tau2 simulation → episode JSON |
| `scripts/episodes_to_nemotron_training.py` | episode JSON → Lhotse Shar (the generator) |
| `scripts/repair_tau2_shards.py` | one-off repair for shards produced before the format bugs were understood |
| `scripts/extract_stt_checkpoint.py` | pull the STT component out of the released 11B |
| `scripts/remap_checkpoint_for_lora.py` | rename keys for peft's `.base_layer.` infix |
| `scripts/verify_checkpoint_load.py`, `verify_training_setup.py`, `smoke_test_inference.py` | pre-flight gates |
| `scripts/make_synthetic_shards.py`, `prepare_lhotse_data.py` | synthetic and alternate data paths |
| `scripts/run_finetune.sh` | launch wrapper (calls `torchrun` — needs an allocation) |
| `FINETUNING_11B.md`, `CODE_WALKTHROUGH.md` | docs |

Everything under `nemo/collections/speechlm2/` is upstream and unmodified. When something
misbehaves there we work around it **in config** rather than patching, which is why the yaml
carries so much prose.

---

## 3. The call chain

```
examples/speechlm2/s2s_duplex_stt_train.py          <- 87 lines, the whole entry point
  |
  +- DuplexSTTModel(cfg)                             models/duplex_stt_model.py:180
  +- DuplexS2SDataset(...)                           data/s2s_dataset.py:585
  +- DataModule(cfg.data, ...)                       data/datamodule.py:25
  +- trainer.fit(model, datamodule)
       |
       +- training_step(batch, idx)                   duplex_stt_model.py:2313
            +- prepare_inputs(batch)                  duplex_stt_model.py:1680   <-- the heart
            |    +- audio augmentation                L1697-1786
            |    +- self.perception(...)              L1791   audio -> embeddings
            |    +- _build_function_calling_channel   L936
            |    +- _expand_channels_with_insertions  L1500
            |    +- prepare_labels(...)               parts/label_prep.py:64
            |    +- self.embed_tokens(text_inputs)    L2112
            |    +- self.fusion_module(...)           L2198   <- 4 channels -> 1
            |    +- builds loss_scale masks           L2225-2290
            +- self.forward(input_embeds)             duplex_stt_model.py:466
            |    +- self.llm(inputs_embeds=...) -> lm_head / function_head / asr_head
            +- cross_entropy * loss_scale             L2366, L2375, L2462
```

Only ~87 lines of glue in the entry point; everything lives in the model class.

---

## 4. Layer by layer

### 4a. `__init__` — assembling a Frankenstein model

`duplex_stt_model.py:180-395`. **Order matters enormously.**

```python
L228  llm = load_pretrained_hf(cfg.pretrained_llm)          # base HF Nemotron
L242  self.llm      = getattr(llm, "backbone")              # Nemotron: 'backbone', not 'model'
L243  self.lm_head  = llm.lm_head
L245  self.embed_tokens = getattr(self.llm, "embeddings")
L247  delattr(self.llm, embed_tokens_name)

L275  self.asr_head         = copy.deepcopy(self.lm_head)   # only if predict_user_text
L283  self.embed_asr_tokens = copy.deepcopy(self.embed_tokens)
L296  self.fusion_module    = create_fusion_module(...)
L307  self.function_head    = copy.deepcopy(self.lm_head)   # if use_function_head
L313  maybe_install_lora(self)                              # wraps self.llm with peft
L316  setup_speech_encoder(self)                            # builds self.perception
L321  if self.cfg.get("pretrained_s2s_model", None):          # <-- LOADS LAST
L333      safe_open(...)                                      #     incremental_loading: true (ours)
L356      self.init_from_model_from_ckpt(...)                 #     fallback branch
```

**The trap that shaped our config:** the `deepcopy`s at L275/L283/L307 happen *before* the
checkpoint load at L321. Any head whose weights are absent from the checkpoint silently keeps a
copy of the **base HF** `lm_head` — not the VoiceChat-finetuned one. That is why our yaml sets
`predict_user_text: false` (yaml 74-88) and why `remap_checkpoint_for_lora.py
--warm-start-asr-head` exists.

Three model-family branches exist because attribute names differ: Nemotron (L231),
Qwen2.5 (L248), generic fallback (L265). Our config supplies all three overrides:

```yaml
base_model_name: backbone      # not "model"
embed_tokens_name: embeddings  # not "embed_tokens"
cache_key: cache_params        # Mamba state, not past_key_values
```

Special token IDs are pinned here too (L234-240): `bos='<s>'`, `eos='</s>'`,
`pad='<SPECIAL_12>'`, and `user_bos='^'` / `user_eos='$'` — literal caret and dollar characters,
which is surprising but deliberate.

Lazy init at the tail (L390-394): the silence template and its frame rate are deferred because the
model is not on-device yet during `__init__`.

### 4b. `perception` — audio to LLM-dim embeddings

`modules/perception.py:24`, `forward` at L104. Four stages:

```
waveform 16 kHz
  -> preprocessor      mel-spectrogram; BUFFERS ONLY, no nn.Parameters
  -> encoder           Fast Conformer, 0.61B, d_model=1024, 8x subsampling
  -> modality_adapter  IdentityConnector for us (a no-op)
  -> proj              nn.Linear(1024 -> 4480)
```

The `preprocessor` holding only buffers (`featurizer.fb`, `featurizer.window`) is why
`freeze_params: ["^perception\.preprocessor\..+$"]` reports UNMATCHED —
`named_parameters()` does not iterate buffers. Benign.

`setup_speech_encoder` (`parts/pretrained.py`) does something sneaky: it **overwrites**
`cfg.perception.preprocessor` and `.encoder` wholesale from the ASR `.nemo`, then re-applies only
the keys you explicitly set. That is how `att_context_size: [70, 0]` survives (yaml 173-177) while
everything else comes from the pretrained ASR. It also loads with `strict=False` and deliberately
does **not** register the RNNT decoder/joint, to keep them out of `state_dict()`.

The `IdentityConnector` choice (yaml 160-170) is load-bearing: a Conformer adapter would make
`proj` a `Linear(512, 4480)`, so the pretrained `proj` weights would silently fail to load.

### 4c. `fusion_module` — 4 channels to 1 vector

`parts/fusion.py`. Four strategies; we use `add`. `AddFusion.forward` (L93) *is* the architecture:

```python
output = agent_text_embeds * self.agent_text_weight            # 1.0
output = output + user_audio_embeds * self.user_audio_weight   # 1.0
output = output + user_text_embeds  * self.user_text_weight    # inert for us
output = output + function_embeds   * self.function_weight     # 2.0  <- NOT the default
```

These scalars are **architectural, not hyperparameters** — they are baked into what the loaded
weights expect. `duplex_function_channel_weight: 2.0` against NeMo's default of `1.0` is the single
easiest way to silently break the checkpoint. Set from the released 11B's `config.json`.

The other three strategies add learned parameters and are only for training from scratch:

| `fuse_method` | Class | Mechanism |
|---|---|---|
| `add` (ours) | `AddFusion` (L72) | weighted sum, no params |
| `concat` | `ConcatFusion` (L109) | per-channel LayerNorm, concat, `Linear(4D -> D)` |
| `gated_simple` | `GatedFusionSimple` (L171) | per-timestep softmax gate over 4 streams |
| `gated_gmu` | `GatedFusionGMU` (L260) | per-timestep **and** per-dimension gate |

Gated methods override `tie_and_roll_embed` and `use_channel_embeds` (L214-222) with a warning.
Note `function_weight` is passed `0.0`, not `1.0`, when `use_function_head=False`
(`duplex_stt_model.py:302`).

### 4d. `prepare_labels` — the time-shift bookkeeping

`parts/label_prep.py:64`. Standard next-token shift, but on the frame axis (L316):

```python
text_inputs = target_tokens[:, :-1]   # frames 0..T-1
text_labels = target_tokens[:, 1:]    # frames 1..T
```

with the matching `user_audio_embeds = source_encoded[:, :-1]` at `duplex_stt_model.py:2124`.

The interesting logic is the channel shifting:

| Knob | Effect | Where |
|---|---|---|
| `advance_text_channel_by` | text channel moves **earlier** — model commits to words before audio finishes | L124-166 |
| `delay_text_channel_by` | text channel moves **later** | L168-217 |
| `delay_text_eos_by` / `delay_text_bos_by` | move only the turn-boundary token, via `delay_eos()` at L22 | L220-224 |
| `delay_source_text_by` | delays the ASR channel only | L236-246 |
| `protect_prompt_from_shift` | keeps `[0:prompt_len]` fixed and shifts only content after it | L139-155 |

**The function channel is deliberately never shifted** (L161-166, L212-217), quoting the source:

> `DO NOT shift function calling channel - function calls must stay at their true timeline
> positions. Agent text has PAD tokens at function positions anyway, so no conflict.`

`delay_eos()` (L22) is worth reading once: it moves each EOS forward by `shift`, replacing the
original with PAD, and **skips** the move if it would go out of bounds or clobber another EOS.

### 4e. The function-calling channel — the genuinely unusual part

Two steps.

**Step 1: build the channel.** `_build_function_calling_channel` (`duplex_stt_model.py:936`)
collects events and wraps each in special tokens:

```python
L1015  wrapped_call = torch.cat([SOTC, call_tokens, EOTC])
L1020  events.append((call_step_adjusted, wrapped_call, True))   # True = compute loss
```

Special tokens, Nemotron-specific (L908-912):

| Token | Name | Meaning |
|---|---|---|
| `<SPECIAL_20>` | SOTC | Start Of Tool Call |
| `<SPECIAL_21>` | EOTC | End Of Tool Call |
| `<SPECIAL_22>` | EOTR | End Of Tool Response |

Calls get **loss enabled**; responses come from the API, so they are inserted with loss
**disabled** — the model should learn to emit calls, not to hallucinate results. Positions arrive
in original coordinates and get `prompt_offset` added (L1011, L1035) to land in prompt-inclusive
coordinate space.

**Step 2: open a hole in every channel.** `_expand_channels_with_insertions` (L1500):

```python
L1575  tokens  = cat([tokens[:pos],  PAD * len,  tokens[pos:]])     # agent text
L1579  silence = self._get_silence_embeddings(insert_length, ...)
L1580  encoded = cat([encoded[:pos], silence,    encoded[pos:]])    # user audio
L1584  src_tokens = cat([...,        PAD * len,  ...])              # ASR channel
```

So a tool call **stops the clock**: the sequence grows from `L` to `L+F`, and during those `F`
frames the user channel is fed *encoded actual silence* while the agent text channel is PAD. This
is why `max_fc_total_tokens` indirectly bounds sequence length — a tau2 cut costs roughly
`1842 prompt + 2632 audio frames + 3894 inserted ~= 8.4k` positions (yaml 105-122).

**The distributed subtlety** (L1539-1560) matters if you ever touch this code:

```python
# all ranks must call perception.forward() the SAME NUMBER OF TIMES to avoid NCCL deadlocks
max_insertions_tensor = ...; all_reduce(..., op=MAX)
for _ in range(dummy_calls_needed):
    dummy_silence = self._get_silence_embeddings(1, subsampling_factor)
```

Different ranks get different numbers of tool calls, so ranks with fewer make **dummy** silence
calls purely to keep the collectives aligned. The same defensive pattern recurs throughout
`training_step`: losses are multiplied by `0.0` for "minimal batches" (L2581-2585) rather than
skipped, and `self.log(..., sync_dist=True)` is called unconditionally when the function head is
active (L2660-2672). **Any early `return` you add to `training_step` will hang the job, not crash
it.**

#### Silence frame rate — a known benign artifact

`_ensure_silence_fps_initialized` (L1154) probes the encoder with a **1-second** clip — see the
source comment at L1383, *"only creates a 1-second template to compute the ratio (much faster than
60s)"* — and computes `frames_per_second = num_frames / duration_seconds` (L1315). Constant edge
overhead of ~1.5 frames makes a 1 s probe read **14.00** while the asymptotic rate is **12.5**
(a 60 s probe reads 12.525; verified against real data: 199.6 s / 0.08 = 2495 frames).

The value is used only to size the silence buffer in `get_silence_embeddings_from_ratio`
(`data/utils.py:177`): `seconds_with_buffer = seconds_needed * 1.1` (L211), so
`1.1 * (12.5/14) = 0.982` — the buffer lands ~1.8% short and the repeat safeguard at L253 fires
routinely instead of never. Output length is always exact via the trailing `[:length]` slice and
the content is silence either way, so training is unaffected. The inference path builds a separate
60 s template (L1203) and never sees this. **Do not "fix" this without measuring — the released
weights were trained under it.**

### 4f. Losses

All three heads share one shape: per-position cross-entropy, times a weight mask, summed, divided
by frame count (`duplex_stt_model.py:2366`, `:2375`, `:2462`):

```python
text_loss = (F.cross_entropy(text_logits.flatten(0, 1),
                             inputs["text_labels"].flatten(0, 1),
                             reduction="none")
             * inputs["loss_scale"][:, :, 0].flatten(0, 1)
            ).sum(-1) / num_frames
```

Combined at L2593-2602:

```python
loss = cfg.text_loss_weight     * text_loss                    # L2593
     + cfg.asr_loss_weight      * asr_loss        # L2596, only if compute_asr
     + cfg.function_loss_weight * function_loss   # L2602, only if function_labels present
```

then optionally blended with a pure-text objective (L2647):

```python
res["loss"] = (1 - text_to_text_loss_weight) * audio_loss
            +      text_to_text_loss_weight  * text_to_text_loss
```

`text_loss_weight` is a **relative** weight against `function_loss_weight`, not an absolute scale.
The 1.1B template pairs `text_loss_weight: 3` with `audio_loss_weight: 4`; copying the 3 without
the 4 gives the text channel 3x the gradient of the function channel — backwards for tau2, where
tool calls are the point. Ours is `1` / `1.0` (yaml 42-50).

#### `loss_scale` — where the real tuning lives

Built in `prepare_inputs` at L2225-2290 as nested `torch.where` over **label identity**. Both
weight dicts are gated by `if self.cfg.get(...)`, so **omitting them falls back to a uniform 1.0**
— catastrophic here. At 12.5 fps the overwhelming majority of frames are PAD, so uniform weighting
makes "stay silent" the loss-minimizing policy.

```yaml
token_loss_weight:          { bos: 12.5, eos: 7.5, pad: 1.0, text: 5.0 }
function_token_loss_weight: { call: 64.0, sotc: 6.0, eotc: 6.0, eotr: 3.0, pad: 0.3 }
```

`bos` is 12.5x because BOS/EOS **are** the turn-taking decisions. The function channel is more
extreme still — `call: 64.0` vs `pad: 0.3` is a **213x ratio** — because tool-call content occupies
a handful of frames per conversation. Leave these unset and `function_head` trains almost entirely
on PAD, i.e. it learns to emit nothing.

Two more masks stack on top:

- `seq_mask` (L2205-2213) zeroes everything past `target_token_lens`, with a `-1` correction when
  function calling expanded the sequence.
- The system-prompt region is explicitly zeroed (L2217-2224), also `-1`-corrected for the
  `[:, 1:]` label shift, so no loss is computed on the prompt regardless of `pad_weight`.

Auxiliary metrics logged alongside: `token_accuracy`, and `function_sotc_acc` /
`function_eotc_acc` / `function_eotr_acc` — per-special-token accuracies that are the fastest
signal that the FC channel is learning anything.

### 4g. Data loading

`DataModule` (`data/datamodule.py`) is thin — 175 lines delegating to
`get_lhotse_dataloader_from_config`. Two things matter:

- **Lhotse samplers control batching, not the Dataset.** `DuplexS2SDataset.__getitem__`
  (`s2s_dataset.py:1232`) receives a whole `CutSet`, not an index. This is why shard count caps
  data-parallel width: Lhotse hands **whole shards** to ranks, so N shards can feed at most N
  ranks.
- `_get_dp_rank` / `_get_world_size` (L150, L163) read the device mesh under model parallelism and
  fall back to plain `torch.distributed.get_rank()` under DDP.
- Validation and test configs are force-set to `force_finite = True` and
  `force_map_dataset = True` (L65-69), and multiple val sets are combined with
  `CombinedLoader(mode="max_size")` (L148).

`__getitem__` returns a nested dict:

```python
{"audio_data": {...}, "text_data": ..., "early_interruption_stats": ..., "mcq_delay_stats": ...}
```

where `audio_data` holds `source_audio`, `source_audio_lens`, `target_audio`, `target_tokens`,
`source_tokens`, `prompt_token_lens`, the `function_*` tensors, and `formatter` — a per-cut string
that gates which augmentations apply (e.g. `duplex_stt_model.py:1700`).

For the data format contract itself — what a cut must contain, the `supervisions[1]` gate, the
text-vs-function exclusivity rule — see the module docstring of
`scripts/repair_tau2_shards.py`, which re-implements the dataset's own preconditions as
standalone validation.

---

## 5. Where to look for X

| Question | File:line |
|---|---|
| What does one training step do? | `models/duplex_stt_model.py:2313` |
| How is a batch turned into embeddings? | `models/duplex_stt_model.py:1680` (`prepare_inputs`) |
| How are the 4 channels combined? | `parts/fusion.py:93` (`AddFusion.forward`) |
| Why did my sequence length change? | `models/duplex_stt_model.py:1500` (insertion expansion) |
| Where are tool-call special tokens defined? | `models/duplex_stt_model.py:908` |
| Why is my loss not moving? | `models/duplex_stt_model.py:2225-2290` (`loss_scale`) |
| What must my data contain? | `data/s2s_dataset.py:1586-1600` (FC gate), `:2274` (text/function exclusivity) |
| Which params actually train? | `parts/optim_setup.py:95-145` |
| How is LoRA attached? | `parts/lora.py:31` |
| Audio to embeddings | `modules/perception.py:104` |
| Which metrics exist to monitor? | `models/duplex_stt_model.py:2730` (`on_validation_epoch_end`) |
| Time-shift / turn-boundary logic | `parts/label_prep.py:64`, `:22` |
| Inference (a separate ~1300-line path) | `models/duplex_stt_model.py:3597`, `:3914`, `:4458` |
| Where do the pretrained weights come from? | §8, plus `scripts/extract_stt_checkpoint.py`, `scripts/remap_checkpoint_for_lora.py` |
| Did my checkpoint actually load? | `scripts/verify_checkpoint_load.py` (`FRESH_BY_DESIGN` at `:46`) |
| Where the RNNT head gets reattached | `parts/pretrained.py:156` ← `inference/model_wrappers/nemotron_voicechat_inference_wrapper.py:563` |

---

## 6. Things that will bite you

1. **The LLM never sees token IDs.** Anything assuming `input_ids` — `.generate()`, peft's
   `PeftModelForCausalLM`, HF-computed loss — is wrong here. Hence our yaml deliberately omits
   `task_type: CAUSAL_LM` (yaml 137-143): peft's causal-LM wrapper reads
   `base_model.prepare_inputs_for_generation`, which exists on `NemotronHForCausalLM` but not on
   the bare `NemotronHModel` we assign to `self.llm`, and raises `AttributeError`.

2. **`cfg.get(...)`-gated keys fail silently, not loudly.** `duplex_function_channel_weight`,
   `token_loss_weight`, `function_token_loss_weight`, `augment_fc_system_prompt`,
   `max_fc_total_tokens` all have defaults that differ from what the released 11B was trained
   with. Omitting one does not error — it changes the architecture or the objective quietly. Every
   comment block in our yaml is one of these.

3. **`training_step` is a distributed minefield.** Never add an early `return`; never make a
   `self.log` conditional on batch content. The code multiplies losses by `0.0` rather than
   skipping them (L2581) for exactly this reason. Symptom of getting it wrong is a hang, not a
   traceback.

4. **`trainer.limit_train_batches` / `val_check_interval` count dataloader BATCHES; `max_steps`
   and `every_n_train_steps` count OPTIMIZER steps.** With `accumulate_grad_batches: 4` they
   differ 4x. Validation here is autoregressive generation through the no-cache Nemotron path at
   ~16 s per sample, so a mis-set interval can put most of the run inside validation.

5. **Monitor a metric that is actually logged.** `val_asr_bleu` does not exist in this model — it
   is a DuplexS2S/EAR-TTS metric. `on_validation_epoch_end` (L2730) logs `val_txt_bleu`,
   `val_txt_bleu_after_tool`, `val_txt_bleu_tool_call` and turn-taking metrics; the `val_src_*`
   ASR metrics only exist when `predict_user_text: true`. Monitoring a missing key makes
   `ModelCheckpoint` raise at the *first validation*, i.e. an hour into the run.

6. **The `trainable=100.00%` startup line is a counting artifact.**
   `parts/optim_setup.py:133` counts params it *yields to the optimizer*, ignoring `requires_grad`
   that peft already set, and yields everything because no `freeze_params` regex matched anything
   with parameters; the log line is L138. Trust Lightning's
   `2.4 B Trainable / 7.7 B Non-trainable` summary. Likewise the
   `freeze-preventing patterns UNMATCHED ['^.+\.lora_.+$']` warning (L146) is a short-circuit
   artifact — L128 is `if _exclude(name) and not _must_keep(name)`, so `_must_keep` is never called
   when nothing matched `_exclude` and its counter stays 0 — not a sign LoRA is missing. That
   pattern is appended at runtime by `parts/lora.py:31`, which is also why the saved config shows
   `prevent_freeze_params: []`.

7. **`gate_proj` in our LoRA `target_modules` matches nothing.** Nemotron-H's MLP is up/down only.
   peft raises only if *no* target matches, so a partially-dead target list is accepted silently.
   Related: the 27 Mamba layers' `in_proj`/`out_proj` are not adapted at all, so LoRA touches 29
   of 56 layers (`q,k,v,o_proj` on the 4 attention layers, `up/down_proj` on the 25 MLP layers).

---

## 7. The model family — `DuplexSTTModel` vs `DuplexS2SModel`

`nemo/collections/speechlm2/models/__init__.py` exports six models. They are siblings, not a class
hierarchy — each is a standalone `LightningModule`, and the shared machinery lives in
`parts/` and `modules/` rather than in a base class.

```
speech in ──┬── SALM                        offline speech understanding, no timeline
            │
            └── duplex (frame-synchronous, 12.5 Hz shared timeline)
                 ├── DuplexSTTModel              speech in  -> TEXT out   (+ function channel)
                 ├── DuplexS2SModel              speech in  -> text + AUDIO CODES out
                 ├── DuplexS2SSpeechDecoderModel  as above, separate speech decoder
                 ├── DuplexEARTTS                 text in    -> AUDIO out
                 └── NemotronVoiceChat            = DuplexSTTModel + DuplexEARTTS  (§8)
```

`DuplexSTTModel` is the one we finetune. The most instructive comparison is with `DuplexS2SModel`
(697 lines vs our 5350), because they solve the same duplex problem with a different output
modality, and the differences are exactly the interesting parts.

| | `DuplexS2SModel` | `DuplexSTTModel` |
|---|---|---|
| Output modality | text **+ speech codes** | text only |
| Extra modules | `audio_codec`, `embed_audio_tokens` (K embeddings), `audio_head` | `function_head`, optional `asr_head`, pluggable `fusion_module` |
| Channel fusion | hardcoded in-place `.add_()` | `create_fusion_module()` factory, 4 strategies |
| Loss | uniform CE, `reduction="sum"` | per-token-class weighted CE via `loss_scale` |
| Length misalignment | **truncate to shortest** | **insertion expansion** `L -> L+F` |
| Tool calls | none | the function channel (§4e) |

**The audio output path** (`duplex_s2s_model.py:59-86`) is what STT does not have:

```python
setup_audio_codec(self)                                                              # :59
self._codebook_size = self.audio_codec.vector_quantizer.codebook_size
self._num_codebooks = self.audio_codec.vector_quantizer.num_groups
...
self.embed_audio_tokens = torch.nn.ModuleList(                                       # :80
    [torch.nn.Embedding(self.speech_vocab_size, self.embed_tokens.embedding_dim)
     for _ in range(self._num_codebooks)]
)
self.audio_head = torch.nn.Linear(                                                   # :86
    self.llm.config.hidden_size, self.speech_vocab_size * self._num_codebooks)
```

Speech control tokens are **carved out of the codebook** rather than added to a text vocabulary
(`:103-115`): `speech_bos_id = codebook_size`, `speech_eos_id = +1`, `speech_delay_id = +2`. Targets
are produced on the fly at `:215` — `with fp32_precision(), torch.no_grad(): self.audio_codec.encode(...)`.

**S2S fusion is in-place and order-sensitive** (`:277-283`), with a load-bearing comment:

```python
# Note: the order of addition should be consistent with inference code due to
#       a low numerical precision, i.e.: Input speech + (Output text + Output speech)
#       Remember that addition is not associative in low precision floating point!
input_embeds = self.embed_tokens(text_inputs)
for cbidx in range(self._num_codebooks):
    input_embeds.add_(self.embed_audio_tokens[cbidx](audio_inputs[..., cbidx]))
input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))
```

Our `AddFusion.forward` (`parts/fusion.py:93`) is the same idea factored out, which is what makes
`fuse_method: gated_gmu` a config change rather than a rewrite.

**The alignment philosophies are opposite.** Both models face the problem that the audio grid and
the text/codes grid don't have the same length. S2S truncates to the shortest and warns if the gap
is large (`:224-238`, `if diff > 2: logging.warning(...)`). STT *expands* the timeline to make room
for tool-call tokens, and must then re-align every other channel to match — see §4e. Truncation is
cheap and lossy; insertion is exact and is why the STT file is 8× longer.

**Loss shape.** S2S (`:301-314`) is `loss = cfg.text_loss_weight*text_loss + cfg.audio_loss_weight*audio_loss`
over `reduction="sum"` CE — every token weighted equally. STT weights per token *class*
(§4f), which is why `token_loss_weight` / `function_token_loss_weight` exist at all and why
omitting them silently changes the objective (§6.2).

Practical consequence: **`audio_loss_weight: 0` in our yaml is not disabling anything** — there is
no audio head in `DuplexSTTModel`. It is inherited vestigially from the S2S config schema. Equally,
S2S/EAR-TTS metric names like `val_asr_bleu` do not exist here (§6.5).

---

## 8. The released checkpoint — and how the training stack is reconstructed

### 8a. The release is one flat state dict with two prefixes

`/fsx/home/kai.li/data/voicechat/voicechat-11b/model.safetensors` is 44,382,749,892 bytes holding
**1632 tensors, all fp32** (plus 6 int64 buffers). Every key begins with one of exactly two
prefixes, and there are **no unprefixed keys**:

| prefix | tensors | elements | bytes when split out |
|---|---|---|---|
| `stt_model.*` | 997 | 10.0981 B | 40,392,680,692 |
| `tts_model.*` | 635 | 0.9968 B | 3,990,051,120 |

The two sum to 44,382,731,812 — the release to within 18,080 bytes, which is just the safetensors
JSON header (the release carries 1632 extra 10-character key prefixes = 16,320 bytes of key text).
**The "composition" is purely a naming convention.** There is no per-model file, no shard index, no
adapter layered over a base. `NemotronVoiceChat` is a `LightningModule` holding two child modules
(`nemotron_voicechat.py:200`, `:213`), so PyTorch's `state_dict()` prefixes their keys with the
attribute names. That is the entire mechanism.

Inside each half, after prefix-stripping:

```
stt_extracted/      997  perception 640 | llm 339 | embed_tokens 1 | lm_head 1
                         | function_head 1 | rnnt_decoder 9 | rnnt_joint 6
stt_extracted_tts/  635  tts_model 418 | audio_codec 214 | _control_codes 1
                         | codec_silence_tokens 1 | audio_prompt_latents 1
```

(the TTS half still shows a `tts_model.` prefix because `DuplexEARTTS` has its own child of that
name — original keys were `tts_model.tts_model.*`.)

Note what is **absent** from the STT half: no `asr_head`, no `embed_asr_tokens` — consistent with
the release having trained `predict_user_text: false` / `use_separate_asr_head: false` — and no
`audio_codec`, which lives entirely in the TTS half (`pretrained_audio_codec: null`).

### 8b. There is a third thing in there, belonging to neither model class

The 15 `rnnt_decoder.*` / `rnnt_joint.*` tensors (8.94 M params) are **not** constructed by
`DuplexSTTModel.__init__`. They were grafted on after training by
`examples/speechlm2/combine_s2s_rnnt_checkpoint.py`, which also writes the `rnnt_tokenizer/`
directory and appends the `_rnnt_merge_info` block to `config.json` (`decoder_config`,
`joint_config`, class paths, `rnnt_vocab_size: 1024`). `parts/pretrained.py:149` says so:

> They are loaded separately at inference time via `setup_rnnt_from_combined_checkpoint`.

That function's only caller is
`inference/model_wrappers/nemotron_voicechat_inference_wrapper.py:563`. So the release is really
**STT + TTS + a bolted-on streaming-ASR head that only the inference wrapper knows how to
reattach.** For training those 15 tensors are dead weight, which is why
`remap_checkpoint_for_lora.py` reports `dropped (no home in model): 15`.

### 8c. `config.json` is not a model config — it is the whole training yaml, recursively nested

The 31,984-byte `config.json` has top-level keys `data`, `exp_manager`, `model`, `trainer`,
`_rnnt_merge_info`. And `model.stt` *itself* has keys `data`, `exp_manager`, `model` — a complete
nested run config. Same for `model.speech_generation`. That is because every model in this
collection eats a **whole run config**, not a model section
(`nemotron_voicechat.py:194`, `duplex_stt_model.py:189`):

```python
cfg = DictConfig(cfg)
self.cfg = cfg.model
self.target_sample_rate = cfg.data.target_sample_rate
self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, ...)
```

So composing two models means nesting two whole run configs.
`examples/speechlm2/conf/nemotron-labs-voicechat.yaml` has the identical shape — the released
`config.json` is that file after training, with `${}` interpolations resolved and runtime-derived
values baked in.

The baking matters. The released STT config has 58 keys ours lacks, and nearly all are
`perception.encoder.*` plus `perception.output_dim: 4480`. Ours omits them because
`setup_speech_encoder` overwrites `cfg.perception.encoder` from the ASR `.nemo` at load time and
`parts/pretrained.py:137` sets `perception.output_dim = model.llm.config.hidden_size`. The release
has `pretrained_asr: ''` precisely *because* the encoder config is already materialized — it no
longer needs the `.nemo`.

### 8d. Reconstruction: three legs, only one of which is the checkpoint

```
voicechat-11b/model.safetensors        1632 tensors — stt_model.* + tts_model.*
  │
  ├─ scripts/extract_stt_checkpoint.py            strip the "stt_model." prefix
  │     └─> stt_extracted/                        997 tensors, 40.4 GB
  │
  ├─ scripts/remap_checkpoint_for_lora.py         rename keys for the peft wrapper
  │     │   (instantiates DuplexSTTModel from OUR yaml to learn the key names,
  │     │    so whatever peft does is reflected automatically)
  │     └─> stt_extracted_lora/                   984 tensors, 45.1 GB
  │
  ▼
DuplexSTTModel(cfg).load_state_dict(...)
  ▲                        ▲
  │                        └── LEG 3  hyperparameters: our s2s_duplex_stt_11b.yaml
  └── LEG 1  architecture: upstream duplex_stt_model.py, rebuilt from scratch

     LEG 2  weights: 10.089 B params from the release
                   +   36.1 M LoRA adapters, fresh by design
```

**Leg 1 — architecture from code.** `__init__` rebuilds every module from scratch (§4a): pulls
Nemotron Nano 9B v2 from HF, `delattr`s its embedding table, builds `perception` from the ASR
`.nemo`, `deepcopy`s `lm_head` into `function_head`, installs LoRA. Nothing about the model's
*shape* comes from the checkpoint.

**Leg 2 — weights by exact name match.** Two mechanical rewrites stand between the release and a
loadable checkpoint:

- `scripts/extract_stt_checkpoint.py` strips the `stt_model.` prefix → `stt_extracted/`
  (997 tensors, 40.4 GB). It nests the original config untouched under
  `{"source": "extracted_from_nemotron_voicechat_11b", "original_config": ...}` — it does **not**
  split the config.
- `scripts/remap_checkpoint_for_lora.py` renames for peft → `stt_extracted_lora/` (984 tensors,
  45.1 GB). It gets *bigger* because `--warm-start-asr-head` copies `lm_head`/`embed_tokens` into
  `asr_head`/`embed_asr_tokens` (2 × 131072 × 4480 fp32 = 4.70 GB); it drops the 15 RNNT tensors.
  Result: 66 keys carry peft's `.base_layer.` infix, all 339 LLM tensors sit under
  `llm.base_model.`.

  **This is a naming problem, not a compatibility problem.** Verified: across all 982 shared
  tensors the shapes are identical, zero mismatches. Only the key string changes:

  ```
                      extracted from release             what our LoRA model expects
  non-targeted param  llm.layers.0.mixer.in_proj.weight  llm.base_model.model.layers.0.mixer.in_proj.weight
  LoRA-targeted       llm.layers.14.mixer.q_proj.weight  llm.base_model.model.layers.14.mixer.q_proj.base_layer.weight
  non-LLM param       lm_head.weight                     lm_head.weight        (unchanged)
  ```

  Two independent transforms, both from peft: the **prefix** grows on all 339 LLM tensors because
  `get_peft_model` returns a `PeftModel` whose `.base_model` is a `LoraModel` whose `.model` is the
  original (two extra levels of module path); the **infix** appears on the 66 targeted projections
  because each `nn.Linear` is replaced by a `peft.tuners.lora.Linear` holding the original as its
  `.base_layer` child. Those two rules are exactly `PEFT_RENAMES` in
  `scripts/verify_checkpoint_load.py:40-43`.

  Note whose problem this is: the release was a **full finetune with no LoRA** — its config has no
  `lora` key — so its names are correct for its own model definition. The divergence appears only
  because *our* yaml adds `lora:`. Remove that block and `stt_extracted/` loads directly.

The parameter accounting closes **exactly**:

```
stt_extracted elements                     10,089,196,944   (after dropping 8.94 M rnnt)
  minus preprocessor buffers (fb, window)         -33,296   (buffers, not parameters)
  = parameters recoverable from the release   10,089,163,648
model as instantiated                          10,125,286,272
  gap                                              36,122,624   == the LoRA adapters, exactly
```

That 36,122,624 is the same number measured independently for the 66 LoRA modules (§6.7). So
**every parameter in our model except the LoRA adapters comes from the released checkpoint**, with
`asr_head`/`embed_asr_tokens` warm-started from their text-channel twins rather than pretrained.
The adapters are fresh by design — `FRESH_BY_DESIGN` in `scripts/verify_checkpoint_load.py`
excludes them from the mapping.

The census is exhaustive, which is what rules out unaccounted-for weights. Every parameter in the
instantiated model is in exactly one of these buckets:

| bucket | params | source |
|---|---|---|
| loaded from the release | 10,089,163,648 | the `stt_model.*` half, after remap |
| LoRA adapters | 36,122,624 | fresh by design — they are new |
| `asr_head` / `embed_asr_tokens` | 0 | **not built at all** (`predict_user_text: false`) |
| `perception.preprocessor` buffers | 33,296 | buffers, not parameters; from the ASR `.nemo` |
| **total** | **10,125,286,272** | = the instantiated model, exactly |

On the checkpoint side: 997 tensors = 982 loaded + 15 RNNT discarded. Both directions close with
zero remainder.

`scripts/verify_checkpoint_load.py` is what proves this rather than assuming it, because it checks
**both** directions — including the reverse check the training loader omits (§8e):

```python
model_only      = [k for k in model_sd if k not in matched_model_keys]
model_only_real = [k for k in model_only if not FRESH_BY_DESIGN.search(k)]   # the real gaps
model_only_lora = [k for k in model_only if     FRESH_BY_DESIGN.search(k)]   # expected
```

`model_only_real` *is* the set of silently-uninitialized parameters. It also catches shape
mismatches and prints `Pretrained coverage: X / Y (Z%)`. The verdict
`[OK] Every non-LoRA model parameter is covered by the checkpoint, shapes agree.` means that list is
empty. It deliberately does **not** count `ckpt_only` tensors (the 15 RNNT ones) as a failure —
`:173-176` explains why. It is CPU-only: no GPU needed, but it does instantiate the full model.

**Leg 3 — hyperparameters from our yaml.** This is the leg that can silently break the
reconstruction, because a config value that changes *behaviour* rather than *shape* loads 100% of
the weights and still trains the wrong model. Diffed against the release's `model.stt.model`, our
yaml is **identical on every architectural and data-semantics key**:

| key | value (release == ours) |
|---|---|
| `pretrained_llm` / `base_model_name` / `embed_tokens_name` / `cache_key` | Nemotron-Nano-9B-v2 / `backbone` / `embeddings` / `cache_params` |
| `duplex_function_channel_weight` | **2.0** (NeMo's default is 1.0 — this one is architectural) |
| `duplex_user_channel_weight` / `duplex_text_channel_weight` | 1.0 / 1.0 |
| `token_loss_weight` | `{bos: 12.5, eos: 7.5, pad: 1.0, text: 5.0}` |
| `function_token_loss_weight` | `{call: 64.0, sotc: 6.0, eotc: 6.0, eotr: 3.0, pad: 0.3}` |
| `use_function_head` / `augment_fc_system_prompt` / `predict_user_text` | true / true / false |
| `text_loss_weight` / `function_loss_weight` / `audio_loss_weight` | 1 / 1.0 / 0 |
| `modality_adapter` / `encoder.att_context_size` | `IdentityConnector d_model=1024` / `[70, 0]` |
| `scoring_asr` | `nvidia/parakeet-tdt-1.1b` |

The **only** value divergences are optimizer, schedule, and plumbing:

| key | released | ours | why |
|---|---|---|---|
| `optimizer.lr` | 5e-5 | 1e-4 | short finetune |
| `lr_scheduler._target_` | InverseSquareRootAnnealing | CosineAnnealing | short finetune |
| `lr_scheduler.warmup_steps` | 2500 | 200 | short finetune |
| `optimizer.weight_decay` | 0 | 0.01 | our choice |
| `max_fc_total_tokens` | 3000 | **8000** | tau2 tool schemas are long |
| `pretrained_weights` | false | true | release resumed from ckpt; we bootstrap from HF |
| `pretrained_asr` | `''` | `nemotron-speech-streaming-en-0.6b` | release had the encoder cfg baked in (§8c) |
| `freeze_params` | `['^audio_codec\..+$']` | + `^perception\.preprocessor\..+$` | ours is a no-op (buffers only, §6.6) |
| `lora.*`, `fuse_method`, `predict_user_text_prob` | absent | present | the release was a full finetune, not LoRA |

`max_fc_total_tokens: 8000` is the one deliberate behavioural divergence.

### 8e. Two ways this reconstruction fails silently

Neither raises. Both matter because the loader (`:333-350`) iterates over **checkpoint** keys, not
model keys:

```python
for key in f.keys():
    if key in model_state_dict:
        model_state_dict[key].copy_(f.get_tensor(key))   # exact string match, no normalization
        loaded_keys.append(key)
    else:
        missing_keys.append(key)
logging.info(f"Loaded {len(loaded_keys)} tensors from pretrained model")
if missing_keys:
    logging.warning(f"Keys in checkpoint but not in model: {len(missing_keys)} keys")
```

The warning reports the **checkpoint's** orphans. The reverse direction is never checked, so a model
parameter that receives no weights produces **no log line at all**.

1. **Skip the remap** and all 339 LLM tensors miss, because peft renamed them (§8d). Measured:
   **643 of 997** checkpoint keys match, and the subtraction has two distinct causes —

   ```
   997 checkpoint tensors
    -339  LLM tensors renamed by peft   -> recoverable: the remap fixes it
    - 15  rnnt_decoder / rnnt_joint     -> structural: no such module exists, correctly discarded
    =643  match without the remap
   ```

   The only symptom is that low `Loaded N tensors` count plus
   `Keys in checkpoint but not in model: 339 keys` — and you have to know that 339 is exactly the
   LLM to read anything into it.

   What you then finetune is a mismatched hybrid: `perception`, `embed_tokens`, `lm_head` and
   `function_head` from VoiceChat, but the 7.75 B backbone from **base Nemotron-Nano-9B-v2**. Our
   yaml sets `pretrained_weights: true`, so `parts/pretrained.py:62` takes the `from_pretrained`
   branch — these are base weights, not random ones (random is the `pretrained_weights: false`
   branch; the only truly random parameter in the skip-the-remap case is `perception.proj`, which
   has no HF source). A text LLM that has never seen a fused speech embedding or a 12.5 Hz timeline.

   **You cannot detect this by reading the output.** Verified empirically: a load-nothing run still
   wrote polished prose — it simply ignored the audio and free-associated on the system prompt. The
   reliable tell is the **absence of turn-taking timestamps** (`<$0.72$> <|2.08|>`) in
   `scripts/smoke_test_inference.py` output; those tokens only exist in the VoiceChat finetuning, so
   they are the signal that the released weights are actually live.
2. **`__init__` ordering.** `copy.deepcopy(self.lm_head)` runs at `:275` (`asr_head`) and `:307`
   (`function_head`) — *before* the checkpoint load at `:321`. A head absent from the checkpoint
   therefore keeps its base HF weights rather than erroring: exactly the situation `asr_head` is in
   without `--warm-start-asr-head`.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **frame** | one 0.08 s slot on the shared timeline; the sequence axis of the model |
| **channel** | one parallel token/embedding stream along that timeline (agent text, user audio, user text/ASR, function) |
| **duplex** | user and agent occupy the same timeline simultaneously, rather than alternating turns |
| **SOTC / EOTC / EOTR** | Start Of Tool Call / End Of Tool Call / End Of Tool Response — `<SPECIAL_20..22>` |
| **insertion** | how the function channel is added: the sequence *expands* `L -> L+F`; it does not overwrite the audio grid |
| **perception** | the speech encoder stack: preprocessor → encoder → modality adapter → proj |
| **Shar** | Lhotse's sharded container format; one `cuts.*.jsonl.gz` + one tar per audio field per shard |
| **cut** | one conversation, in Lhotse terms |
| **supervision** | one turn within a cut, carrying `speaker`, `text`, `start`, `duration`, `custom` |
| **formatter** | per-cut string tag that gates which augmentations apply |
