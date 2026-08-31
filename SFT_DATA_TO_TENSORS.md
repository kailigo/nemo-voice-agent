# From a collected trajectory to SFT tensors

How a multi-turn, multi-hop duplex conversation with function calls turns into the tensors
`DuplexSTTModel` actually trains on. Read out of the code at `a30b124fa` (2026-08-31), not
inferred — see the file:line citations. For how to *launch* a training run, see
`FINETUNING_11B.md`; for the Lhotse Shar data contract a converter must produce, see the
`duplex-stt-training-data-contract` memory and `tau-voice-2/scripts/episodes_to_nemotron_training.py`
(the reference converter). This doc is about what happens to that data once it reaches the model.

## 1. The raw data unit: one Lhotse cut per conversation

Each cut holds the **whole multi-turn conversation on one shared timeline**, not per-turn
examples:

- `cut.recording` — user audio, spans the full conversation, 16 kHz (`data.source_sample_rate`)
- `cut.custom['target_audio']` — agent audio, same span, 22.05 kHz (`data.target_sample_rate`)
- `cut.supervisions` — a list of role-labeled segments (`system`, `user`, `agent`) with
  start/duration, each carrying text and/or a `custom['function']` string

So a conversation with several tool calls across several turns is *one* cut with many
supervisions at different timestamps. There is no per-turn chunking anywhere in the pipeline.

## 2. Everything lives on a common frame grid

The perception encoder subsamples user audio to **12.5 frames/sec** (`frame_length: 0.08` in
`data`). Every channel — agent text, user/ASR text, function-calling — is a `[B, T]` tensor of
token IDs aligned to that same frame axis, filled with `text_pad_id` almost everywhere, with a
real token dropped in at the frame where a supervision starts.

A spoken turn's text is placed starting at `supervision.start`; its available slot width is
`duration / 0.08` frames. If the text doesn't fit that span it is truncated **with no log line**
in the common case — a real, measured lossy edge case (see trap 7 in the data-contract memory:
~8% of target text tokens deleted across a past 900-label sample). This is a property of the
*converter* that builds the cut, not of the model-side code below, but it matters for anyone
authoring new-domain data: widen the span to fit, bounded by the next same-role turn.

## 3. Function calls get their own dedicated frames — sequence length actually expands

This is the part that matters most for multi-hop tool use. A call/response is not shoehorned
into the speech-frame budget; the model builds a separate **function channel** and *inserts*
tokens into it, expanding the whole batch from length `L` to `L+F`.

Built in `DuplexSTTModel._build_function_calling_channel`
(`nemo/collections/speechlm2/models/duplex_stt_model.py:1085`):

- A call becomes `<SOTC> {call tokens} <EOTC>`, placed at its recorded step, **with loss
  enabled** (`:1169`) — the model must learn to produce this.
- A response becomes `{response tokens} <EOTR>` at its step (`:1182-1197`) — but the response
  content itself has **loss disabled** (`:1189`, `compute_loss=False`; it's copied from the API,
  not generated), while the `<EOTR>` marker has loss enabled (`:1193`) — the model must learn
  *when the tool result ends*.
- PAD frames between events also get loss enabled (`:1210`, `:1227`, comment: "prevent
  hallucination") — the model is trained to predict silence during quiet stretches too, not just
  let off the hook there.

Events are **not sorted by timestamp** — they're processed in call order per turn
(`for turn_idx in range(num_turns)`, `:1153`), because two calls can legitimately land on the
same tick and sorting by timestamp would silently swap a call for a response. This is trap 6 in
the data-contract memory and it is a real, previously-hit bug class: pair by list position, never
re-sort.

Every insertion point is then mirrored into the **other** channels at the same position, in
`_expand_channels_with_insertions` (`:1649`):

- agent-text channel gets PAD inserted (`:1723-1724`)
- user-audio channel gets *real encoded-silence embeddings* inserted (`:1728-1729`) — not zeros;
  an actual silence recording run through the perception encoder
- ASR/user-text channel gets PAD inserted too, if present (`:1732-1733`)

All channels end up the same expanded length `L+F`, frame-for-frame aligned.

**One real trap worth knowing when authoring new-domain data**: if an agent turn *both* speaks
("Let me check that...") *and* calls a tool, that must be **two separate supervisions** — a
text-only one and a function-only one. The converter drops a turn from the text channel entirely
if it has any `custom['function']` content (data-contract memory, trap 1). Merge them into one
supervision and the spoken acknowledgment silently vanishes from training with no error.

## 4. Turning the frame tensors into model input

In `DuplexSTTModel.prepare_inputs`
(`nemo/collections/speechlm2/models/duplex_stt_model.py:1829`):

- `self.perception(user_audio) → source_encoded` — the audio channel, frame-aligned (`:1940`).
- `self.embed_tokens(agent_text_channel)` and `self.embed_asr_tokens(user_text_channel)` — text
  channels, embedded per frame (`:2262`, `:2267`).
- A **fusion module** (`nemo/collections/speechlm2/parts/fusion.py`) combines audio + agent-text
  + user-text (+ function, via channel embeddings) into one `input_embeds` per frame — this is
  the actual tensor fed to the LLM.
- **Causal shift for next-frame prediction**, in `prepare_labels`
  (`nemo/collections/speechlm2/parts/label_prep.py:64`): `text_inputs = target_tokens[:, :-1]`,
  `text_labels = target_tokens[:, 1:]` (`:316-317`, or `:256-257` in the `predict_user_text`
  branch). Standard teacher-forced LM training, just applied to a mostly-PAD frame sequence
  instead of a dense text sequence. The same shift is applied to the function channel
  (`:286-288`, `:335-337`) — note the function channel is deliberately **not** shifted by
  `advance_text_channel_by`/`delay_text_channel_by` (`:161-166`, `:212-217`): those knobs retime
  the agent *speech* channel relative to audio, but function calls must stay at their true
  timeline positions regardless.

## 5. Three independent losses, each with its own mask

In `DuplexSTTModel.training_step`
(`nemo/collections/speechlm2/models/duplex_stt_model.py:2462`):

- `text_loss` — agent text channel, cross-entropy against `text_labels` (`:2515`).
- `asr_loss` — user/ASR channel, cross-entropy against `asr_labels` (`:2524`), scaled by
  `asr_loss_scale` (`:2380-2430`). Computed **probabilistically per batch**
  (`predict_user_text_prob`, `:2481-2496`), not every step.
- `function_loss` — function channel, cross-entropy against `function_labels`, scaled by
  `function_loss_scale` (built from `function_loss_mask`, `:2433-2438`). This mask is what turns
  off gradient on the copied-in tool-response text while keeping it on for the call itself, the
  `<EOTR>` marker, and the PAD regions.

The three losses are combined with configurable weights (`text_loss_weight`, `asr_loss_weight`,
`function_loss_weight` in the training config); `audio_loss_weight: 0` for this STT-only variant
— there is no codec/TTS channel or loss at all (see `voicechat-fusion-weights-are-architectural`
memory for why the *fusion* weights, unlike the loss weights, are not just tunable hyperparameters).

## Summary of the path

```
one long multi-turn Lhotse cut
  -> frame-aligned multi-channel tensors (12.5 Hz)
  -> function calls carved into their own expanded, position-tracked channel (L -> L+F)
  -> silence/PAD mirrored into the other channels at the same insertion points
  -> perception + embedding + fusion -> input_embeds
  -> causal shift (:-1 / 1:) for next-frame prediction
  -> three masked cross-entropy losses (text, asr, function), summed with configurable weights
```

## Open questions for anyone authoring new-domain trajectories

These are the traps that have actually bitten this project before (data-contract memory has the
full list of seven); the ones most relevant to *new* domains:

1. Emit two supervisions for any agent turn that both speaks and calls a tool.
2. Never sort function call/response events by timestamp — preserve call order.
3. Every supervision needs `custom = {"function": ...}` (even `""` for plain speech), never
   `None` — the converter crashes rather than degrading gracefully if it's missing.
4. Widen a turn's timing span if its text won't fit at 12.5 tokens/sec, rather than letting it
   silently truncate.
5. Keep the shard directory free of any file that isn't a Shar field — a stray `metadata.json`
   inside the shard dir gets misparsed as a phantom Lhotse field.
