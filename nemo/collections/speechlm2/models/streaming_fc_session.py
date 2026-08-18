# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stepwise streaming driver for :class:`DuplexSTTModel` with *live* function calling.

Why this exists
---------------
``DuplexSTTModel`` ships two inference entry points and neither can drive an
interactive session:

* ``offline_inference`` needs the whole user waveform up front, and its function
  channel is built by ``_expand_for_function_calling``, which precomputes the entire
  expanded timeline from **ground-truth** call/response lengths. There is no ground
  truth in an interactive session.
* ``online_inference`` *looks* like a streaming loop but is not one. Two problems:

  1. It writes its freshly-encoded causal window into
     ``inference_state["input_embeds"][:, t]`` (``duplex_stt_model.py:5167``), but
     ``_step_inference`` **overwrites** that same slot with the fusion output at
     ``:3969`` before reading it at ``:3981``. The audio the LM actually consumes comes
     from ``inference_state["audio_embeds"][:, t]`` (``:3952``), which ``online_inference``
     never touches. So the per-step encoder calls are discarded work, and the model is
     silently fed the **full non-causal encode** that ``_init_inference`` produced --
     i.e. ``online_inference`` leaks future audio and is not causal at all. Its
     docstring's claim that it updates ``asr_emb`` is likewise not implemented.
  2. It has no function-calling seam whatsoever. (``_prepare_function_responses_for_detection``
     at ``:4042`` looks like one, but its ``response_queue`` is never consumed by any
     caller -- it is an unused stub.)

This module supplies the missing driver. It is **purely additive**: it calls the
model's existing ``_init_inference`` / ``_step_zero`` / ``_step_inference`` helpers and
mutates only its own ``inference_state`` dict, so the training and offline paths are
untouched.

The two-clock design
--------------------
The offline path expands the timeline because inserted function tokens need LM
positions that no audio frame occupies. We get the same effect without knowing the
expansion in advance by decoupling two counters:

``t_lm``
    Next LM position to step. Advances on **every** ``_step_inference`` -- audio frames
    *and* injected function tokens.
``t_audio``
    Audio frames consumed. Advances only as ticks deliver audio.

``t_lm - t_audio`` is the accumulated function-token insertion, and it is allowed to
grow without bound (up to the preallocated horizon). Because the horizon is
preallocated by running the perception encoder over a zero waveform, every
not-yet-written position already holds a genuine **silence** embedding -- which is
exactly what ``_expand_for_function_calling`` puts at inserted positions. Injection
therefore costs no extra embedding work: write the token into ``gen_function`` and step.

Function-channel mechanics
--------------------------
The function channel is *frame-synchronous*: one ``gen_function[:, t]`` token per LM
position, not a sequence splice. ``_step_inference`` reads ``gen_function[:, t-1]`` as
input (``:3955``) and writes its prediction into ``gen_function[:, t]`` (``:4001``)
**unless** that slot is already non-PAD::

    already_injected = (gen_function[:, t] != text_pad_id)
    should_predict   = ~is_prompt_position & ~already_injected

So "injecting" a tool response is just: write the response token ids into the upcoming
positions before stepping over them. The ``already_injected`` gate preserves them, and
they are consumed as input on the following step. This matches the training layout built
by ``_build_function_calling_channel`` (``:1013-1044``)::

    ... PAD ...  <SOTC> call_tokens <EOTC>  response_tokens  <EOTR>  ... PAD ...
                 |----- model generates ----| |-- we inject --| |-- model generates

``<EOTR>`` is loss-bearing in training, so we let the model emit it rather than forcing
it (see ``force_eotr``).

The KV cache: there isn't one, and that is correct
--------------------------------------------------
``_init_inference`` sets ``use_cache = False`` and ``cache = None`` whenever the LLM name
contains ``Nemotron`` (``:3719``). That looks like a stub worth overriding -- every step
then re-runs the full prefix, which is O(T^2) -- but it is the only correct setting
available through ``DuplexSTTModel.forward`` today. Measured on the real checkpoint
(2026-08-17), there are three independent blockers, in increasing order of severity:

1. ``DynamicCache`` is the wrong type. Nemotron-Nano-9B-v2 is a hybrid Mamba2/attention
   model (``modeling_nemotron_h.py``); its mixer needs the conv and SSM state that only
   ``HybridMambaAttentionDynamicCache`` carries. Forcing a ``DynamicCache`` dies with
   ``AttributeError: 'DynamicCache' object has no attribute 'conv_kernel_size'``.
2. That class does not actually work either. It never assigns ``self.conv_kernel_size``
   (only the mixer does, ``:294``), yet ``:461``/``:546`` read it off the cache; and
   ``update_conv_state(cache_init=True)`` dereferences ``self.conv_states.device`` on a
   *list*. Both are bugs in the checkpoint's remote code, on the prefill-with-cache path.
3. The decisive one: incremental decode is unreachable from here. The Mamba mixer's
   single-step branch is gated on ``cache_position[0] > 0`` (``:375``), and
   ``NemotronHModel.forward`` defaults ``cache_position`` to ``arange(seq_len)`` (``:1359``)
   when the caller passes none. ``DuplexSTTModel.forward`` (``:466``) has no
   ``cache_position`` parameter at all, so a one-position step always arrives as
   ``cache_position=[0]`` and every layer takes the *prefill* branch, treating the token as
   a fresh sequence and discarding the SSM state. It would not error; it would be silently
   wrong.

So the driver does not force the cache on. Making it work means threading
``cache_position`` through ``DuplexSTTModel.forward`` (additive: an optional kwarg) plus a
fixed cache subclass, and would also let the ~4.3k-token prompt prefill in one pass instead
of one position at a time. That is a real optimisation with a real correctness argument
behind it, but it is a change to the model's forward path, not to this driver -- so it is
deliberately out of scope here and left to be justified by the cost this driver measures.

Usage
-----
::

    session = StreamingFCSession(model, prompt_tokens, prompt_token_lens)
    session.start()
    while ...:
        out = session.push_audio(pcm_16k_float_samples)   # one tick's worth
        for call in out.tool_calls:
            session.push_tool_result(call.call_id, json.dumps(result))
        print(out.text_delta)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import DynamicCache

from nemo.utils import logging

__all__ = ["StreamingFCSession", "StreamingToolCall", "StreamStepOutput"]


# The model is trained to emit calls as
# `<TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>` between <SOTC>/<EOTC>
# (see the FC verification dump in a `fc_log=true` training run).
#
# The CLOSING TAG IS PART OF THE FORMAT. An earlier version of this pattern was
# `<TOOLCALL>\s*(\[.*\])\s*$`, anchored to end-of-string: it never matched real output, the
# blob kept its tags, json.loads failed, and every call silently decoded to name="" with
# empty arguments. Caught by check 4 in scripts/check_streaming_driver.py, which forces a
# known call instead of waiting for a checkpoint able to emit one.
#
# `(.*?)` rather than `(\[.*?\])` so a bare `{...}` object parses too; non-greedy is safe
# because the match is anchored on the closing tag (or end of string), so a `]` inside
# nested arguments cannot terminate it early. Both tags are tolerated as missing.
_TOOLCALL_BLOCK = re.compile(r"<TOOLCALL>\s*(.*?)\s*(?:</TOOLCALL>|$)", re.S)
_TOOLCALL_TAGS = re.compile(r"</?TOOLCALL>")

_FRAME_SECONDS = 0.08  # 12.5 Hz LM frame rate; matches _extract_online_audio_window


@dataclass
class StreamingToolCall:
    """One tool call decoded from the function channel."""

    call_id: str
    name: str
    arguments: Dict[str, Any]
    raw_text: str
    lm_position: int  # LM position of the <EOTC> that closed the call


@dataclass
class StreamStepOutput:
    """What one :meth:`StreamingFCSession.push_audio` produced."""

    text_delta: str = ""
    tool_calls: List[StreamingToolCall] = field(default_factory=list)
    frames_consumed: int = 0
    fc_tokens_injected: int = 0
    turn_ended: bool = False  # agent emitted text EOS during this step batch
    stalled_on_tool: bool = False  # waiting for a tool result; audio is buffered
    t_lm: int = 0
    t_audio: int = 0


class StreamingFCSession:
    """Drives ``DuplexSTTModel`` one audio frame at a time, with live tool calling.

    Batch size is fixed at 1: an interactive session is a single conversation, and the
    per-batch bookkeeping the offline path needs (ragged prompt lengths, FSDP width
    reconciliation) has no analogue here.

    Args:
        model: A ``DuplexSTTModel`` in eval mode on the target device.
        prompt_tokens: ``(1, P)`` system-prompt token ids.
        prompt_token_lens: ``(1,)`` true prompt length.
        max_audio_seconds: Audio horizon to preallocate. Ticks past this raise.
        max_fc_tokens: Extra LM positions reserved for injected function tokens.
            Defaults to the model's ``max_fc_total_tokens``, else 12000.
        window_size: Causal encoder window in frames. Defaults to the model's
            ``online_window_size`` (70 frames = 5.6 s).
        temperature/top_p/repetition_penalty/presence_penalty: Text-channel sampling.
            ``online_inference`` is greedy-only; these are threaded through
            ``inference_state``, which is where ``_step_inference`` reads them (``:3973``).
        stall_after_eotc: If True (default, and faithful to training) stop consuming
            audio once a call closes, until :meth:`push_tool_result` supplies the
            response. Training places response tokens immediately after ``<EOTC>`` with
            no audio frames between; consuming audio while a tool is in flight would put
            frames there and create a train/test mismatch. Incoming audio is buffered,
            not dropped, so no user speech is lost. Set False to keep listening through
            tool latency (more natural, off-distribution).
        force_eotr: If True, inject ``<EOTR>`` after the response instead of letting the
            model predict it. Off by default because ``<EOTR>`` is loss-bearing in
            training; turn on only if the model fails to emit it and the session stalls.
        force_use_cache: Override the KV-cache decision ``_init_inference`` makes. **Leave
            this None.** It exists only so ``scripts/check_streaming_driver.py`` can
            demonstrate what happens otherwise. Setting it True is unsound today -- see
            "The KV cache" in the module docstring for the measured reason.
    """

    def __init__(
        self,
        model,
        prompt_tokens: torch.Tensor,
        prompt_token_lens: torch.Tensor,
        max_audio_seconds: float = 240.0,
        max_fc_tokens: Optional[int] = None,
        window_size: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stall_after_eotc: bool = True,
        force_eotr: bool = False,
        force_use_cache: Optional[bool] = None,
        silence_probe_frames: int = 250,
    ):
        if prompt_tokens.shape[0] != 1:
            raise ValueError(f"StreamingFCSession is batch-size 1 only, got {prompt_tokens.shape[0]}")
        if not model.use_function_head:
            logging.warning(
                "[StreamingFC] model.use_function_head is False -- tool calling is disabled; "
                "this session will only stream agent text."
            )

        self.model = model
        self.prompt_tokens = prompt_tokens
        self.prompt_token_lens = prompt_token_lens
        self.device = model.device
        self.sample_rate = int(model.source_sample_rate)
        self.samples_per_frame = int(_FRAME_SECONDS * self.sample_rate)
        self.window_size = int(window_size if window_size is not None else model.cfg.get("online_window_size", 70))
        self.max_fc_tokens = int(
            max_fc_tokens if max_fc_tokens is not None else model.cfg.get("max_fc_total_tokens", 12000)
        )
        self.max_audio_frames = int(max_audio_seconds / _FRAME_SECONDS)
        self.stall_after_eotc = stall_after_eotc
        self.force_eotr = force_eotr
        self._force_use_cache = force_use_cache
        self._silence_probe_frames = int(silence_probe_frames)

        self._sampling = {
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        }

        if model.use_function_head:
            self.sotc_id, self.eotc_id, self.eotr_id = model._get_function_call_special_tokens()
        else:
            self.sotc_id = self.eotc_id = self.eotr_id = -1

        # --- inference state (populated by start()) ---
        self.state: Optional[dict] = None
        self._ans = None
        self.t_lm = 0
        self.t_audio = 0
        self._prompt_len = 0
        self._T = 0
        self._started = False
        self._finished = False
        self._turn_ended = False

        # --- audio bookkeeping ---
        # Full history is retained so each causal window is an exact slice. 240 s of
        # float32 @16 kHz is ~15 MB -- cheap next to the ~119 GB the 11B model uses.
        self._audio_history = torch.zeros(0, dtype=torch.float32, device=self.device)
        self._residual = torch.zeros(0, dtype=torch.float32, device=self.device)

        # --- function-channel decoding ---
        self._in_call = False
        self._call_buf: List[int] = []
        self._pending_calls: List[StreamingToolCall] = []
        self._awaiting_result = False
        self._inject_queue: List[int] = []
        self._call_counter = 0
        self._injected_spans: List[Tuple[int, int]] = []

        # --- text-channel decoding ---
        self._text_ids: List[int] = []
        self._emitted_chars = 0
        self._text_special = {model.text_pad_id, model.text_bos_id, model.text_eos_id}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @property
    def horizon(self) -> int:
        """Total preallocated LM positions after the prompt."""
        return self.max_audio_frames + self.max_fc_tokens

    @property
    def fc_positions_used(self) -> int:
        """LM positions spent on injected function tokens so far."""
        return self.t_lm - self.t_audio - self._prompt_len

    @torch.no_grad()
    def start(self) -> None:
        """Initialise state, preallocate the silence horizon, and step over the prompt."""
        if self._started:
            raise RuntimeError("start() already called")

        # Preallocate the horizon with real encoder silence, so injected FC positions need
        # no special-casing (cf. _expand_for_function_calling, which slices
        # _get_silence_embeddings_from_template for exactly that purpose).
        #
        # Not by encoding the whole horizon in one call: the horizon is thousands of frames
        # (12750 frames = 1020 s at the default FC budget), and the conformer would have to
        # attend over all of it at once. Instead encode a short probe and tile it, which is
        # exact rather than approximate -- the input is digital silence, so the encoder's
        # output is translation-invariant and every interior frame is identical. Only the
        # edge frames differ, and those are positions real audio overwrites anyway.
        self.state = self._init_with_tiled_silence()
        self.state.update(self._sampling)
        # Same set offline_inference builds (:4956) -- _sample_text_token uses it to keep
        # repetition/presence penalties off the special tokens.
        self.state["_text_special_ids"] = {
            self.model.text_pad_id,
            self.model.text_bos_id,
            self.model.text_eos_id,
        }
        self.state["rnnt_src_text"] = None
        self._apply_cache_override()

        self._prompt_len = int(self.state["start_gen_pos"])
        T = int(self.state["T"])
        if T < self._prompt_len + self.horizon:
            # perception subsampling is not exactly 1/1280, so trust the realised T.
            logging.info(
                f"[StreamingFC] realised horizon {T - self._prompt_len} frames "
                f"(requested {self.horizon}); clamping."
            )
        self._T = T

        # _step_zero returns (ans, inference_state), not ans -- offline_inference unpacks it
        # at :4990, online_inference does not (:5126, :5144), which is one more sign that
        # path is unexercised. Unpack it, or _step_inference's cached branch reads
        # ans["cache"] off a tuple.
        self._ans, _ = self.model._step_zero(self.state)
        self.t_lm = 1

        t0 = time.time()
        if self.state["use_cache"]:
            # With a cache, every prompt position must actually be stepped -- that is how
            # the cache gets built.
            for t in range(1, self._prompt_len):
                self._ans = self.model._step_inference(t, self.state, self._ans, None)
                self.t_lm = t + 1
        else:
            self._prefill_prompt_embeds()
            self.t_lm = self._prompt_len
        logging.info(
            f"[StreamingFC] started in {time.time() - t0:.1f}s: prompt={self._prompt_len} "
            f"positions, T={T}, audio horizon={self.max_audio_frames} frames, "
            f"fc budget={self.max_fc_tokens}, use_cache={self.state['use_cache']}"
        )
        self.t_audio = 0
        self._started = True

    def _prefill_prompt_embeds(self) -> None:
        """Fill ``input_embeds`` across the prompt block with no LLM forward passes.

        Exactly equivalent to stepping positions ``1..prompt_len-1`` one at a time, but only
        in the **no-cache** regime -- which is the regime this driver runs in (see "The KV
        cache" above). The argument:

        * Each ``_step_inference`` in that regime is stateless: it re-runs
          ``input_embeds[:, :t+1]`` from scratch (``:4007``), so nothing is carried between
          steps except the writes those steps perform.
        * At a prompt position every write is suppressed. All of the ``gen_text`` /
          ``gen_function`` / ``gen_asr`` sampling sits behind ``if not
          is_prompt_position.all()`` (``:4014``), as does forced turn-taking.
        * So the only surviving effect of the loop is the ``input_embeds[:, t]`` write at
          ``:3969`` -- and every input to that fusion is elementwise over positions
          (``_apply_embedding_transformations`` documents ``(B, T, D)``), so the whole block
          computes in one shot.
        * The RNNT branch (``:3919``) is gated on ``_run_rnnt_in_loop``, which we never set.

        This matters: the tau2 system prompt is ~4.3k tokens, so the loop is ~4.3k forwards
        whose results are all discarded, and being quadratic it dwarfs the conversation
        itself. With a cache it would be load-bearing, hence the branch in ``start()``.
        """
        m, st = self.model, self.state
        p = self._prompt_len
        if p <= 1:
            return
        prev = slice(0, p - 1)  # position t reads the channels at t-1

        agent_text_emb = m.embed_tokens(st["gen_text"][:, prev])
        user_text_emb = m.embed_asr_tokens(st["gen_asr"][:, prev]) if m.predict_user_text else None
        agent_text_emb, user_text_emb = m._apply_embedding_transformations(agent_text_emb, user_text_emb)
        function_emb = m.embed_tokens(st["gen_function"][:, prev]) if m.use_function_head else None

        st["input_embeds"][:, 1:p] = m.fusion_module(
            agent_text_embeds=agent_text_emb,
            user_audio_embeds=st["audio_embeds"][:, 1:p],
            user_text_embeds=user_text_emb,
            function_embeds=function_emb,
        )

    def _encode_silence_frame(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One steady-state encoder frame for digital silence: ``(1,1,H)``, ``(1,1,D)``.

        Taken from the middle of a short zeros probe so it is clear of the encoder's
        left/right padding edges.
        """
        n_samples = self._silence_probe_frames * self.samples_per_frame
        probe = torch.zeros(1, n_samples, dtype=torch.float32, device=self.device)
        probe_len = torch.tensor([n_samples], dtype=torch.long, device=self.device)
        encoded, lengths, asr_emb = self.model.perception(
            input_signal=probe, input_signal_length=probe_len, return_encoder_emb=True
        )
        mid = max(min(int(lengths[0].item()), encoded.shape[1]) // 2, 0)
        enc_frame = encoded[:, mid : mid + 1, :]
        asr_frame = asr_emb[:, mid : mid + 1, :] if asr_emb is not None else None
        return enc_frame, asr_frame

    def _init_with_tiled_silence(self) -> dict:
        """Run ``_init_inference`` with perception stubbed out to return tiled silence.

        ``_init_inference`` derives the whole timeline (T, every channel tensor, the
        prompt prepend) from what perception returns, so the horizon has to come from
        there. Patching ``perception.forward`` for the duration of the call is the least
        invasive seam available: it leaves ``_init_inference`` itself untouched, so the
        training and offline paths keep using the real encoder. The stub is restored in
        ``finally``, including on the error path.
        """
        enc_frame, asr_frame = self._encode_silence_frame()
        n = self.horizon
        encoded = enc_frame.expand(1, n, enc_frame.shape[-1]).contiguous()
        asr = asr_frame.expand(1, n, asr_frame.shape[-1]).contiguous() if asr_frame is not None else None

        def _tiled_forward(input_signal=None, input_signal_length=None, return_encoder_emb=False, **kwargs):
            # _init_inference mutates `lengths` in place when it prepends the prompt, so
            # hand it a fresh tensor rather than a shared one.
            lengths = torch.tensor([n], dtype=torch.long, device=self.device)
            if return_encoder_emb:
                return encoded, lengths, asr
            return encoded, lengths

        stub_signal = torch.zeros(1, self.samples_per_frame, dtype=torch.float32, device=self.device)
        stub_len = torch.tensor([self.samples_per_frame], dtype=torch.long, device=self.device)

        self.model.perception.forward = _tiled_forward
        try:
            return self.model._init_inference(
                input_signal=stub_signal,
                input_signal_lens=stub_len,
                input_pad_len=0,
                force_bos_positions=None,
                prompt_tokens=self.prompt_tokens,
                prompt_token_lens=self.prompt_token_lens,
            )
        finally:
            del self.model.perception.forward

    def _apply_cache_override(self) -> None:
        if self._force_use_cache is None:
            return
        if self._force_use_cache and not self.state["use_cache"]:
            logging.warning(
                "[StreamingFC] force_use_cache=True on a Nemotron checkpoint. This is "
                "expected to fail or to produce wrong output -- a plain DynamicCache has "
                "no Mamba conv/SSM state, and DuplexSTTModel.forward passes no "
                "cache_position, so the mixer re-runs its prefill branch every step. See "
                "the module docstring. Proceeding only because you asked."
            )
            self.state["use_cache"] = True
            self.state["cache"] = DynamicCache()
        elif not self._force_use_cache and self.state["use_cache"]:
            self.state["use_cache"] = False
            self.state["cache"] = None

    # ------------------------------------------------------------------
    # driving
    # ------------------------------------------------------------------

    @torch.no_grad()
    def push_audio(self, samples: torch.Tensor) -> StreamStepOutput:
        """Feed one tick of user audio and advance the model over whatever it completes.

        Args:
            samples: 1-D float waveform at ``model.source_sample_rate``, any length.
                A tick that is not a whole number of 80 ms frames leaves a residual that
                is carried into the next call, so a 200 ms tick grid yields 2,3,2,3,...
                frames per tick with no drift (mean 2.5, residual bounded at 640 samples).

        Returns:
            :class:`StreamStepOutput` describing what happened during this tick.
        """
        if not self._started:
            raise RuntimeError("call start() before push_audio()")

        out = StreamStepOutput()
        samples = samples.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        self._residual = torch.cat([self._residual, samples])

        # 1. Injected function tokens take precedence: they occupy LM positions that no
        #    audio frame may use, and training places them immediately after <EOTC>.
        out.fc_tokens_injected = self._drain_injection()

        # 2. Stalled waiting on a tool result -> buffer the audio and return.
        if self._awaiting_result and self.stall_after_eotc:
            out.stalled_on_tool = True
            out.tool_calls = self._take_calls()
            out.text_delta = self._take_text()
            out.t_lm, out.t_audio = self.t_lm, self.t_audio
            return out

        # 3. Consume every whole frame the residual now holds.
        n_frames = self._residual.numel() // self.samples_per_frame
        for _ in range(n_frames):
            if not self._can_step(1):
                logging.warning("[StreamingFC] horizon exhausted; dropping remaining audio")
                self._finished = True
                break
            frame = self._residual[: self.samples_per_frame]
            self._residual = self._residual[self.samples_per_frame :]
            self._audio_history = torch.cat([self._audio_history, frame])
            self._step_audio_frame()
            out.frames_consumed += 1
            # A call that closes mid-tick must stop audio consumption immediately, or the
            # remaining frames of this tick land between <EOTC> and the response.
            if self._awaiting_result and self.stall_after_eotc:
                out.stalled_on_tool = True
                break

        out.tool_calls = self._take_calls()
        out.text_delta = self._take_text()
        out.turn_ended = self._turn_ended
        self._turn_ended = False
        out.t_lm, out.t_audio = self.t_lm, self.t_audio
        return out

    def _can_step(self, n: int) -> bool:
        return self.t_lm + n <= self._T

    def _step_audio_frame(self) -> None:
        """Encode the causal window ending at the newest frame and step one LM position."""
        t = self.t_lm
        audio_emb, asr_emb = self._encode_causal_tail()
        self.state["audio_embeds"][:, t : t + 1, :] = audio_emb
        if asr_emb is not None and self.state.get("asr_emb") is not None:
            asr_store = self.state["asr_emb"]
            if asr_emb.shape[-1] == asr_store.shape[-1]:
                asr_store[:, t : t + 1, :] = asr_emb
        self._ans = self.model._step_inference(t, self.state, self._ans, None)
        self.t_lm = t + 1
        self.t_audio += 1
        self._observe(t)

    def _encode_causal_tail(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode the trailing ``window_size`` frames and return the last frame's embedding.

        This is the causality that ``online_inference`` intended but discarded: the encoder
        sees only samples up to and including the current frame, never beyond.
        """
        window_samples = self.window_size * self.samples_per_frame
        tail = self._audio_history[-window_samples:]
        n = tail.numel()
        window = tail.reshape(1, n)
        lens = torch.tensor([n], dtype=torch.long, device=self.device)
        encoded, _, asr_emb = self.model.perception(
            input_signal=window, input_signal_length=lens, return_encoder_emb=True
        )
        # Raw encoder output, unweighted: _init_inference stores audio_embeds unweighted
        # (:3713) and the fusion module applies duplex_user_channel_weight itself.
        # (online_inference pre-multiplied at :5167, but that write was dead.)
        last_asr = asr_emb[:, -1:, :] if asr_emb is not None else None
        return encoded[:, -1:, :], last_asr

    # ------------------------------------------------------------------
    # function channel
    # ------------------------------------------------------------------

    def _observe(self, t: int) -> None:
        """Read back the channels the model just wrote at position ``t``."""
        text_tok = int(self.state["gen_text"][0, t].item())
        if text_tok == self.model.text_eos_id:
            self._turn_ended = True
        if text_tok == self.model.text_bos_id:
            self._text_ids = []
            self._emitted_chars = 0
        elif text_tok not in self._text_special:
            self._text_ids.append(text_tok)

        if not self.model.use_function_head:
            return
        fn_tok = int(self.state["gen_function"][0, t].item())
        if fn_tok == self.sotc_id:
            self._in_call = True
            self._call_buf = []
        elif self._in_call and fn_tok == self.eotc_id:
            self._in_call = False
            self._close_call(t)
        elif self._in_call and fn_tok != self.model.text_pad_id:
            self._call_buf.append(fn_tok)

    def _close_call(self, t: int) -> None:
        raw = self.model.tokenizer.ids_to_text(self._call_buf)
        self._call_buf = []
        parsed = self._parse_call(raw)
        if parsed is None:
            logging.warning(f"[StreamingFC] unparseable tool call at t={t}: {raw!r}")
            # Still mark the turn as awaiting a result: the model emitted <EOTC> and, per
            # the training layout, expects response tokens next. The caller should push an
            # error result so the session can continue.
        self._call_counter += 1
        name, args = parsed if parsed is not None else ("", {})
        call = StreamingToolCall(
            call_id=f"call_{self._call_counter}",
            name=name,
            arguments=args,
            raw_text=raw,
            lm_position=t,
        )
        self._pending_calls.append(call)
        self._awaiting_result = True

    @staticmethod
    def _parse_call(raw: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        m = _TOOLCALL_BLOCK.search(raw.strip())
        # Fallback strips any stray tags rather than passing them to json.loads, so a
        # garbled or partial opening tag degrades to a parse attempt instead of a hard None.
        blob = m.group(1) if m else _TOOLCALL_TAGS.sub("", raw).strip()
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(obj, list):
            if not obj:
                return None
            obj = obj[0]
        if not isinstance(obj, dict) or "name" not in obj:
            return None
        args = obj.get("arguments", obj.get("parameters", {}))
        return str(obj["name"]), args if isinstance(args, dict) else {}

    def push_tool_result(self, content: str, is_error: bool = False) -> int:
        """Queue a tool result for injection into the function channel.

        The wire format matches training exactly -- the response tokens are the tokenised
        ``<TOOL_RESPONSE>[{"content": ...}]</TOOL_RESPONSE>`` string produced by
        ``scripts/episodes_to_nemotron_training.py``. ``<EOTR>`` is *not* appended: it is
        loss-bearing in training, so the model predicts it (unless ``force_eotr``).

        Args:
            content: The tool's return value. A JSON string is embedded as a JSON value
                (single-encoded, matching the repaired training data); anything else is
                embedded as a string.
            is_error: Recorded in the payload so the model can see the call failed.

        Returns:
            Number of tokens queued.
        """
        payload: Any = content
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                payload = parsed
        entry: Dict[str, Any] = {"content": payload}
        if is_error:
            entry["is_error"] = True
        text = f"<TOOL_RESPONSE>{json.dumps([entry], separators=(',', ':'))}</TOOL_RESPONSE>"
        ids = list(self.model.tokenizer.text_to_ids(text))
        if self.force_eotr:
            ids.append(self.eotr_id)
        budget = self.max_fc_tokens - self.fc_positions_used
        if len(ids) > budget:
            logging.warning(
                f"[StreamingFC] tool response is {len(ids)} tokens but only {budget} of the "
                f"{self.max_fc_tokens}-token FC budget remain; truncating."
            )
            ids = ids[: max(0, budget)]
        self._inject_queue.extend(ids)
        self._awaiting_result = False
        return len(ids)

    def _drain_injection(self) -> int:
        """Step the LM over every queued function token.

        Each injected token is written into ``gen_function`` *before* the step, so
        ``_step_inference``'s ``already_injected`` gate (:3999) keeps it instead of
        overwriting it with a prediction. The audio channel at these positions already
        holds encoder silence from the preallocated horizon, so nothing else is needed.
        """
        if not self._inject_queue:
            return 0
        start = self.t_lm
        n = 0
        while self._inject_queue:
            if not self._can_step(1):
                logging.warning("[StreamingFC] horizon exhausted mid-injection")
                self._finished = True
                break
            tok = self._inject_queue.pop(0)
            t = self.t_lm
            self.state["gen_function"][0, t] = tok
            self._ans = self.model._step_inference(t, self.state, self._ans, None)
            self.t_lm = t + 1
            n += 1
            # Deliberately no _observe() here: these tokens are ours, not predictions.
            # Text emitted at injected positions IS a prediction though -- capture it.
            self._observe_text_only(t)
        if n:
            self._injected_spans.append((start, start + n))
        return n

    def _observe_text_only(self, t: int) -> None:
        text_tok = int(self.state["gen_text"][0, t].item())
        if text_tok == self.model.text_eos_id:
            self._turn_ended = True
        if text_tok == self.model.text_bos_id:
            self._text_ids = []
            self._emitted_chars = 0
        elif text_tok not in self._text_special:
            self._text_ids.append(text_tok)

    # ------------------------------------------------------------------
    # output accessors
    # ------------------------------------------------------------------

    def _take_calls(self) -> List[StreamingToolCall]:
        calls, self._pending_calls = self._pending_calls, []
        return calls

    def _take_text(self) -> str:
        """Incremental detokenisation.

        Decoding the whole utterance and diffing on characters is deliberate: BPE pieces
        do not detokenise independently (leading-space and multi-byte pieces), so decoding
        token-by-token would corrupt the text.
        """
        if not self._text_ids:
            return ""
        full = self.model.tokenizer.ids_to_text(self._text_ids)
        if len(full) <= self._emitted_chars:
            return ""
        delta = full[self._emitted_chars :]
        self._emitted_chars = len(full)
        return delta

    @property
    def full_text(self) -> str:
        """The agent's current utterance so far."""
        return self.model.tokenizer.ids_to_text(self._text_ids) if self._text_ids else ""

    def function_channel_text(self) -> str:
        """Everything on the function channel so far, for debugging."""
        if not self.model.use_function_head:
            return ""
        toks = self.state["gen_function"][0, : self.t_lm]
        keep = toks[toks != self.model.text_pad_id].tolist()
        return self.model.tokenizer.ids_to_text(keep) if keep else ""

    def stats(self) -> Dict[str, Any]:
        return {
            "t_lm": self.t_lm,
            "t_audio": self.t_audio,
            "prompt_len": self._prompt_len,
            "fc_positions_used": self.fc_positions_used,
            "fc_budget": self.max_fc_tokens,
            "injected_spans": list(self._injected_spans),
            "awaiting_result": self._awaiting_result,
            "residual_samples": int(self._residual.numel()),
            "finished": self._finished,
        }
