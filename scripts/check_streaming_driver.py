#!/usr/bin/env python
"""GPU check for StreamingFCSession -- the Phase 0 streaming+FC inference driver.

Runs three independent checks against a real 11B checkpoint and a real cut from the
repaired tau2 Shar. Each check is skippable so a failure in one still reports the others.

  1. cache               Demonstrates that the KV cache cannot be forced on for this
                         checkpoint. Nemotron-Nano-9B-v2 is a hybrid Mamba/attention model,
                         so a DynamicCache has no conv/SSM state, and DuplexSTTModel.forward
                         passes no cache_position -- see "The KV cache" in
                         streaming_fc_session.py. This check exists to keep that conclusion
                         reproducible rather than folklore: it PASSES when forcing the cache
                         on fails, because that is the finding. The consequence is that every
                         step re-runs the full prefix (O(T^2)), which check 2 prices.
  2. stream              StreamingFCSession fed the cut's audio in 200 ms ticks. Always
                         reports ms/tick against the 200 ms realtime budget, which is the
                         number that decides whether this driver is usable live. With
                         --compare-offline it also runs offline_inference and reports the
                         similarity: these are NOT expected to match exactly, because
                         offline sees the full non-causal encode while the driver sees a
                         causal 5.6 s window, so the gap measures what causal windowing
                         costs -- the one Phase 0 parameter (online_window_size) that could
                         force a design change.
  3. fc                  Does the driver detect a <SOTC>...<EOTC> call on the function
                         channel, and does push_tool_result() inject cleanly and let the
                         session continue? Uses the cut's own tool schemas via its system
                         prompt, so a tau2-tuned checkpoint should call something.

                         Read the verdict with the checkpoint in mind. Against
                         `stt_extracted_lora` -- the key-remapped *pretrained* checkpoint that
                         is the INPUT to our SFT, not a product of it -- no tool call is the
                         expected result: it has never seen tau2 data. Observed there: stable
                         off-task policy text and a function channel carrying stray prompt-ish
                         tokens with no <SOTC>. That is an untuned model, not a broken driver.
                         What this check validates on such a checkpoint is the mechanism --
                         see the two-clock arithmetic in stats(), which is exact.

Usage (must run on a GPU node; see the Slurm note in the repo plan doc):

  srun --jobid=<id> --overlap --gres=gpu:1 \
    /fsx/home/kai.li/miniforge3/envs/voicechat/bin/python scripts/check_streaming_driver.py \
      --shards /fsx/home/kai.li/data/voicechat/tau2_canonical/shards \
      --checkpoint /fsx/home/kai.li/data/voicechat/stt_extracted_lora \
      --checks cache,fc
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True, help="Lhotse Shar dir with the eval cuts")
    p.add_argument("--checkpoint", required=True, help="pretrained_s2s_model dir")
    p.add_argument(
        "--config",
        default=str(REPO / "examples/speechlm2/conf/finetune/s2s_duplex_stt_11b.yaml"),
        help="Model config to build from",
    )
    p.add_argument("--cut-index", type=int, default=0, help="Which cut in the shard to use")
    p.add_argument("--tick-ms", type=int, default=200, help="Tick size fed to push_audio")
    p.add_argument("--max-seconds", type=float, default=30.0, help="Truncate the cut's audio")
    p.add_argument(
        "--checks",
        default="cache,stream,fc,fcforce",
        help="Comma-separated subset of: cache, stream, fc, fcforce",
    )
    p.add_argument(
        "--force-at-frame",
        type=int,
        default=20,
        help="Audio frame at which check 4 starts forcing its synthetic call. The call is "
        "~60 tokens at one token per 80 ms frame, so --max-seconds must cover "
        "(this + 60) frames plus room for the injected response.",
    )
    p.add_argument("--window-size", type=int, default=None, help="Override online_window_size")
    p.add_argument(
        "--compare-offline",
        action="store_true",
        help="In check 2, also run offline_inference and report the similarity. Costly: "
        "offline runs one full-prefix forward per LM position including the ~4.3k prompt "
        "positions, i.e. ~30x the streaming pass it is being compared against.",
    )
    p.add_argument(
        "--fc-budget",
        type=int,
        default=2000,
        help="LM positions the session reserves for injected function tokens. The model "
        "config's max_fc_total_tokens (12000) is a whole-conversation budget; a short "
        "check does not need it, and every reserved position costs KV cache.",
    )
    p.add_argument(
        "--log-dir",
        default="/fsx/home/kai.li/data/voicechat/streaming_check",
        help="exp_manager.explicit_log_dir (the model reads it at construction). "
        "Keep this off /tmp -- /tmp is the small local root, not Lustre.",
    )
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[check] {msg}", flush=True)


def load_model(config_path: str, checkpoint: str, log_dir: str):
    """Build the model the way s2s_duplex_stt_train.py does, then move it to eval/GPU.

    DuplexSTTModel takes the *whole* config as a plain dict (it reads cfg.data and
    cfg.exp_manager as well as cfg.model), and its dist-aware branches need a process
    group -- hence the single-rank nccl init, matching the training script's line 41.
    """
    import os

    import torch
    from omegaconf import OmegaConf

    from nemo.collections.speechlm2 import DuplexSTTModel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True

    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    cfg.model.pretrained_s2s_model = checkpoint
    # fc_log/debug_fc off: this script prints its own summaries, and the FC dumps run to
    # thousands of lines per cut (one `Non-PAD positions:` line alone is ~40 KB).
    cfg.model.fc_log = False
    cfg.model.debug_fc = False
    cfg.exp_manager.explicit_log_dir = log_dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.resolve(cfg)

    log("building DuplexSTTModel (loads the 11B weights; ~10 min per the training-run notes)")
    t0 = time.time()
    model = DuplexSTTModel(OmegaConf.to_container(cfg, resolve=True))
    model = model.to(torch.device("cuda")).eval()
    log(f"model ready in {time.time() - t0:.0f}s")
    return model, cfg


def load_cut(shards: str, index: int):
    """Pull one cut plus its system prompt out of the Shar."""
    from lhotse import CutSet

    cuts = CutSet.from_shar(in_dir=shards)
    for i, cut in enumerate(cuts):
        if i == index:
            return cut
    raise IndexError(f"shard has fewer than {index + 1} cuts")


def cut_audio_and_prompt(model, cut, max_seconds: float):
    """Return (waveform (1,N) float32 cuda, prompt_tokens (1,P), prompt_token_lens (1,))."""
    import numpy as np
    import torch

    audio = cut.load_audio()  # (channels, samples)
    wav = np.asarray(audio)[0].astype("float32")
    n_max = int(max_seconds * cut.sampling_rate)
    wav = wav[:n_max]

    if cut.sampling_rate != model.source_sample_rate:
        import torchaudio

        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, cut.sampling_rate, model.source_sample_rate)
    else:
        wav_t = torch.from_numpy(wav).unsqueeze(0)

    wav_t = wav_t.to("cuda")

    # Reproduce collate_system_prompt (s2s_dataset.py:2118) exactly, because that is what
    # produced prompt_tokens during training. Two things it does that are easy to miss:
    #
    #   * cut.custom["system_prompt"] takes the FIRST branch (:2148). Our tau2 cuts set it
    #     (byte-identically to supervisions[0].text), so the `augment_fc_system_prompt`
    #     wrap on the *elif* branch never fires -- the model trained on the raw tools-only
    #     prompt despite the YAML flag being true. Verified against the shard.
    #   * the ids are wrapped: [bos] + text_to_ids(prompt) + [eos] (:2220).
    system_prompt = (cut.custom or {}).get("system_prompt")
    if not system_prompt:
        system_prompt = cut.supervisions[0].text
    ids = [model.tokenizer.bos] + model.tokenizer.text_to_ids(system_prompt) + [model.tokenizer.eos]
    prompt_tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
    prompt_lens = torch.tensor([len(ids)], dtype=torch.long, device="cuda")
    log(f"cut {cut.id}: {wav_t.shape[-1] / model.source_sample_rate:.1f}s audio, prompt={len(ids)} tokens")
    return wav_t, prompt_tokens, prompt_lens


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# offline_inference's decoded text carries the audio-timestamp tokens (`<$0.72$>`, `<|1.52|>`)
# and other specials; the streaming driver filters them via _text_special. Comparing the two
# raw makes the ratio a formatting artifact rather than a measurement, so normalize first.
_SPECIAL_TOK = re.compile(r"<\$[^>]*\$>|<\|[^|]*\|>|<[A-Z_]+\d*>")


def normalize_text(s: str) -> str:
    return " ".join(_SPECIAL_TOK.sub(" ", s).split()).strip()


def run_offline(model, wav, prompt_tokens, prompt_lens, use_cache: bool | None):
    """offline_inference with an optional forced cache setting.

    _init_inference decides use_cache from the LLM name (:3719); to override it we patch
    the returned state, which is the only seam that does not touch the training path.
    """
    import torch
    from transformers import DynamicCache

    original = model._init_inference

    def patched(*a, **kw):
        state = original(*a, **kw)
        if use_cache is not None:
            state["use_cache"] = use_cache
            state["cache"] = DynamicCache() if use_cache else None
        return state

    model._init_inference = patched
    try:
        with torch.no_grad():
            ans = model.offline_inference(
                input_signal=wav,
                input_signal_lens=torch.tensor([wav.shape[-1]], device="cuda"),
                decode_audio=False,
                prompt_tokens=prompt_tokens,
                prompt_token_lens=prompt_lens,
            )
    finally:
        model._init_inference = original

    text = ans["text"]
    return text[0] if isinstance(text, (list, tuple)) else str(text)


def check_cache(model, wav, prompt_tokens, prompt_lens) -> bool:
    """Reproduce the finding that the cache cannot be forced on for this checkpoint.

    PASS here means "forcing the cache failed", because that is the documented
    conclusion. If this ever starts passing the cache path, the driver's cost model and
    the module docstring both need revisiting -- so a silent success is a FAIL.
    """
    log("=" * 70)
    log("CHECK 1: can the KV cache be forced on? (expected: no, by construction)")
    log("=" * 70)
    # Cache ON goes FIRST and the no-cache baseline is computed only if it survives. The
    # failure happens in the Mamba mixer on step 0, so the expected outcome costs seconds,
    # whereas the no-cache reference is a full O(T^2) offline pass (~4.5k+ prefix forwards).
    # Paying for a comparison against a branch that cannot run is pure waste.
    t0 = time.time()
    try:
        with_cache = run_offline(model, wav, prompt_tokens, prompt_lens, use_cache=True)
    except Exception as exc:  # noqa: BLE001 -- the exception type IS the measurement
        log(f"cache ON raised {type(exc).__name__} after {time.time() - t0:.0f}s: {str(exc)[:300]}")
        log("RESULT: PASS -- forcing the cache on is unsound, as documented. The driver "
            "correctly leaves force_use_cache=None and eats the full-prefix cost.")
        return True

    log(f"cache ON  : {with_cache[:200]!r}")
    t0 = time.time()
    without = run_offline(model, wav, prompt_tokens, prompt_lens, use_cache=False)
    log(f"cache OFF ({time.time() - t0:.0f}s): {without[:200]!r}")
    ratio = similarity(with_cache, without)
    log(f"similarity = {ratio:.4f}")
    log("RESULT: FAIL -- forcing the cache on did NOT fail. Either the checkpoint's remote "
        "code changed or cache_position is now threaded; re-read 'The KV cache' in "
        "streaming_fc_session.py, because its cost model assumes no cache is possible.")
    return False


def build_session(model, prompt_tokens, prompt_lens, args, **kw):
    from nemo.collections.speechlm2.models.streaming_fc_session import StreamingFCSession

    return StreamingFCSession(
        model,
        prompt_tokens=prompt_tokens,
        prompt_token_lens=prompt_lens,
        max_audio_seconds=max(args.max_seconds * 2, 60.0),
        max_fc_tokens=args.fc_budget,
        window_size=args.window_size,
        **kw,
    )


def drive(session, wav, tick_ms: int, sample_rate: int, on_call=None, max_ticks: int | None = None):
    """Feed `wav` to `session` in `tick_ms` slices. Returns (text, calls, tick_count)."""
    samples_per_tick = int(sample_rate * tick_ms / 1000)
    texts, calls = [], []
    n = wav.shape[-1]
    offset = 0
    ticks = 0
    while offset < n:
        chunk = wav[0, offset : offset + samples_per_tick]
        offset += samples_per_tick
        out = session.push_audio(chunk)
        ticks += 1
        if out.text_delta:
            texts.append(out.text_delta)
        for call in out.tool_calls:
            calls.append(call)
            log(f"  tick {ticks}: TOOL CALL {call.name}({call.arguments}) @ t_lm={call.lm_position}")
            if on_call is not None:
                on_call(session, call)
        if out.stalled_on_tool and on_call is None:
            log(f"  tick {ticks}: stalled waiting for a tool result (no handler) -- stopping")
            break
        if max_ticks and ticks >= max_ticks:
            break
    return "".join(texts), calls, ticks


def check_stream(model, wav, prompt_tokens, prompt_lens, args) -> bool:
    log("=" * 70)
    log("CHECK 2: streaming throughput, and (optionally) agreement with offline")
    log("=" * 70)
    # The offline comparison is off by default because offline_inference is O(T^2) in the
    # SAME way streaming is (no cache), and its T includes the ~4.3k-token prompt: it runs
    # ~4.5k full-prefix forwards where streaming runs one per frame. That makes it roughly
    # 30x the cost of the thing being measured, for a number that only matters once the
    # throughput below is viable at all. Enable with --compare-offline.
    offline = None
    if args.compare_offline:
        t0 = time.time()
        offline = run_offline(model, wav, prompt_tokens, prompt_lens, use_cache=None)
        log(f"offline_inference took {time.time() - t0:.0f}s")

    session = build_session(model, prompt_tokens, prompt_lens, args)
    t0 = time.time()
    session.start()
    log(f"session.start() took {time.time() - t0:.0f}s")
    t0 = time.time()
    streamed, _, ticks = drive(session, wav, args.tick_ms, model.source_sample_rate)
    elapsed = time.time() - t0
    audio_s = wav.shape[-1] / model.source_sample_rate

    log(f"streamed ({len(streamed)} chars): {streamed[:200]!r}")
    if offline is not None:
        log(f"offline  ({len(offline)} chars): {offline[:200]!r}")
        no, ns = normalize_text(offline), normalize_text(streamed)
        log(f"similarity raw        = {similarity(offline, streamed):.4f}")
        log(f"similarity normalized = {similarity(no, ns):.4f}  "
            f"(specials stripped; {len(no)} vs {len(ns)} chars)")
        log("NOTE: on a pre-SFT checkpoint both outputs are off-task boilerplate, so this "
            "number does NOT answer the online_window_size question -- see 0b-ter.")
    else:
        log("offline  : (skipped; pass --compare-offline for the causal-window similarity)")
    log(f"stats: {session.stats()}")
    log(f"{ticks} ticks in {elapsed:.1f}s for {audio_s:.1f}s of audio "
        f"= {elapsed / max(audio_s, 1e-6):.2f}x realtime "
        f"({1000 * elapsed / max(ticks, 1):.0f} ms/tick vs {args.tick_ms} ms budget)")
    # Not a pass/fail: this check measures the causal-windowing gap and the speed.
    log("RESULT: informational (no threshold) -- see similarity and realtime factor above")
    return True


def check_fc(model, wav, prompt_tokens, prompt_lens, args) -> bool:
    log("=" * 70)
    log("CHECK 3: live function calling (detect <SOTC>..<EOTC>, inject a response)")
    log("=" * 70)
    session = build_session(model, prompt_tokens, prompt_lens, args)
    session.start()

    injected = []

    def handle(sess, call):
        # A canned result: this check is about the injection mechanism, not tool fidelity.
        n = sess.push_tool_result('{"status": "ok", "note": "synthetic result"}')
        injected.append((call.name, n))
        log(f"    injected {n} response tokens for {call.name}")

    text, calls, ticks = drive(session, wav, args.tick_ms, model.source_sample_rate, on_call=handle)

    log(f"agent text: {text[:300]!r}")
    log(f"function channel: {session.function_channel_text()[:400]!r}")
    log(f"calls detected: {[(c.name, c.arguments) for c in calls]}")
    log(f"responses injected: {injected}")
    log(f"stats: {session.stats()}")
    ok = len(calls) > 0
    if ok:
        log("RESULT: PASS -- call detected and response injected")
    else:
        log("RESULT: INCONCLUSIVE -- no tool call in this window. Try a longer --max-seconds or "
            "another --cut-index. Note this is the EXPECTED outcome against a pre-SFT "
            "checkpoint (see check 3's note in the module docstring); the detection path "
            "cannot be exercised by a model that never learned to emit <SOTC>.")
    return ok


def check_fc_forced(model, wav, prompt_tokens, prompt_lens, args) -> bool:
    """Exercise detect -> parse -> inject with a *synthetic* call on the function channel.

    Check 3 can only pass if the checkpoint spontaneously emits <SOTC>, which a pre-SFT
    checkpoint never will -- so on its own it leaves the entire FC round-trip unvalidated
    until after training, which is the wrong order. This check removes that dependency.

    It writes a known <SOTC> <TOOLCALL>[...]</TOOLCALL> <EOTC> sequence into
    state["gen_function"] at consecutive positions as they are stepped, by wrapping
    _step_inference (which runs immediately BEFORE _observe reads that channel, see
    _step_audio_frame). Everything downstream is the real code path: _observe's state
    machine, _close_call, _parse_call, the mid-tick stall, push_tool_result's wire
    formatting, and _drain_injection stepping the LM over the response.

    One FC token per audio frame matches the training layout, where call tokens occupy
    consecutive LM positions while audio keeps flowing (see the channel diagram at
    streaming_fc_session.py:80).

    The ONLY thing not covered is the model's own decision to emit <SOTC>. That is a
    checkpoint property, not a driver property, and check 3 is where it gets measured.
    """
    log("=" * 70)
    log("CHECK 4: forced function call -- validates the FC seam independent of checkpoint")
    log("=" * 70)
    if not model.use_function_head:
        log("RESULT: SKIP -- model has no function head")
        return True

    session = build_session(model, prompt_tokens, prompt_lens, args)
    session.start()
    if session.sotc_id < 0:
        log("RESULT: SKIP -- no FC special tokens on this tokenizer")
        return False

    call_text = (
        '<TOOLCALL>[{"name": "find_user_id_by_name_zip", "arguments": '
        '{"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}}]</TOOLCALL>'
    )
    queue = [session.sotc_id] + list(model.tokenizer.text_to_ids(call_text)) + [session.eotc_id]
    log(f"forcing a {len(queue)}-token call starting at audio frame {args.force_at_frame}")

    original_step = model._step_inference

    def patched(t, st, ans, *a, **kw):
        out = original_step(t, st, ans, *a, **kw)
        # Fire only while the queue lasts, and only on audio frames. Once <EOTC> is
        # observed the queue is empty, so this can never clobber _drain_injection's own
        # writes to gen_function.
        if queue and session.t_audio >= args.force_at_frame:
            st["gen_function"][0, t] = queue.pop(0)
        return out

    model._step_inference = patched
    injected = []

    def handle(sess, call):
        n = sess.push_tool_result('{"user_id": "yusuf_rossi_9620"}')
        injected.append((call.name, n))
        log(f"    injected {n} response tokens for {call.name}")

    try:
        text, calls, ticks = drive(
            session, wav, args.tick_ms, model.source_sample_rate, on_call=handle
        )
    finally:
        model._step_inference = original_step

    st = session.stats()
    log(f"calls detected: {[(c.name, c.arguments) for c in calls]}")
    log(f"responses injected: {injected}")
    log(f"agent text: {text[:200]!r}")
    log(f"stats: {st}")

    # Grade every link in the chain separately, so a partial failure names itself.
    problems = []
    if not calls:
        problems.append("no call detected at all -- the forced tokens never reached _observe")
    else:
        # Assert on the FIRST call only. Additional calls are not a failure: once the model
        # has seen one forced call plus its injected response, it can imitate the format and
        # emit its own -- which is evidence the injection is being *read*, not just stepped
        # over. Requiring exactly one would fail the check for succeeding too well.
        c = calls[0]
        if c.name != "find_user_id_by_name_zip":
            problems.append(f"name mis-parsed: {c.name!r} (raw={c.raw_text[:120]!r})")
        if c.arguments.get("zip") != "19122":
            problems.append(f"arguments mis-parsed: {c.arguments!r}")
        for extra in calls[1:]:
            log(f"  model-initiated follow-up call (in-context imitation): "
                f"{extra.name}({extra.arguments}) @ t_lm={extra.lm_position}")
    if not injected or injected[0][1] <= 0:
        problems.append("no response tokens were queued")
    if st["fc_positions_used"] <= 0:
        problems.append("fc_positions_used stayed 0 -- the LM never stepped over the response")
    if st["awaiting_result"]:
        problems.append("session still awaiting a result after push_tool_result")
    # The two-clock invariant: every LM position is a prompt position, an audio frame, or an
    # injected FC token. If this drifts, the timeline is silently corrupt.
    expected_t_lm = st["prompt_len"] + st["t_audio"] + st["fc_positions_used"]
    if st["t_lm"] != expected_t_lm:
        problems.append(
            f"two-clock drift: t_lm={st['t_lm']} != prompt {st['prompt_len']} + audio "
            f"{st['t_audio']} + fc {st['fc_positions_used']} = {expected_t_lm}"
        )
    else:
        log(f"two-clock invariant holds: t_lm {st['t_lm']} = prompt {st['prompt_len']} "
            f"+ audio {st['t_audio']} + fc {st['fc_positions_used']}")

    if problems:
        for p in problems:
            log(f"  PROBLEM: {p}")
        log("RESULT: FAIL -- the FC round-trip is broken (see problems above)")
        return False
    log("RESULT: PASS -- call detected, parsed, result injected, LM stepped, clocks exact")
    return True


def main() -> int:
    args = parse_args()
    checks = {c.strip() for c in args.checks.split(",") if c.strip()}

    model, _ = load_model(args.config, args.checkpoint, args.log_dir)
    cut = load_cut(args.shards, args.cut_index)
    wav, prompt_tokens, prompt_lens = cut_audio_and_prompt(model, cut, args.max_seconds)

    results = {}
    if "cache" in checks:
        results["cache"] = check_cache(model, wav, prompt_tokens, prompt_lens)
    if "stream" in checks:
        results["stream"] = check_stream(model, wav, prompt_tokens, prompt_lens, args)
    if "fc" in checks:
        results["fc"] = check_fc(model, wav, prompt_tokens, prompt_lens, args)
    if "fcforce" in checks:
        results["fcforce"] = check_fc_forced(model, wav, prompt_tokens, prompt_lens, args)

    log("=" * 70)
    for name, ok in results.items():
        log(f"{name:10s} {'PASS' if ok else 'FAIL/INCONCLUSIVE'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
