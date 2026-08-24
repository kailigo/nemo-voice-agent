#!/usr/bin/env python3
"""Localise mode H: WHY does the container refuse a tau-voice domain policy?

Mode H (NEMO_FAILURE_MODES.md §3) is the blocker on arm A': the served NIM container greets
the customer, then refuses the task and repeats the refusal until the session dies. Retail:
"I am unable to assist with requests that involve unauthorized access to user accounts,
payment information, or personal data." Airline refuses differently ("fraudulent documents"),
so it is not one canned string, and it fires before the user has asked for anything -- which
points at the SYSTEM PROMPT rather than at user speech.

Arm A'' already ruled out the obvious fix: +305 chars of explicit authorization ("Do not
refuse these requests") changed the refusal byte-for-byte not at all. What is still unknown is
which PROPERTY of the prompt triggers it. This script varies one property at a time against
the same audio and the same server.

  condition                what it changes vs baseline        what a non-refusal would mean
  ------------------------ ---------------------------------- ------------------------------
  baseline                 nothing (the exact arm-A' prompt)  the harness is wrong, stop
  policy_only              drops the voice instruction        the instruction triggers it
  instruction_only         drops the domain policy            the policy text triggers it
  benign_short             ~200 chars, same tools             length or subject matter
  baseline_no_tools        tools=[]                           the tool schemas trigger it
  truncated_policy         first ~1500 chars of the policy    length, not subject matter

WHY THIS COSTS ALMOST NOTHING. It talks to the already-running container over the same
websocket the provider uses, and it replays a REAL user utterance that tau-voice already
persisted (`both.wav` + `user_labels.txt`) rather than synthesising one. So: no ElevenLabs
quota, no episode, no second GPU. The container is idle between arms anyway.

BASELINE IS A GATE, NOT A FORMALITY. Sampling is greedy, so `baseline` must reproduce the
refusal. If it does not, this harness is not sending what the provider sends and every other row
is uninterpretable; the summary says so explicitly rather than letting the table be read.

The prompt is not hand-copied: it is assembled with tau2's own template and instruction
constant (`AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN`, `AUDIO_NATIVE_VOICE_INSTRUCTION`) and the length
is checked against the 7469 chars the server logged for the real run. A hand-typed
approximation would be the same class of bug as the mode-A normaliser: indistinguishable from
a real result.

Usage (container must be up; no GPU needed by this process):
    NEMO_RT_URL=ws://ip-10-1-76-28:9000 python scripts/mode_h_probe.py
    NEMO_RT_URL=... python scripts/mode_h_probe.py --condition baseline --seconds 60
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

TAU2 = Path("/fsx/home/kai.li/code/tau-voice-2")
sys.path.insert(0, str(TAU2 / "src"))

# Where the stimulus comes from. Deliberately a GEMINI episode, not one of ours: the point is
# to feed the container a user side that actually reaches the account-and-order part of the
# task, and every arm-A retail episode died in its opening exchange. Arm A's retail__7 has just
# two user turns ("I need to exchange a couple items", "I don't have that with me") and never
# mentions an account, so it cannot elicit a refusal *about* account access. Gemini's retail
# task 7 is the same task carried to completion: 9 user turns, the user id spelled out letter
# by letter, 267 s.
#
# This is an OPEN-LOOP replay. The user turns were responses to Gemini's replies, not to this
# container's, so the exchange is not coherent past the first turn. That is acceptable and even
# useful here -- we are testing what the PROMPT makes the model do with realistic task content,
# not scoring a conversation -- but it means nothing about turn-taking or task success may be
# read off this script. Only the refusal behaviour is interpretable.
STIM_RUN = TAU2 / "data/simulations/gemini_baseline_0821b/retail"
STIM_TASK = "task_7"

CHUNK_MS = 200          # the tick size the episode uses
INPUT_RATE = 16000      # CONTAINER_MODEL_SAMPLE_RATE; the one field that must be right
EXPECTED_BASELINE_CHARS = 7469  # what the server logged for the real arm-A' session

# Exact event types, not a substring match. The container also runs ASR on the USER channel and
# emits it as `conversation.item.input_audio_transcription.delta` -- a "transcript" match would
# fold the user's own words into what we score as the agent's reply.
AGENT_TEXT = "response.output_audio_transcript.delta"
# Only the DELTA feeds `heard`. The `.completed` event repeats the whole turn's transcript, so
# accumulating both made every user turn appear twice ("hellohellois anyone there?is anyone
# there?") and read as if the container had double-processed the input. It had not; that was
# this script. `.completed` is used as a turn boundary instead.
USER_ASR_DELTA = "conversation.item.input_audio_transcription.delta"
USER_ASR_DONE = "conversation.item.input_audio_transcription.completed"
USER_ASR = (USER_ASR_DELTA, USER_ASR_DONE)
TOOL_DONE = "response.function_call_arguments.done"

# Refusal detector. A fixed list of literal strings was the first attempt and it FAILED on the
# very first run: the model said "I am sorry, but I cannot help with that request. I am unable to
# make modifications to delivered orders", which matched none of ("unable to assist", "I can't
# help", "I cannot assist", "not able to assist") and printed as a clean non-refusal. The refusal
# is paraphrased per domain and per turn, so match the SHAPE (a negated capability verb) and
# print what matched so a wrong classification is visible rather than silent.
REFUSAL_RE = re.compile(
    r"\b(?:cannot|can ?not|can't|unable to|not able to|won't be able to|"
    r"am not permitted to|not authorized to)\s+"
    r"(?:\w+\s+){0,3}?"
    r"(?:help|assist|access|make|modify|process|provide|complete|do|share|change|"
    r"give|look|proceed|fulfill|support)\b",
    re.IGNORECASE,
)

# TWO KINDS OF REFUSAL, and conflating them cost this script its first conclusion. The mode-H
# refusal that killed arm A' is a SAFETY refusal -- it names the request category, not the
# domain state: "requests that involve unauthorized access to user accounts, payment
# information, or personal data" (airline's variant says "fraudulent documents"). What the
# first probe run actually reproduced was a TASK refusal -- "I cannot make any changes to it"
# because the order is already delivered -- which is a wrong-but-ordinary policy application,
# not mode H at all. Both match REFUSAL_RE. Only the first one is the phenomenon under study,
# so the classes are reported separately and the baseline gate tests for the SAFETY class.
SAFETY_RE = re.compile(
    r"unauthorized access|personal data|payment information|fraudulent|"
    r"illegal|privacy|sensitive information|violat\w*|"
    r"unauthorized (?:use|modification)",
    re.IGNORECASE,
)


def loop_stats(text: str, min_repeats: int = 3) -> dict:
    """Does the transcript degenerate into one sentence emitted over and over?

    THIS, not the refusal, is the mode-H phenomenon -- established 2026-08-24, when the real
    7469-char prompt produced no refusal but still looped, and a 131-char benign prompt with no
    domain policy looped too. The refusal was only *what* one server process happened to lock
    onto; the loop is what kills every session. A probe that reports refusals alone therefore
    calls a looping session clean, which is how the first reading of this script went wrong.

    Counts the most frequent sentence rather than adjacency: the container interleaves the
    repeated unit with occasional other output, so a strict "same sentence twice in a row" test
    undercounts. Sentences under 12 chars are ignored -- "Hi!" recurs innocently.
    """
    units = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 12]
    if not units:
        return {"looped": False, "repeats": 0, "unit": None}
    counts: dict[str, int] = {}
    for u in units:
        counts[u] = counts.get(u, 0) + 1
    unit, n = max(counts.items(), key=lambda kv: kv[1])
    return {
        "looped": n >= min_repeats,
        "repeats": n,
        "unit": unit[:160] if n >= min_repeats else None,
    }


def build_prompts():
    """Assemble the prompt variants using tau2's own template, not a hand copy."""
    from tau2.agent.discrete_time_audio_native_agent import (
        AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN,
        AUDIO_NATIVE_VOICE_INSTRUCTION,
    )
    from tau2.domains.retail.environment import get_environment

    env = get_environment()
    policy = env.get_policy()
    tools = env.get_tools()

    def plain(instruction: str, pol: str) -> str:
        return AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN.format(
            agent_instruction=instruction, domain_policy=pol
        )

    baseline = plain(AUDIO_NATIVE_VOICE_INSTRUCTION, policy)

    benign = (
        "You are a helpful customer service agent for an online retailer, speaking with a "
        "customer on the phone. Help them with their order."
    )

    return tools, {
        "baseline": (baseline, True),
        "policy_only": (policy, True),
        "instruction_only": (AUDIO_NATIVE_VOICE_INSTRUCTION, True),
        "benign_short": (benign, True),
        "baseline_no_tools": (baseline, False),
        "truncated_policy": (plain(AUDIO_NATIVE_VOICE_INSTRUCTION, policy[:1500]), True),
    }


def parse_labels(path: Path):
    out = []
    for line in path.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 3:
            try:
                out.append((float(p[0]), float(p[1]), p[2]))
            except ValueError:
                continue
    return out


def load_user_channel(first_turn_only: bool = False) -> tuple[bytes, list[str], float]:
    """The user side of the stimulus episode as 16 kHz mono pcm16, original timing preserved.

    Returns (pcm, turn_texts, seconds). The whole channel is taken verbatim rather than turns
    being spliced together, so the silences between turns are the real ones -- pacing is part
    of what a duplex model responds to, and re-timing it would be an uncontrolled change.

    The user channel is DETECTED, not assumed: per-channel energy inside the user's labelled
    spans is compared against the same measure inside the assistant's. Guessing channel 0 and
    being wrong would replay the *agent's* voice back at it, which would still produce
    transcripts and would look exactly like a result.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    hits = sorted(STIM_RUN.glob(f"artifacts/{STIM_TASK}/sim_*/audio/both.wav"))
    if not hits:
        raise SystemExit(f"no both.wav for {STIM_TASK} under {STIM_RUN}")
    both = hits[0]
    user = parse_labels(both.parent / "user_labels.txt")
    if not user:
        raise SystemExit(f"no user labels next to {both}")

    audio, rate = sf.read(str(both), always_2d=True)
    n_ch = audio.shape[1]
    if n_ch == 1:
        ch = 0
    else:
        def energy(spans):
            tot = 0.0
            for s, e, _ in spans[:4]:
                seg = audio[int(s * rate) : int(e * rate), :]
                if len(seg):
                    tot += np.sqrt(np.mean(seg**2, axis=0))
            return np.atleast_1d(tot)

        asst = parse_labels(both.parent / "assistant_labels.txt")
        u = energy(user)
        ch = int(np.argmax(u / (energy(asst) + 1e-9))) if asst else int(np.argmax(u))

    if first_turn_only:
        lo, hi = int(user[0][0] * rate), int(user[0][1] * rate)
        texts = [user[0][2]]
    else:
        # From 0, not from the first turn, so the model gets the same opening silence the real
        # episode gave it -- that silence is what the container answers with its greeting.
        lo, hi = 0, int(user[-1][1] * rate)
        texts = [t for _, _, t in user]

    clip = audio[lo:hi, ch]
    if rate != INPUT_RATE:
        from math import gcd

        g = gcd(int(rate), INPUT_RATE)
        clip = resample_poly(clip, INPUT_RATE // g, int(rate) // g)

    pcm = (np.clip(clip, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return pcm, texts, len(pcm) / 2 / INPUT_RATE


async def run_condition(url, name, prompt, tools_json, pcm, seconds):
    import websockets

    chunk_bytes = int(INPUT_RATE * CHUNK_MS / 1000) * 2
    silence = b"\x00" * chunk_bytes
    n_ticks = int(seconds * 1000 / CHUNK_MS)

    transcript_parts: list[str] = []
    user_asr_parts: list[str] = []
    event_counts: dict[str, int] = {}
    tool_calls: list[str] = []
    error = None

    async with websockets.connect(url, ping_interval=None, max_size=None) as ws:
        data = json.loads(await ws.recv())
        if data.get("type") != "session.created":
            return {"condition": name, "error": f"expected session.created, got {data.get('type')}"}

        session = {
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": INPUT_RATE}}},
            "instructions": prompt,
            "tools": tools_json,
        }
        await ws.send(json.dumps({"type": "session.update", "session": session}))

        while True:
            data = json.loads(await ws.recv())
            t = data.get("type")
            if t == "session.updated":
                break
            if t == "error":
                return {"condition": name, "error": f"session.update rejected: {data}"}

        async def receiver():
            nonlocal error
            try:
                while True:
                    data = json.loads(await ws.recv())
                    t = data.get("type", "?")
                    event_counts[t] = event_counts.get(t, 0) + 1
                    if t == AGENT_TEXT:
                        transcript_parts.append(data.get("delta") or "")
                    elif t == USER_ASR_DELTA:
                        # Kept separately: useful for checking the model heard the clip at
                        # all, but it is NOT the agent's reply and must not be scored as one.
                        user_asr_parts.append(data.get("delta") or "")
                    elif t == USER_ASR_DONE:
                        user_asr_parts.append(" | ")  # turn boundary, not more text
                    elif t == TOOL_DONE:
                        tool_calls.append(
                            f"{data.get('name')}({data.get('arguments')})"[:300]
                        )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # a server-side abort closes the socket
                error = f"{type(e).__name__}: {e}"

        rx = asyncio.create_task(receiver())

        # Stream the real utterance, then silence, at real time. The model consumes
        # continuously, so silence is what keeps a duplex session alive while it answers.
        stream = pcm + silence * n_ticks
        try:
            for i in range(0, min(len(stream), n_ticks * chunk_bytes), chunk_bytes):
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(stream[i : i + chunk_bytes]).decode(),
                        }
                    )
                )
                await asyncio.sleep(CHUNK_MS / 1000)
        except Exception as e:
            error = error or f"send failed: {type(e).__name__}: {e}"

        await asyncio.sleep(1.0)
        rx.cancel()
        try:
            await rx
        except asyncio.CancelledError:
            pass

    text = "".join(transcript_parts)
    spans = REFUSAL_RE.findall(text)
    safety = SAFETY_RE.findall(text)
    loop = loop_stats(text)
    return {
        "condition": name,
        "prompt_chars": len(prompt),
        "n_tools": len(tools_json),
        "transcript_chars": len(text),
        # The actual mode-H phenomenon; see loop_stats.
        "looped": loop["looped"],
        "loop_repeats": loop["repeats"],
        "loop_unit": loop["unit"],
        "refusals": len(spans),
        "refused": bool(spans),
        "refusal_spans": sorted(set(spans))[:6],
        # The mode-H class specifically: a refusal that cites the request CATEGORY.
        "safety_refusal": bool(safety),
        "safety_spans": sorted(set(s.lower() for s in safety))[:6],
        "tool_calls": len(tool_calls),
        "tool_calls_detail": tool_calls[:5],
        "transcript": text,
        "heard": "".join(user_asr_parts),
        "events": event_counts,
        "error": error,
    }


async def main_async(args) -> int:
    raw = os.environ.get("NEMO_RT_URL")
    if not raw:
        print("set NEMO_RT_URL, e.g. ws://ip-10-1-76-28:9000", file=sys.stderr)
        return 2

    tools, variants = build_prompts()

    # Reuse the provider's own URL rule and tool formatter, so the endpoint and the tool block
    # are byte-identical to the real run rather than a re-derivation that could drift from it.
    # Both are self-independent (`_normalise_url` is a staticmethod; `_format_tools_for_api`
    # reads only `tool.openai_schema`), so neither needs an instance -- and constructing one
    # would open a websocket.
    from tau2.voice.audio_native.nemo_rt.provider import NemotronRealtimeProvider

    url = NemotronRealtimeProvider._normalise_url(raw)
    tools_json = NemotronRealtimeProvider._format_tools_for_api(None, tools)

    base_chars = len(variants["baseline"][0])
    print(f"baseline prompt: {base_chars} chars (server logged {EXPECTED_BASELINE_CHARS} "
          f"for the real run){'  MATCH' if base_chars == EXPECTED_BASELINE_CHARS else '  <-- MISMATCH'}")
    if base_chars != EXPECTED_BASELINE_CHARS:
        print("  The reconstruction differs from what arm A' actually sent. Conditions would\n"
              "  not be comparable to the observed refusal. Fix before trusting any row.",
              file=sys.stderr)

    pcm, texts, clip_s = load_user_channel(first_turn_only=args.first_turn_only)
    # Listen past the end of the clip: the refusal we are hunting arrives in the model's reply,
    # which starts only after the user turn it answers.
    seconds = args.seconds or int(clip_s + 25)
    print(f"stimulus: {STIM_RUN.parent.name}/{STIM_RUN.name} {STIM_TASK}, "
          f"{len(texts)} user turn(s), {clip_s:.1f}s (open-loop replay)")
    print(f"  first: {texts[0]!r}")
    print(f"{len(tools_json)} tools, {seconds}s per condition\n")

    names = [args.condition] if args.condition else list(variants)
    rows = []
    for name in names:
        if name not in variants:
            print(f"unknown condition {name!r}; have {list(variants)}", file=sys.stderr)
            return 2
        prompt, with_tools = variants[name]
        print(f"-- {name} ...", flush=True)
        r = await run_condition(
            url, name, prompt, tools_json if with_tools else [], pcm, seconds
        )
        rows.append(r)
        print(f"   refusals={r.get('refusals')} safety={r.get('safety_refusal')} "
              f"chars={r.get('transcript_chars')} tool_calls={r.get('tool_calls')} "
              f"err={r.get('error')}")

    print(f"\n{'condition':<20} {'prompt':>7} {'tools':>5} {'said':>6} "
          f"{'refuse':>7} {'safety':>7} {'calls':>6} {'loop':>6}")
    print("-" * 76)
    for r in rows:
        print(f"{r['condition']:<20} {r.get('prompt_chars',0):>7} {r.get('n_tools',0):>5} "
              f"{r.get('transcript_chars',0):>6} {str(r.get('refused')):>7} "
              f"{str(r.get('safety_refusal')):>7} {r.get('tool_calls',0):>6} "
              f"{('x' + str(r.get('loop_repeats',0))) if r.get('looped') else '-':>6}")

    # THE LOOP IS THE VERDICT, not the refusal. An earlier version of this script gated on the
    # baseline producing a SAFETY refusal and declared every other row uninterpretable when it
    # did not. That was backwards: the baseline did not refuse and still looped, so the gate was
    # hiding the finding. Refusals are still reported, as a sub-case.
    looped = [r["condition"] for r in rows if r.get("looped")]
    clean = [r["condition"] for r in rows if not r.get("looped")]
    print(f"\nlooped ({len(looped)}/{len(rows)}): {looped or 'none'}")
    print(f"did NOT loop: {clean or 'none'}")
    if len(looped) == len(rows) and len(rows) > 1:
        print("  Every prompt loops, including the short benign one -- so the loop is not\n"
              "  caused by policy length or subject matter. Next suspect is the session\n"
              "  itself (duration, or what the provider does that this replay does not).")

    base = next((r for r in rows if r["condition"] == "baseline"), None)
    if base is not None and not base.get("safety_refusal"):
        print("\nNo SAFETY refusal on the baseline. Expected: as of 2026-08-24 this replay does\n"
              "not reproduce the refusal even with a byte-identical prompt, while the real\n"
              "provider on the same server does. The loop above is the comparable signal.")

    for r in rows:
        print(f"\n--- {r['condition']} ---")
        print(f"agent said : {(r.get('transcript') or '')[:700]!r}")
        print(f"refusal    : {r.get('refusal_spans')}  safety={r.get('safety_spans')}")
        # What the container's own ASR made of our clip. If this is empty the model never
        # heard the stimulus and the condition says nothing about prompts.
        print(f"model heard: {(r.get('heard') or '')[:200]!r}")
        if r.get("tool_calls_detail"):
            print(f"tool calls : {r['tool_calls_detail']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--condition", help="run just one condition")
    ap.add_argument("--seconds", type=int, default=0,
                    help="listen window per condition; default = clip length + 25 s")
    ap.add_argument("--first-turn-only", action="store_true",
                    help="send only the opening user turn (cheap, but see the note in "
                         "load_user_channel: it may not reach the account part of the task, "
                         "and then the safety refusal cannot appear at all)")
    ap.add_argument("--json-out", type=Path)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
