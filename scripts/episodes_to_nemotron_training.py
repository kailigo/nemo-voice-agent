#!/usr/bin/env python3
"""
Convert tau2-voice simulation episodes into Lhotse Shar training data
formatted for Nemotron VoiceChat DuplexS2SDataset with function calling.

This produces cuts with:
  - recording: user audio @ 16kHz
  - custom["target_audio"]: agent audio @ 22050Hz
  - custom["system_prompt"]: policy + tool schemas in <AVAILABLE_TOOLS> format
  - supervisions: speech segments (user/agent) + function call/response pairs

The function calling format follows NeMo's convention:
  - Supervision with speaker="system", text=<system_prompt> (index 0)
  - Speech supervisions with speaker="user"/"agent" and text=transcript
  - FC supervisions with custom["function"] containing:
    - Agent calls: <TOOLCALL>[{"name": "...", "arguments": {...}}]</TOOLCALL>
    - Tool responses: <TOOL_RESPONSE>[{"content": "..."}]</TOOL_RESPONSE>

Usage:
  python scripts/episodes_to_nemotron_training.py \
      --sim_dir data/simulations \
      --output /tmp/tau2_nemotron_training \
      --min_reward 1.0 \
      --tool_schemas ../tau-voice-2/data/tool_schemas.json

ALWAYS PASS --tool_schemas. Without it the tool block is reconstructed from the calls the
teacher happened to make (see extract_tool_schemas), which advertises only the tools the
model is about to call, with no descriptions -- teaching "call what's in the prompt"
instead of tool selection, and quietly invalidating the cross-domain experiment. Generate
the sidecar with tau-voice-2/scripts/export_tool_schemas.py.
"""

import argparse
import json
import os
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

USER_TARGET_SR = 16000
AGENT_TARGET_SR = 22050


def parse_labels(path: str) -> list[dict]:
    """Parse Audacity label file into list of {start, end, text}."""
    labels = []
    if not os.path.exists(path):
        return labels
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) < 3:
                parts = line.split(None, 2)
            if len(parts) >= 3:
                labels.append({
                    "start": float(parts[0]),
                    "end": float(parts[1]),
                    "text": parts[2],
                })
    return labels


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = gcd(target_sr, orig_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def discover_episodes(sim_dir: str) -> list[dict]:
    """Walk simulation directories and collect episode metadata."""
    episodes = []
    sim_path = Path(sim_dir)

    for run_dir in sorted(sim_path.iterdir()):
        if not run_dir.is_dir():
            continue
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            results = json.load(f)

        sim_index = results.get("simulation_index")
        if not sim_index:
            continue

        # The domain is recorded once per run, unconditionally, in EnvironmentInfo
        # (tau-voice-2/src/tau2/environment/environment.py:27). Its sibling `tool_defs` is
        # only populated when get_info(include_tool_info=True), which the orchestrator does
        # not pass -- hence the schema sidecar rather than reading schemas from here.
        env_info = (results.get("info") or {}).get("environment_info") or {}
        domain = env_info.get("domain_name")
        run_policy = env_info.get("policy", "")

        for sim_entry in sim_index:
            sim_id = sim_entry["id"]
            task_id = sim_entry["task_id"]
            reward = sim_entry.get("reward")

            audio_dir = run_dir / "artifacts" / f"task_{task_id}" / f"sim_{sim_id}" / "audio"
            sim_json = run_dir / "simulations" / f"{sim_id}.json"

            if not audio_dir.exists() or not (audio_dir / "both.wav").exists():
                continue

            episodes.append({
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "sim_id": sim_id,
                "task_id": task_id,
                "reward": reward,
                "termination_reason": sim_entry.get("termination_reason"),
                "audio_dir": str(audio_dir),
                "sim_json_path": str(sim_json) if sim_json.exists() else None,
                "domain": domain,
                "run_policy": run_policy,
            })

    return episodes


def extract_tool_interactions(sim_data: dict) -> list[dict]:
    """
    Extract tool call/result pairs from simulation ticks.
    Returns list of {time, name, arguments, result, result_time}.
    """
    tick_duration = 0.2
    interactions = []

    results_by_id = {}
    for tick in sim_data.get("ticks", []):
        for tr in tick.get("agent_tool_results") or []:
            results_by_id[tr["id"]] = {
                "content": tr.get("content", ""),
                "error": tr.get("error", False),
                "tick_id": tick["tick_id"],
            }

    for tick in sim_data.get("ticks", []):
        for tc in tick.get("agent_tool_calls") or []:
            entry = {
                "time": tick["tick_id"] * tick_duration,
                "name": tc["name"],
                "arguments": tc.get("arguments", {}),
            }
            result = results_by_id.get(tc["id"])
            if result:
                entry["result"] = result["content"]
                entry["error"] = result["error"]
                entry["result_time"] = result["tick_id"] * tick_duration
            else:
                entry["result"] = ""
                entry["error"] = False
                entry["result_time"] = entry["time"]
            interactions.append(entry)

    return interactions


def load_tool_schemas(path: str | None) -> dict:
    """Load the per-domain schema sidecar written by tau-voice-2/scripts/export_tool_schemas.py.

    Shape: {"<domain>": {"tools": [<openai_schema>], "policy": str, "n_tools": int, ...}}.
    Returns {} when no path is given, so the caller falls back to extract_tool_schemas.
    """
    if not path:
        return {}
    with open(path) as f:
        schemas = json.load(f)
    print(f"  Tool schemas: {len(schemas)} domain(s) from {path}")
    for name in sorted(schemas):
        print(f"    {name:20s} {schemas[name]['n_tools']:3d} tools")
    return schemas


def extract_tool_schemas(sim_data: dict) -> list[dict]:
    """
    FALLBACK ONLY -- prefer the --tool_schemas sidecar. See the module docstring.

    Extract tool schemas from the tool calls seen in the simulation. Since the sim JSON does
    not carry schemas (EnvironmentInfo.tool_defs is None unless include_tool_info=True), we
    reconstruct minimal OpenAI-format schemas from the observed calls. That is lossy in three
    ways that matter: only the called tools appear (5 of retail's 16), there are no
    descriptions, and every observed argument is marked required.
    """
    seen = {}
    for tick in sim_data.get("ticks", []):
        for tc in tick.get("agent_tool_calls") or []:
            name = tc["name"]
            if name not in seen:
                args = tc.get("arguments", {})
                properties = {}
                for k, v in args.items():
                    if isinstance(v, str):
                        properties[k] = {"type": "string"}
                    elif isinstance(v, bool):
                        properties[k] = {"type": "boolean"}
                    elif isinstance(v, int):
                        properties[k] = {"type": "integer"}
                    elif isinstance(v, float):
                        properties[k] = {"type": "number"}
                    elif isinstance(v, list):
                        properties[k] = {"type": "array", "items": {"type": "string"}}
                    else:
                        properties[k] = {"type": "string"}
                seen[name] = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": list(args.keys()),
                        }
                    }
                }
    return list(seen.values())


def format_system_prompt_with_tools(policy: str, tool_schemas: list[dict]) -> str:
    """
    Format system prompt with tool descriptions in the expected training format.
    """
    tools_json = json.dumps(tool_schemas, separators=(",", ":"))
    available_tools = f"<AVAILABLE_TOOLS>{tools_json}</AVAILABLE_TOOLS>"
    return f"{policy}\n\n{available_tools}"


def format_toolcall(name: str, arguments: dict) -> str:
    """Format a tool call in TOOLCALL tags."""
    call_obj = [{"name": name, "arguments": arguments}]
    return f"<TOOLCALL>{json.dumps(call_obj)}</TOOLCALL>"


def format_tool_response(content: str, unnest: bool = True) -> str:
    """Format a tool response in TOOL_RESPONSE tags.

    tau2 tool results arrive as JSON *strings*. Wrapping one in {"content": <str>} and
    re-encoding escapes every quote, so the payload is JSON-encoded twice for no added
    information. Measured over the 3 prototype cuts, that double encoding costs ~677 tokens
    per cut -- 18.3% of their total FC prompt tokens -- and tool responses are 84% of all
    segment cost (scripts/measure_fc_token_budget.py). Parsing the string back and letting
    json.dumps emit it once is lossless.

    Only dict/list payloads are un-nested. A bare "123" or "true" would parse to an int/bool
    and silently change type, and plain-text results (error messages) are not JSON at all --
    both keep the string form.

    Note the asymmetry with format_toolcall, which deliberately keeps default separators:
    tool calls are what the model *generates*, so their surface form is a learned target and
    not worth perturbing to save ~0 tokens. Responses are what the model *reads*.
    """
    payload = content
    if unnest and isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            payload = parsed
    resp_obj = [{"content": payload}]
    return f"<TOOL_RESPONSE>{json.dumps(resp_obj, separators=(',', ':'))}</TOOL_RESPONSE>"


def build_training_cut(
    episode: dict,
    output_dir: str,
    tool_schemas_by_domain: dict | None = None,
    unnest_responses: bool = True,
):
    """
    Build a Lhotse MonoCut with full FC training format from an episode.
    """
    from lhotse import Recording, SupervisionSegment, MonoCut

    audio_dir = episode["audio_dir"]
    wav_path = os.path.join(audio_dir, "both.wav")

    # Load stereo, split channels, resample
    data, orig_sr = sf.read(wav_path, dtype="float32", always_2d=True)
    user_audio = resample_audio(data[:, 0], orig_sr, USER_TARGET_SR)
    agent_audio = resample_audio(data[:, 1], orig_sr, AGENT_TARGET_SR)
    duration = data.shape[0] / orig_sr

    # Write audio files
    audio_out = os.path.join(output_dir, "audio")
    os.makedirs(audio_out, exist_ok=True)
    cut_id = f"{episode['task_id']}_{episode['sim_id'][:8]}"

    user_path = os.path.join(audio_out, f"{cut_id}_user.wav")
    agent_path = os.path.join(audio_out, f"{cut_id}_agent.wav")
    sf.write(user_path, user_audio, USER_TARGET_SR)
    sf.write(agent_path, agent_audio, AGENT_TARGET_SR)

    user_recording = Recording.from_file(user_path, recording_id=f"{cut_id}_user")
    agent_recording = Recording.from_file(agent_path, recording_id=f"{cut_id}_agent")

    # Parse speech labels
    user_labels = parse_labels(os.path.join(audio_dir, "user_labels.txt"))
    agent_labels = parse_labels(os.path.join(audio_dir, "assistant_labels.txt"))

    # Load simulation data for policy + tool interactions
    sim_data = {}
    if episode["sim_json_path"]:
        with open(episode["sim_json_path"]) as f:
            sim_data = json.load(f)

    # Prefer the run's own policy (a run may override the domain default); fall back to the
    # one recorded in EnvironmentInfo.
    policy = sim_data.get("policy") or episode.get("run_policy") or ""
    tool_interactions = extract_tool_interactions(sim_data)

    # Canonical schemas from the sidecar, keyed by the episode's domain. The lossy
    # reconstruction is the fallback and says so loudly, because a silent fallback is exactly
    # how the truncated 5-tool retail prompt got into the prototype shards.
    domain = episode.get("domain")
    schemas_by_domain = tool_schemas_by_domain or {}
    if domain and domain in schemas_by_domain:
        tool_schemas = schemas_by_domain[domain]["tools"]
        episode["schema_source"] = f"sidecar:{domain}"
    else:
        tool_schemas = extract_tool_schemas(sim_data)
        why = "no --tool_schemas given" if not schemas_by_domain else f"domain {domain!r} not in sidecar"
        print(f"\n    [WARN] reconstructing {len(tool_schemas)} schemas from observed calls ({why})")
        episode["schema_source"] = "reconstructed"

    # Build system prompt with tools
    system_prompt = format_system_prompt_with_tools(policy, tool_schemas)

    # Build supervisions list
    supervisions = []
    sup_idx = 0

    # Supervision 0: system prompt
    #
    # NOTE the custom={"function": ""} on this and every speech supervision below. It looks
    # redundant but it is REQUIRED, because s2s_dataset.py gates all function-call extraction on
    # supervisions[1] specifically (~L1593):
    #
    #     if len(cut.supervisions) > 1 and 'function' in cut.supervisions[1].custom:
    #
    # supervisions[1] here is a speech turn, and lhotse's SupervisionSegment.custom defaults to
    # None, so `'function' in None` raises TypeError and the whole batch dies in the dataloader.
    # The very next line then calls sup.custom.get("function") across ALL of supervisions[1:], so
    # it is not enough to patch index 1 alone — every supervision needs the dict.
    #
    # An empty string is the safe value: the text-channel inclusion rule (~L2274) keeps a turn
    # when custom['function'] == '', so speech turns still train the text channel normally.
    supervisions.append(SupervisionSegment(
        id=f"{cut_id}_sup_{sup_idx:03d}",
        recording_id=user_recording.id,
        start=0.0,
        duration=0.0,
        text=system_prompt,
        speaker="system",
        custom={"function": ""},
    ))
    sup_idx += 1

    # Speech supervisions (user + agent)
    speech_segments = []
    for label in user_labels:
        speech_segments.append({
            "start": label["start"],
            "end": label["end"],
            "text": label["text"],
            "speaker": "user",
        })
    for label in agent_labels:
        speech_segments.append({
            "start": label["start"],
            "end": label["end"],
            "text": label["text"],
            "speaker": "agent",
        })
    speech_segments.sort(key=lambda s: s["start"])

    for seg in speech_segments:
        supervisions.append(SupervisionSegment(
            id=f"{cut_id}_sup_{sup_idx:03d}",
            recording_id=user_recording.id,
            start=seg["start"],
            duration=seg["end"] - seg["start"],
            text=seg["text"],
            speaker=seg["speaker"],
            custom={"function": ""},  # required, not redundant -- see supervision 0 above
        ))
        sup_idx += 1

    # Function calling supervisions (call + response pairs)
    fc_supervisions = []
    for tc in tool_interactions:
        # Tool call (from agent)
        call_text = format_toolcall(tc["name"], tc["arguments"])
        fc_supervisions.append(SupervisionSegment(
            id=f"{cut_id}_sup_{sup_idx:03d}",
            recording_id=user_recording.id,
            start=tc["time"],
            duration=0.0,
            text="",
            speaker="agent",
            custom={"function": call_text},
        ))
        sup_idx += 1

        # Tool response (to agent)
        response_text = format_tool_response(tc["result"], unnest=unnest_responses)
        fc_supervisions.append(SupervisionSegment(
            id=f"{cut_id}_sup_{sup_idx:03d}",
            recording_id=user_recording.id,
            start=tc["result_time"],
            duration=0.0,
            text="",
            speaker="user",
            custom={"function": response_text},
        ))
        sup_idx += 1

    # Combine all supervisions
    all_supervisions = supervisions + fc_supervisions

    # Create MonoCut
    cut = MonoCut(
        id=cut_id,
        start=0.0,
        duration=duration,
        channel=0,
        recording=user_recording,
        supervisions=all_supervisions,
        custom={
            "target_audio": agent_recording,
            "system_prompt": system_prompt,
        },
    )
    # Mark as function calling cut
    cut.s2s_duplex_function_calling = True

    return cut


def validate_training_cut(cut) -> tuple[bool, str]:
    """Validate a cut meets DuplexS2SDataset + FC requirements."""
    issues = []

    if cut.recording is None:
        issues.append("missing recording")
    elif cut.recording.sampling_rate != USER_TARGET_SR:
        issues.append(f"user SR={cut.recording.sampling_rate}")

    target = cut.custom.get("target_audio")
    if target is None:
        issues.append("missing target_audio")
    elif target.sampling_rate != AGENT_TARGET_SR:
        issues.append(f"agent SR={target.sampling_rate}")

    if not cut.supervisions:
        issues.append("no supervisions")
    elif cut.supervisions[0].speaker != "system":
        issues.append("first supervision not system")

    # Check for FC content
    fc_sups = [s for s in cut.supervisions if s.custom and s.custom.get("function")]
    if not fc_sups:
        issues.append("no function calling supervisions")

    # Check system prompt has tools
    sys_prompt = cut.custom.get("system_prompt", "")
    if "<AVAILABLE_TOOLS>" not in sys_prompt:
        issues.append("system prompt missing AVAILABLE_TOOLS")

    if issues:
        return False, "; ".join(issues)
    return True, "ok"


def export_to_shar(cuts, output_dir: str, shard_size: int = 1) -> str:
    """Export cuts to Lhotse Shar format.

    shard_size drives how many shards you get, and that caps data-parallel width: Lhotse hands
    whole shards to ranks, so a run on N GPUs needs >= N shards or the surplus ranks sit idle
    (and NeMo errors when a rank gets nothing). Default 1 cut/shard maximises parallelism, which
    is what you want for the small prototyping sets; raise it for real corpora where thousands of
    single-cut shards would be wasteful.

    Nothing except the Shar files may live in shard_dir. Lhotse infers field names from EVERY
    entry in the directory (lhotse/shar/readers/lazy.py ~L202,
    `fields = set(p.stem.split(".")[0] for p in in_dir.glob("*"))`), so a stray metadata.json
    becomes a phantom field named "metadata" and is then parsed as JSONL -- which fails with a
    confusing `JSONDecodeError: Expecting property name ... line 2 column 1 (char 2)` at the first
    training batch. That is why metadata.json is written to output_dir, one level up.
    """
    from lhotse import CutSet

    shar_dir = os.path.join(output_dir, "shards")
    os.makedirs(shar_dir, exist_ok=True)

    cutset = CutSet.from_cuts(cuts)
    cutset.to_shar(
        output_dir=shar_dir,
        fields={"recording": "wav", "target_audio": "wav"},
        shard_size=shard_size,
    )
    return shar_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert tau2-voice episodes to Nemotron VoiceChat training format"
    )
    parser.add_argument("--sim_dir", required=True, help="Path to data/simulations directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--min_reward", type=float, default=1.0, help="Minimum reward filter")
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1,
        help="Cuts per shard. Shard count caps data-parallel width, so keep it low enough that "
        "num_cuts/shard_size >= your GPU count (default 1 = one cut per shard).",
    )
    parser.add_argument(
        "--tool_schemas",
        default=None,
        help="JSON sidecar of canonical per-domain tool schemas, from "
        "tau-voice-2/scripts/export_tool_schemas.py. Strongly recommended: without it the "
        "tool block is reconstructed from observed calls only (lossy -- see module docstring).",
    )
    parser.add_argument(
        "--no_unnest_responses",
        action="store_true",
        help="Keep tool-response payloads double JSON-encoded (the old behaviour). Costs ~18%% "
        "more FC prompt tokens for no information; only useful for reproducing old shards.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Just discover episodes")
    args = parser.parse_args()

    print("=" * 60)
    print("EPISODES TO NEMOTRON TRAINING FORMAT")
    print("=" * 60)
    print(f"  sim_dir: {args.sim_dir}")
    print(f"  output: {args.output}")
    print(f"  min_reward: {args.min_reward}")

    tool_schemas_by_domain = load_tool_schemas(args.tool_schemas)
    if not tool_schemas_by_domain:
        print(
            "  [WARN] no --tool_schemas: tool blocks will be reconstructed from observed\n"
            "         calls, which advertises only the tools each episode happens to call,\n"
            "         with no descriptions. See the module docstring."
        )

    # Discover
    print("\nDiscovering episodes...")
    episodes = discover_episodes(args.sim_dir)
    print(f"  Found {len(episodes)} total episodes")

    passing = [e for e in episodes if e["reward"] is not None and e["reward"] >= args.min_reward]
    print(f"  Passing (reward >= {args.min_reward}): {len(passing)}")

    if args.dry_run:
        for ep in passing:
            print(
                f"    task={ep['task_id']} sim={ep['sim_id'][:8]} reward={ep['reward']} "
                f"domain={ep.get('domain')}"
            )
        missing = sorted({e.get("domain") for e in passing} - set(tool_schemas_by_domain))
        if missing and tool_schemas_by_domain:
            print(f"  [WARN] domains absent from the sidecar: {missing}")
        return

    if not passing:
        print("No passing episodes. Nothing to convert.")
        return

    # Convert
    print(f"\nConverting {len(passing)} episodes...")
    cuts = []
    for i, episode in enumerate(passing):
        print(f"  [{i+1}/{len(passing)}] task={episode['task_id']} sim={episode['sim_id'][:8]}...", end=" ")
        try:
            cut = build_training_cut(
                episode,
                args.output,
                tool_schemas_by_domain=tool_schemas_by_domain,
                unnest_responses=not args.no_unnest_responses,
            )
            valid, msg = validate_training_cut(cut)
            if valid:
                cuts.append(cut)
                fc_count = len([s for s in cut.supervisions if s.custom and s.custom.get("function")])
                speech_count = len([s for s in cut.supervisions if s.speaker in ("user", "agent") and not (s.custom and s.custom.get("function"))])
                print(f"OK ({cut.duration:.1f}s, {speech_count} speech segs, {fc_count//2} tool calls)")
            else:
                print(f"INVALID: {msg}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n  Valid cuts: {len(cuts)}/{len(passing)}")

    if not cuts:
        print("No valid cuts.")
        return

    # Export
    os.makedirs(args.output, exist_ok=True)
    print(f"\nExporting to Lhotse Shar...")
    shar_dir = export_to_shar(cuts, args.output, shard_size=args.shard_size)
    print(f"  Exported to: {shar_dir}")

    # Metadata
    total_duration = sum(c.duration for c in cuts)
    total_fc = sum(
        len([s for s in c.supervisions if s.custom and s.custom.get("function")]) // 2
        for c in cuts
    )
    metadata = {
        "num_cuts": len(cuts),
        "total_duration_seconds": total_duration,
        "total_duration_hours": total_duration / 3600,
        "total_tool_calls": total_fc,
        "user_sample_rate": USER_TARGET_SR,
        "agent_sample_rate": AGENT_TARGET_SR,
        "min_reward_filter": args.min_reward,
        "format": "nemotron_voicechat_duplex_s2s_fc",
        "shar_path": shar_dir,
        # Provenance: which domains are in here, and whether the tool blocks are canonical.
        # A cut built with reconstructed schemas is not usable for the cross-domain arm.
        "domains": sorted({e.get("domain") for e in passing if e.get("domain")}),
        "tool_schemas_sidecar": args.tool_schemas,
        "schema_sources": sorted({e.get("schema_source") for e in passing if e.get("schema_source")}),
        "unnest_tool_responses": not args.no_unnest_responses,
    }
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Cuts: {len(cuts)}")
    print(f"  Duration: {total_duration:.1f}s ({total_duration/3600:.3f}h)")
    print(f"  Tool calls: {total_fc}")
    print(f"  Shar: {shar_dir}")
    print(f"\n  Training config:")
    print(f"    data.train_ds.input_cfg.0.shar_path={shar_dir}")


if __name__ == "__main__":
    main()
