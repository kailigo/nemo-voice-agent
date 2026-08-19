"""The 12 FDB-v3 tools and the agent instructions, extracted from the benchmark itself.

Why extract instead of transcribe
---------------------------------
Reproducing a published number means giving our model the *same* tool block and the
*same* instructions every other provider in the table got. Those live in the benchmark's
LiveKit agent (``v3/lk_agent_tool.py``): the tool descriptions sit in
``@llm.function_tool(description=...)`` decorators, the parameter descriptions in each
method's ``Args:`` docstring, and the agent instructions in ``VoiceAgent.__init__``.

Hand-copying all of that into our repo would work exactly once. The moment NTU edits a
description, our copy becomes a silent, unattributable difference between our number and
theirs -- and "the prompt was subtly not theirs" is the single most expensive class of
error in this whole eval (see TAU_VOICE_SFT_PLAN 1c-bis, which cost ~10 GPU-hours). So we
parse their file with ``ast`` instead and synthesise the tools from it. No import: their
module pulls in ``livekit`` and constructs a ``MockAPIRegistry`` at import time.

If they restructure that file, this raises. That is the point -- a loud failure beats a
prompt that quietly stops matching the published setup.

The synthesised functions are wrapped in tau2's :class:`Tool`, so ``openai_schema`` comes
from the same code path that generated the tool block in our training data. Two prompt
formats built by two different serialisers is the other way to get a mismatch.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

FDB_V3_DIR = Path("/fsx/home/kai.li/code/Full-Duplex-Bench/v3")

# Names the benchmark's MockAPIRegistry.FUNCTIONS exposes (mock_apis.py:56-69). Used only
# to assert the extraction found exactly the advertised tool set.
EXPECTED_TOOLS = (
    "search_flights",
    "book_flight",
    "update_identity_doc",
    "get_card_benefits",
    "get_exchange_rate",
    "modify_autopay",
    "search_apartments",
    "calculate_commute",
    "update_search_filter",
    "track_order",
    "search_products",
    "add_to_cart",
)

# ast annotation name -> the type object the synthesised signature should carry. Only the
# types lk_agent_tool.py actually annotates with; anything else raises rather than being
# quietly widened to Any, because a widened type changes the JSON schema the model sees.
_TYPES: Dict[str, Any] = {"str": str, "int": int, "float": float, "bool": bool}


def _agent_source(agent_file: Path | None = None) -> ast.Module:
    path = agent_file or (FDB_V3_DIR / "lk_agent_tool.py")
    if not path.exists():
        raise FileNotFoundError(
            f"The benchmark's agent file is missing: {path}. This module derives the tool "
            f"schemas and instructions from it; without it we would be guessing at the "
            f"prompt the published numbers were produced with."
        )
    return ast.parse(path.read_text(encoding="utf-8"))


def extract_instructions(agent_file: Path | None = None) -> str:
    """Return ``VoiceAgent.__init__``'s ``instructions=`` string, concatenated verbatim.

    It is written as a run of adjacent string literals, which the parser folds for us.
    """
    tree = _agent_source(agent_file)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "VoiceAgent"):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            for kw in call.keywords:
                if kw.arg == "instructions":
                    value = ast.literal_eval(kw.value)
                    if not isinstance(value, str) or len(value) < 100:
                        raise ValueError(
                            f"VoiceAgent instructions parsed to something unexpected: "
                            f"{value!r:.120}"
                        )
                    return value
    raise ValueError(
        "Could not find VoiceAgent(instructions=...) in the benchmark's agent file. "
        "The file has been restructured; re-read it before running an eval."
    )


def _parse_tool_methods(agent_file: Path | None = None) -> List[Tuple[str, str, str, List[Tuple[str, Any, Any]]]]:
    """Return ``(name, description, args_doc, params)`` for each decorated tool method.

    ``params`` is ``[(param_name, type_object, default)]`` with ``default`` set to the
    sentinel :data:`Ellipsis` for required parameters, matching what ``Tool`` expects.
    """
    tree = _agent_source(agent_file)
    cls = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "AssistantFnc"),
        None,
    )
    if cls is None:
        raise ValueError("class AssistantFnc not found in the benchmark's agent file.")

    tools = []
    for fn in cls.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        description = None
        for dec in fn.decorator_list:
            if isinstance(dec, ast.Call):
                for kw in dec.keywords:
                    if kw.arg == "description":
                        description = ast.literal_eval(kw.value)
        if description is None:
            continue  # log_tool_call and friends: not tools

        doc = ast.get_docstring(fn) or ""

        args = fn.args
        # Defaults align to the tail of the positional argument list.
        pad = [Ellipsis] * (len(args.args) - len(args.defaults))
        defaults = pad + [ast.literal_eval(d) for d in args.defaults]

        params: List[Tuple[str, Any, Any]] = []
        for arg, default in zip(args.args, defaults):
            if arg.arg == "self":
                continue
            if arg.annotation is None:
                raise ValueError(f"{fn.name}({arg.arg}) has no type annotation.")
            anno = ast.unparse(arg.annotation)
            if anno not in _TYPES:
                raise ValueError(
                    f"{fn.name}({arg.arg}: {anno}) uses a type this extractor does not "
                    f"map. Add it to _TYPES rather than letting it widen to Any."
                )
            params.append((arg.arg, _TYPES[anno], default))

        tools.append((fn.name, description, doc, params))
    return tools


def _synthesise(name: str, description: str, args_doc: str, params) -> Callable:
    """Build a plain function whose signature and docstring match the benchmark's.

    ``Tool`` reads schemas off a real signature and a google-style docstring, so the
    cheapest faithful route is to generate the source and exec it.
    """
    sig_parts = []
    for pname, ptype, default in params:
        if default is Ellipsis:
            sig_parts.append(f"{pname}: {ptype.__name__}")
        else:
            sig_parts.append(f"{pname}: {ptype.__name__} = {default!r}")

    doc = description if not args_doc.strip() else f"{description}\n\n{args_doc.strip()}"
    src = (
        f"def {name}({', '.join(sig_parts)}) -> dict:\n"
        f"    {chr(34) * 3}{doc}\n    {chr(34) * 3}\n"
        f"    raise NotImplementedError('schema-only; execution goes through MockAPIRegistry')\n"
    )
    ns: Dict[str, Any] = {}
    exec(compile(textwrap.dedent(src), f"<fdb_v3_tool:{name}>", "exec"), ns)
    return ns[name]


def build_tools(agent_file: Path | None = None) -> List[Any]:
    """Return tau2 ``Tool`` objects for the 12 benchmark tools, in declaration order."""
    from tau2.environment.tool import Tool

    parsed = _parse_tool_methods(agent_file)
    names = tuple(p[0] for p in parsed)
    if set(names) != set(EXPECTED_TOOLS):
        missing = sorted(set(EXPECTED_TOOLS) - set(names))
        extra = sorted(set(names) - set(EXPECTED_TOOLS))
        raise ValueError(
            f"Extracted tool set does not match the benchmark's 12 tools. "
            f"missing={missing} unexpected={extra}"
        )
    return [Tool(_synthesise(*p)) for p in parsed]


# NVIDIA's own inference-time FC prompt renderer and its default system message, both from
# `examples/speechlm2/offline_voicechat_fc_infer.py` -- the entrypoint the model card points
# at for offline function calling.
_NEMO_REPO = Path("/fsx/home/kai.li/code/nemo-voice-agent")
FC_TEMPLATE = _NEMO_REPO / "examples/speechlm2/function_calling/template.jinja"


def nvidia_default_system_message() -> str:
    """``DEFAULT_SYSTEM_MESSAGE`` from NVIDIA's offline FC entrypoint, read off the file.

    Not used by default: FDB-v3's published table gave every provider the benchmark's own
    ``VoiceAgent`` instructions, and prepending NVIDIA's persona and decision-process text
    would be a prompt no other row in that table saw. It is here because NVIDIA evaluating
    their own model plausibly *did* include it, which makes it worth one control arm rather
    than a guess.
    """
    script = _NEMO_REPO / "examples/speechlm2/offline_voicechat_fc_infer.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEFAULT_SYSTEM_MESSAGE" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"DEFAULT_SYSTEM_MESSAGE not found in {script}.")


def render_system_prompt(system_message: str, tools, template: Path | None = None) -> str:
    """Render the FC system prompt through NVIDIA's own Jinja template.

    Why not our own serialiser: the template does three things ``Tool.openai_schema`` does
    not, and the difference is visible to the model.

      ours   : {"type":"function","function":{"name":"track_order","description":...}}
      theirs : {"description": ..., "name": "track_order", "parameters": {...}}

    It unwraps ``function``, drops the ``type`` key, sorts the JSON keys (Jinja's ``tojson``)
    and separates entries with ``", "`` rather than ``","``. NeMo's own example of a trained
    tool block (``s2s_dataset.py:1159``) is the flattened form too, so the wrapped form is
    not what the checkpoint saw at training *or* inference time. Prompt-format drift is the
    error class this whole module exists to prevent, so the rendering is theirs, from their
    file, not a reimplementation of it.

    ``system_message`` goes in ahead of the tool block; the template supplies the
    <TOOLCALL>/<TOOL_RESPONSE> protocol paragraphs itself.
    """
    from jinja2 import Environment

    path = template or FC_TEMPLATE
    if not path.exists():
        raise FileNotFoundError(
            f"NVIDIA's FC prompt template is missing: {path}. Rendering the tool block "
            f"ourselves instead would silently change the prompt format."
        )
    schemas = [t.openai_schema if hasattr(t, "openai_schema") else t for t in tools]
    rendered = Environment().from_string(path.read_text()).render(
        system_message=system_message, tools=schemas
    )
    if "<AVAILABLE_TOOLS>" not in rendered:
        raise ValueError(
            "Rendered prompt has no <AVAILABLE_TOOLS> block -- the template's `tools` "
            "contract has changed. Re-read it before running an eval."
        )
    return rendered


def build_registry(latency_profile: str = "instant"):
    """The benchmark's own ``MockAPIRegistry``, imported from its directory.

    ``instant`` is the reference default: ``lk_agent_tool.py`` sets
    ``LATENCY_PROFILE = "instant"`` and only overrides it from a ``--latency`` flag that
    the released ``run_agent.sh`` never passes. The per-scenario ``latency_profile`` field
    in the metadata is not read by anything in the released pipeline.
    """
    import sys

    if str(FDB_V3_DIR) not in sys.path:
        sys.path.insert(0, str(FDB_V3_DIR))
    from mock_apis import MockAPIRegistry

    return MockAPIRegistry(latency_profile=latency_profile)


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, "/fsx/home/kai.li/code/tau-voice-2/src")
    tools = build_tools()
    print(f"{len(tools)} tools extracted\n")
    for t in tools:
        print(json.dumps(t.openai_schema, indent=2))
    instr = extract_instructions()
    print(f"\ninstructions: {len(instr)} chars, ascii={instr.isascii()}\n{instr}")
