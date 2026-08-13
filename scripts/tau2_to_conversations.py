#!/usr/bin/env python3
"""
Convert tau2-bench tasks into conversation JSON format for training data preparation.

This generates oracle-guided conversations from tau2 task specifications.
Each task's oracle action sequence is expanded into a full conversation
with realistic user/agent dialogue.

Two modes:
  1. template: Generate simple template-based conversations (fast, no LLM needed)
  2. llm: Use an LLM to generate natural conversations around the oracle actions (higher quality)

Output format matches what prepare_lhotse_data.py expects.

Usage:
  # Template mode (fast, for testing pipeline):
  python scripts/tau2_to_conversations.py \
      --tau2_dir /path/to/tau2-bench \
      --domains banking,healthcare \
      --output conversations.json \
      --mode template

  # LLM mode (higher quality, requires API key):
  python scripts/tau2_to_conversations.py \
      --tau2_dir /path/to/tau2-bench \
      --domains banking,healthcare,hotels,calendar,car_rental,events,housing,media,transit,restaurant \
      --output conversations.json \
      --mode llm \
      --model gpt-4o-mini
"""

import argparse
import json
import os
import sys


def load_tau2_tasks(tau2_dir: str, domains: list[str]):
    """Load tasks from tau2-bench for specified domains."""
    sys.path.insert(0, os.path.join(tau2_dir, "src"))

    # Mock audioop for Python 3.13+
    import types
    audioop_mock = types.ModuleType('audioop')
    audioop_mock.ulaw2lin = lambda data, width: data
    audioop_mock.lin2ulaw = lambda data, width: data
    sys.modules['audioop'] = audioop_mock

    from tau2.registry import registry

    all_tasks = {}
    for domain in domains:
        try:
            tasks_loader = registry.get_tasks_loader(domain)
            tasks = tasks_loader()
            all_tasks[domain] = tasks
            print(f"  Loaded {len(tasks)} tasks from domain: {domain}")
        except KeyError:
            print(f"  WARNING: Domain '{domain}' not found in registry, skipping")

    return all_tasks


def task_to_template_conversation(task, domain: str, env_constructor) -> dict:
    """
    Generate a simple template-based conversation from a task.
    Uses the oracle actions to create a minimal but valid conversation.
    """
    ec = task.evaluation_criteria.model_dump()
    actions = ec.get("actions", [])
    user_scenario = task.user_scenario.model_dump() if task.user_scenario else {}
    instructions = user_scenario.get("instructions", {})

    reason_for_call = instructions.get("reason_for_call", "I need help with something.")
    known_info = instructions.get("known_info", "")

    turns = []

    # User's opening statement
    user_opening = reason_for_call
    if known_info:
        user_opening += f" {known_info}"
    turns.append({"role": "user", "text": user_opening})

    # Simulate the conversation based on oracle actions
    for i, action in enumerate(actions):
        tool_name = action["name"]
        tool_args = action.get("arguments", {})

        if "verify" in tool_name:
            # Verification step — agent asks for info, user provides
            if i == 0:
                turns.append({
                    "role": "agent",
                    "text": "I'd be happy to help you with that. First, let me verify your identity. "
                            "Could you please confirm your name and verification details?"
                })
                # User already provided info in opening, skip extra turn
            else:
                turns.append({
                    "role": "agent",
                    "text": "Let me verify that information for you."
                })
        elif "get_" in tool_name or "search" in tool_name or "find" in tool_name or "check" in tool_name:
            # Lookup step — agent looks up information
            turns.append({
                "role": "agent",
                "text": f"Let me look that up for you. I'm checking the {tool_name.replace('_', ' ')} now."
            })
        elif "transfer_to_human" in tool_name:
            # Escalation
            turns.append({
                "role": "agent",
                "text": "I understand your concern. Let me transfer you to a specialist "
                        "who can better assist you with this matter."
            })
        elif any(w in tool_name for w in ["book", "create", "submit", "make", "cancel", "modify",
                                           "transfer", "pay", "freeze", "close", "change",
                                           "add", "remove", "dispute"]):
            # Action step — agent confirms and executes
            action_desc = tool_name.replace("_", " ")
            turns.append({
                "role": "agent",
                "text": f"I'll go ahead and {action_desc} for you now. "
                        "Let me confirm the details are correct."
            })
            turns.append({"role": "user", "text": "Yes, that's correct. Please go ahead."})
            turns.append({
                "role": "agent",
                "text": f"Done! I've successfully completed the {action_desc}. "
                        "Is there anything else I can help you with?"
            })
        else:
            turns.append({
                "role": "agent",
                "text": f"I'm processing your request now."
            })

    # Closing
    if not any("transfer_to_human" in a["name"] for a in actions):
        turns.append({"role": "user", "text": "No, that's all. Thank you!"})
        turns.append({"role": "agent", "text": "You're welcome! Have a great day."})

    # Get system prompt from policy
    env = env_constructor()
    system_prompt = env.policy[:500]  # First 500 chars as summary

    return {
        "id": f"{domain}_task_{task.id}",
        "turns": turns,
        "system_prompt": system_prompt,
        "metadata": {
            "domain": domain,
            "task_id": str(task.id),
            "num_oracle_actions": len(actions),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Convert tau2-bench tasks to conversation format")
    parser.add_argument("--tau2_dir", required=True, help="Path to tau2-bench directory")
    parser.add_argument("--domains", required=True, help="Comma-separated list of domains")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--mode", default="template", choices=["template", "llm"],
                        help="Conversation generation mode")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for llm mode")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]

    print("=" * 60)
    print("TAU2 TO CONVERSATIONS")
    print("=" * 60)
    print(f"  tau2 dir: {args.tau2_dir}")
    print(f"  domains: {domains}")
    print(f"  mode: {args.mode}")
    print(f"  output: {args.output}")

    # Load tasks
    print(f"\nLoading tasks...")
    all_tasks = load_tau2_tasks(args.tau2_dir, domains)

    total_tasks = sum(len(tasks) for tasks in all_tasks.values())
    print(f"  Total tasks: {total_tasks}")

    # Generate conversations
    print(f"\nGenerating conversations (mode: {args.mode})...")
    conversations = []

    sys.path.insert(0, os.path.join(args.tau2_dir, "src"))
    from tau2.registry import registry

    for domain, tasks in all_tasks.items():
        env_constructor = registry.get_env_constructor(domain)
        for task in tasks:
            if args.mode == "template":
                conv = task_to_template_conversation(task, domain, env_constructor)
            else:
                raise NotImplementedError(
                    "LLM mode not yet implemented. Use --mode template for now."
                )
            conversations.append(conv)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(conversations, f, indent=2)

    print(f"\n  Generated {len(conversations)} conversations")
    print(f"  Saved to: {args.output}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for domain in domains:
        count = len([c for c in conversations if c["metadata"]["domain"] == domain])
        print(f"  {domain}: {count} conversations")
    avg_turns = sum(len(c["turns"]) for c in conversations) / len(conversations)
    print(f"\n  Average turns per conversation: {avg_turns:.1f}")
    print(f"\n  Next: python scripts/prepare_lhotse_data.py --input {args.output} --output_dir <dir>")


if __name__ == "__main__":
    main()
