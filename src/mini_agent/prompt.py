"""
ReAct Prompt Template

Format:
  Thought: <reasoning>
  Action: <tool_name>
  Action Input: <JSON arguments>
  Observation: <tool result>
  Answer: <final answer>

Rules:
  - Always output "Thought:" first
  - If you need a tool, output Action + Action Input
  - After Observation, continue reasoning
  - When you have the final answer, output Answer:
  - Do NOT output Action and Answer in the same turn
"""

SYSTEM_PROMPT = """You are an AI agent that solves tasks using a ReAct loop (Reasoning + Acting).

You have access to the following tools:

{tools}

To use a tool, output EXACTLY in this format:

Thought: <what you are thinking, what you need to do next>
Action: <tool_name>
Action Input: <JSON object with arguments>

After the tool returns an Observation, you will see it and can continue reasoning.

When you have enough information to give a final answer, output:

Thought: <why you now have the answer>
Answer: <your final answer to the user>

Rules:
1. Always start with "Thought:"
2. Output "Action:" and "Action Input:" together when calling a tool
3. NEVER output "Answer:" and "Action:" in the same response
4. Wait for Observation before giving Answer
5. Be concise in your reasoning

Begin!"""

TOOL_DESC_TEMPLATE = "{name}: {description}"


def format_system_prompt(tools: dict) -> str:
    """Format system prompt with tool descriptions."""
    if not tools:
        tool_text = "No tools available. Answer directly."
    else:
        tool_lines = []
        for name, meta in tools.items():
            if isinstance(meta, dict):
                desc = meta.get("description", "")
                schema = meta.get("schema", "")
                tool_lines.append(f"- {name}: {desc}  Schema: {schema}")
            else:
                tool_lines.append(f"- {name}: {meta}")
        tool_text = "\n".join(tool_lines)
    return SYSTEM_PROMPT.format(tools=tool_text)


def format_user_prompt(user_input: str) -> str:
    """Wrap user input for the first turn."""
    return user_input