"""
Agent ReAct Loop (~90 lines)

Agent is stateful: holds message history across turns.
Supports single-turn and multi-turn via reuse.
verbose=True prints step-by-step ReAct trace.
"""

import json
import re
from .llm import LLMProvider
from .tools import TOOLS, execute_tool


class Agent:
    def __init__(self, llm: LLMProvider, tools=None, max_steps=8, system_prompt=None, verbose=False):
        self.llm = llm
        self.tools = tools or TOOLS
        self.max_steps = max_steps
        self.verbose = verbose
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        # Truncate history to prevent context overflow
        if len(self.messages) > 14:
            system_msgs = [m for m in self.messages if m["role"] == "system"]
            recent = self.messages[-12:]
            self.messages = system_msgs + recent

        result = self._react_loop(self.messages)
        self.messages.append({"role": "assistant", "content": result})
        return result

    def reset(self):
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        self.messages = list(system_msgs)

    def _react_loop(self, messages: list) -> str:
        if self.verbose:
            print(f"  📋 Messages count: {len(messages)}")
            for m in messages:
                print(f"  [{m['role']}] {str(m['content'])[:80]}...")
        for step in range(self.max_steps):
            output = self.llm.chat(messages)

            # Explicit final answer
            if "Answer:" in output:
                answer = self._extract_answer(output)
                if self.verbose:
                    print(f"  🎯 Answer: {answer}")
                return answer

            # Try to parse tool call
            action_match = re.search(r"Action:\s*(\w+)", output, re.IGNORECASE)
            input_match = re.search(r"Action Input:\s*(\{.+\})", output, re.DOTALL)

            if action_match and input_match:
                tool_name = action_match.group(1).strip()
                tool_input_raw = input_match.group(1).strip()

                # Extract thought if present
                thought = self._extract_thought(output)
                if self.verbose:
                    if thought:
                        print(f"  💭 Thought: {thought}")
                    print(f"  🔧 Action: {tool_name}")
                    print(f"  📥 Input: {tool_input_raw}")

                try:
                    tool_input = json.loads(tool_input_raw)
                    if isinstance(tool_input, dict):
                        result = execute_tool(tool_name, **tool_input)
                    else:
                        result = execute_tool(tool_name, tool_input)
                except json.JSONDecodeError:
                        # Can't parse JSON — return error observation instead of crashing
                        observation = (
                            f"Observation: tool `{tool_name}` failed.\n"
                            f"Error: Invalid JSON in Action Input: {tool_input_raw}\n"
                            f"Please fix the format and try again. Use double quotes around keys and values."
                        )
                        messages.append({"role": "assistant", "content": output})
                        messages.append({"role": "user", "content": observation})
                        continue

                if self.verbose:
                    print(f"  📤 Observation: {result}")
                    print()

                observation = (
                    f"Observation: tool `{tool_name}` succeeded.\n"
                    f"Confirmed result: {result}\n"
                    f"Use this value as ground truth."
                )
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": observation})
            else:
                # No tool call, no Answer: → return as-is (fallback)
                if self.verbose:
                    print(f"  📝 (no tool call, returning directly)\n")
                return output.strip()

        if self.verbose:
            print("  ⚠️ Max steps reached\n")
        return "Max steps reached without final answer."

    def _extract_answer(self, output: str) -> str:
        return output.split("Answer:", 1)[1].strip()

    def _extract_thought(self, output: str) -> str:
        """Extract Thought section if present."""
        if "Thought:" in output:
            thought = output.split("Thought:", 1)[1]
            if "Action:" in thought:
                thought = thought.split("Action:", 1)[0]
            return thought.strip()
        return ""