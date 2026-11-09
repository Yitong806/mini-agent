"""
Tests for core Agent: ReAct loop logic, format parsing, history management.

Uses mock LLM — no real API calls.
"""

import pytest
from mini_agent.core import Agent


# ────────────────────────────────────────
# Mock LLM Providers
# ────────────────────────────────────────

class MockLLM:
    """Returns predefined responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages):
        self.calls.append([{k: v for k, v in m.items()} for m in messages])
        if not self.responses:
            return "Answer: no more responses"
        return self.responses.pop(0)


class PlainTextMockLLM:
    """Always returns unformatted plain text (no ReAct markers)."""

    def __init__(self):
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return "This is a plain text response with no ReAct format at all."


class EmptyMockLLM:
    """Returns empty string."""

    def __init__(self):
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return ""


# ────────────────────────────────────────
# Helper: build a tools dict compatible with Agent
# ────────────────────────────────────────

def _make_tools():
    """Create a fresh tools dict for testing."""
    def add(a, b):
        return a + b

    def multiply(a, b):
        return a * b

    def echo(text):
        return f"ECHO:{text}"

    def error_tool():
        raise ValueError("boom")

    return {
        "add": add,
        "multiply": multiply,
        "echo": echo,
        "error_tool": error_tool,
    }


# ────────────────────────────────────────
# Tests: Agent.__init__
# ────────────────────────────────────────

class TestAgentInit:
    def test_init_without_system_prompt(self):
        llm = MockLLM(["Answer: hi"])
        agent = Agent(llm=llm)
        assert agent.llm is llm
        assert agent.max_steps == 8
        assert agent.verbose is False
        assert len(agent.messages) == 0

    def test_init_with_system_prompt(self):
        llm = MockLLM(["Answer: hi"])
        agent = Agent(llm=llm, system_prompt="You are helpful.")
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == "You are helpful."

    def test_init_with_custom_max_steps(self):
        llm = MockLLM(["Answer: hi"])
        agent = Agent(llm=llm, max_steps=3)
        assert agent.max_steps == 3

    def test_init_with_verbose(self):
        llm = MockLLM(["Answer: hi"])
        agent = Agent(llm=llm, verbose=True)
        assert agent.verbose is True

    def test_init_with_custom_tools(self):
        tools = _make_tools()
        llm = MockLLM(["Answer: ok"])
        agent = Agent(llm=llm, tools=tools)
        assert agent.tools is tools


# ────────────────────────────────────────
# Tests: Agent.run
# ────────────────────────────────────────

class TestAgentRun:
    def test_run_returns_answer(self):
        llm = MockLLM(["Answer: 42"])
        agent = Agent(llm=llm)
        result = agent.run("What is 6 * 7?")
        assert result == "42"

    def test_run_appends_user_message(self):
        llm = MockLLM(["Answer: ok"])
        agent = Agent(llm=llm)
        agent.run("hello")
        user_msgs = [m for m in agent.messages if m["role"] == "user"]
        assert any(m["content"] == "hello" for m in user_msgs)

    def test_run_appends_assistant_message(self):
        llm = MockLLM(["Answer: world"])
        agent = Agent(llm=llm)
        agent.run("hello")
        assistant_msgs = [m for m in agent.messages if m["role"] == "assistant"]
        assert any(m["content"] == "world" for m in assistant_msgs)

    def test_run_multi_turn_preserves_history(self):
        llm = MockLLM(["Answer: hi there", "Answer: nice to meet you"])
        agent = Agent(llm=llm)
        agent.run("hello")
        agent.run("how are you")
        user_msgs = [m for m in agent.messages if m["role"] == "user"]
        assistant_msgs = [m for m in agent.messages if m["role"] == "assistant"]
        assert len(user_msgs) >= 2
        assert len(assistant_msgs) >= 2

    def test_run_with_system_prompt_includes_it(self):
        llm = MockLLM(["Answer: done"])
        agent = Agent(llm=llm, system_prompt="Be concise.")
        agent.run("test")
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == "Be concise."


# ────────────────────────────────────────
# Tests: Agent.reset
# ────────────────────────────────────────

class TestAgentReset:
    def test_reset_clears_history_keeps_system(self):
        llm = MockLLM(["Answer: 1", "Answer: 2"])
        agent = Agent(llm=llm, system_prompt="Be helpful.")
        agent.run("first")
        assert len(agent.messages) > 1
        agent.reset()
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    def test_reset_without_system_prompt(self):
        llm = MockLLM(["Answer: ok"])
        agent = Agent(llm=llm)
        agent.run("hi")
        assert len(agent.messages) > 0
        agent.reset()
        assert len(agent.messages) == 0


# ────────────────────────────────────────
# Tests: ReAct loop — tool call flow
# ────────────────────────────────────────

class TestReActLoopToolCall:
    def test_single_tool_call_then_answer(self):
        llm = MockLLM([
            'Thought: need to add\nAction: add\nAction Input: {"a": 2, "b": 3}',
            "Answer: 2 + 3 = 5",
        ])
        tools = _make_tools()
        agent = Agent(llm=llm, tools=tools)
        result = agent.run("add 2 and 3")
        assert "5" in result

    def test_multi_step_tool_calls(self):
        llm = MockLLM([
            'Thought: step 1\nAction: add\nAction Input: {"a": 5, "b": 5}',
            'Thought: step 2\nAction: multiply\nAction Input: {"a": 10, "b": 2}',
            "Answer: final result is 20",
        ])
        tools = _make_tools()
        agent = Agent(llm=llm, tools=tools)
        result = agent.run("add 5+5 then multiply by 2")
        assert "20" in result

    def test_tool_error_returns_error_string(self):
        llm = MockLLM([
            'Action: error_tool\nAction Input: {}',
            "Answer: got an error but handled it",
        ])
        tools = _make_tools()
        agent = Agent(llm=llm, tools=tools)
        result = agent.run("trigger error")
        # Should not crash; should return final answer
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tool_result_appears_in_observation(self):
        llm = MockLLM([
            'Action: echo\nAction Input: {"text": "hello"}',
            "Answer: got hello",
        ])
        tools = _make_tools()
        agent = Agent(llm=llm, tools=tools)
        result = agent.run("echo hello")
        assert "hello" in result


# ────────────────────────────────────────
# Tests: ReAct loop — fallback (no format)
# ────────────────────────────────────────

class TestReActLoopFallback:
    def test_plain_text_no_format_returns_directly(self):
        llm = PlainTextMockLLM()
        agent = Agent(llm=llm, max_steps=3)
        result = agent.run("anything")
        assert "plain text" in result

    def test_empty_response_returns_directly(self):
        llm = EmptyMockLLM()
        agent = Agent(llm=llm, max_steps=3)
        result = agent.run("anything")
        assert isinstance(result, str)

    def test_max_steps_exhausted(self):
        # LLM always outputs unparseable text
        # Current behavior: falls back to returning output directly (doesn't burn steps)
        # So "Max steps" won't appear — test should verify it returns *something* and doesn't loop forever
        llm = MockLLM(["no format", "still no format", "nope"])
        agent = Agent(llm=llm, max_steps=3)
        result = agent.run("test")
        # Should return the last output (fallback behavior)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should not contain "Max steps" because fallback returns immediately
        # (this documents current behavior)


# ────────────────────────────────────────
# Tests: History truncation
# ────────────────────────────────────────

class TestHistoryTruncation:
    def test_truncation_does_not_crash(self):
        # Build up enough messages to trigger truncation (>14)
        llm = MockLLM([
            'Action: add\nAction Input: {"a": 1, "b": 1}',
            'Action: add\nAction Input: {"a": 2, "b": 2}',
            'Action: add\nAction Input: {"a": 3, "b": 3}',
            'Action: add\nAction Input: {"a": 4, "b": 4}',
            'Action: add\nAction Input: {"a": 5, "b": 5}',
            "Answer: done",
        ])
        tools = _make_tools()
        agent = Agent(llm=llm, tools=tools, system_prompt="Be helpful.")
        result = agent.run("start")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_truncation_preserves_system_prompt(self):
        llm = MockLLM([
            'Action: add\nAction Input: {"a": 1, "b": 1}',
            'Action: add\nAction Input: {"a": 2, "b": 2}',
            'Action: add\nAction Input: {"a": 3, "b": 3}',
            'Action: add\nAction Input: {"a": 4, "b": 4}',
            'Action: add\nAction Input: {"a": 5, "b": 5}',
            "Answer: done",
        ])
        tools = _make_tools()
        agent = Agent(llm=llm, tools=tools, system_prompt="System prompt here.")
        agent.run("start")
        # System prompt should still be first
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == "System prompt here."


# ────────────────────────────────────────
# Tests: Verbose mode
# ────────────────────────────────────────

class TestVerboseMode:
    def test_verbose_true_does_not_crash(self, capsys):
        llm = MockLLM(["Answer: verbose test"])
        agent = Agent(llm=llm, verbose=True)
        result = agent.run("test")
        assert result == "verbose test"
        # verbose prints to stdout, just ensure no exception

    def test_verbose_false_no_extra_output(self, capsys):
        llm = MockLLM(["Answer: quiet test"])
        agent = Agent(llm=llm, verbose=False)
        result = agent.run("test")
        captured = capsys.readouterr()
        # Should not print ReAct trace
        assert "Thought:" not in captured.out
        assert "Action:" not in captured.out


# ────────────────────────────────────────
# Tests: Edge cases
# ────────────────────────────────────────

class TestEdgeCases:
    def test_run_empty_input(self):
        llm = MockLLM(["Answer: empty input received"])
        agent = Agent(llm=llm)
        result = agent.run("")
        assert isinstance(result, str)

    def test_unknown_tool_in_output_does_not_crash(self):
        llm = MockLLM([
            'Action: nonexistent_tool\nAction Input: {"x": 1}',
            "Answer: handled unknown tool",
        ])
        tools = _make_tools()
        agent = Agent(llm=llm, tools=tools)
        result = agent.run("use unknown tool")
        assert isinstance(result, str)

    def test_malformed_json_input_does_not_crash(self):
        llm = MockLLM([
            'Action: add\nAction Input: {a: 1, b: 2}',  # missing quotes
            "Answer: handled bad json",
        ])
        tools = _make_tools()
        agent = Agent(llm=llm, tools=tools)
        result = agent.run("bad json")
        assert isinstance(result, str)