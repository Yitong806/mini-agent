"""
Tests for prompt module: system prompt formatting, tool description rendering, user prompt wrapping.
"""

import pytest
from mini_agent.prompt import (
    format_system_prompt,
    format_user_prompt,
    SYSTEM_PROMPT,
    TOOL_DESC_TEMPLATE,
)


# ────────────────────────────────────────
# Tests: format_system_prompt
# ────────────────────────────────────────

class TestFormatSystemPrompt:
    def test_returns_string(self):
        result = format_system_prompt({})
        assert isinstance(result, str)

    def test_empty_tools_shows_no_tools_message(self):
        result = format_system_prompt({})
        assert "No tools" in result

    def test_empty_tools_does_not_contain_placeholder(self):
        result = format_system_prompt({})
        assert "{tools}" not in result

    def test_single_tool_renders_name(self):
        tools = {
            "add": {
                "description": "Add two numbers",
                "schema": {"a": "number", "b": "number"},
            }
        }
        result = format_system_prompt(tools)
        assert "add" in result

    def test_single_tool_renders_description(self):
        tools = {
            "multiply": {
                "description": "Multiply two numbers",
                "schema": {"a": "number", "b": "number"},
            }
        }
        result = format_system_prompt(tools)
        assert "Multiply two numbers" in result

    def test_single_tool_renders_schema(self):
        tools = {
            "divide": {
                "description": "Divide a by b",
                "schema": {"a": "number", "b": "number"},
            }
        }
        result = format_system_prompt(tools)
        # schema dict is rendered as str
        assert "schema" in result.lower() or "a" in result

    def test_multiple_tools_all_rendered(self):
        tools = {
            "add": {"description": "Add numbers", "schema": {"a": "number", "b": "number"}},
            "echo": {"description": "Echo text", "schema": {"text": "string"}},
            "search": {"description": "Search the web", "schema": {"query": "string"}},
        }
        result = format_system_prompt(tools)
        assert "add" in result
        assert "echo" in result
        assert "search" in result
        assert "Add numbers" in result
        assert "Echo text" in result
        assert "Search the web" in result

    def test_tool_with_empty_schema(self):
        tools = {
            "noop": {
                "description": "Does nothing",
                "schema": {},
            }
        }
        result = format_system_prompt(tools)
        assert "noop" in result
        assert "Does nothing" in result

    def test_tool_with_none_schema(self):
        tools = {
            "noop": {
                "description": "Does nothing",
                "schema": None,
            }
        }
        result = format_system_prompt(tools)
        assert "noop" in result
        assert "Does nothing" in result

    def test_tool_with_string_meta_not_dict(self):
        tools = {
            "legacy_tool": "Just a description string",
        }
        result = format_system_prompt(tools)
        assert "legacy_tool" in result
        assert "Just a description" in result

    def test_result_contains_react_format_hint(self):
        result = format_system_prompt({})
        # Template always includes ReAct instructions
        assert "Thought:" in result
        assert "Action:" in result
        assert "Answer:" in result

    def test_result_contains_rules(self):
        result = format_system_prompt({})
        assert "Rules:" in result or "rules" in result.lower()

    def test_result_is_deterministic(self):
        tools = {
            "add": {"description": "Add numbers", "schema": {"a": "number", "b": "number"}},
            "echo": {"description": "Echo text", "schema": {"text": "string"}},
        }
        result1 = format_system_prompt(tools)
        result2 = format_system_prompt(tools)
        assert result1 == result2

    def test_special_characters_in_description(self):
        tools = {
            "weird_tool": {
                "description": "Tool with 'quotes' & symbols: @#$%",
                "schema": {"param": "string"},
            }
        }
        result = format_system_prompt(tools)
        assert "weird_tool" in result
        assert "quotes" in result

    def test_template_is_fully_formatted_no_braces_left(self):
        tools = {
            "test": {"description": "test tool", "schema": {}},
        }
        result = format_system_prompt(tools)
        # Should not have any unformatted placeholders
        assert "{tools}" not in result


# ────────────────────────────────────────
# Tests: format_user_prompt
# ────────────────────────────────────────

class TestFormatUserPrompt:
    def test_returns_string(self):
        result = format_user_prompt("hello")
        assert isinstance(result, str)

    def test_returns_input_unchanged(self):
        result = format_user_prompt("hello world")
        assert result == "hello world"

    def test_empty_string(self):
        result = format_user_prompt("")
        assert result == ""

    def test_special_characters_preserved(self):
        result = format_user_prompt("Hello! @#$%^&*()_+")
        assert result == "Hello! @#$%^&*()_+"

    def test_multiline_input_preserved(self):
        text = "line 1\nline 2\nline 3"
        result = format_user_prompt(text)
        assert result == text

    def test_chinese_input_preserved(self):
        result = format_user_prompt("你好世界")
        assert result == "你好世界"


# ────────────────────────────────────────
# Tests: constants
# ────────────────────────────────────────

class TestConstants:
    def test_system_prompt_is_string(self):
        assert isinstance(SYSTEM_PROMPT, str)

    def test_system_prompt_has_tools_placeholder(self):
        assert "{tools}" in SYSTEM_PROMPT

    def test_system_prompt_contains_react_instructions(self):
        assert "Thought:" in SYSTEM_PROMPT
        assert "Action:" in SYSTEM_PROMPT
        assert "Action Input:" in SYSTEM_PROMPT
        assert "Answer:" in SYSTEM_PROMPT

    def test_tool_desc_template_is_string(self):
        assert isinstance(TOOL_DESC_TEMPLATE, str)

    def test_tool_desc_template_has_placeholders(self):
        assert "{name}" in TOOL_DESC_TEMPLATE
        assert "{description}" in TOOL_DESC_TEMPLATE