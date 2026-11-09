"""
Tests for tools module: registration, execution, description generation.
"""

import pytest
from mini_agent.tools import TOOLS, register_tool, execute_tool, get_tool_descriptions


class TestRegisterTool:
    def setup_method(self):
        # Clear TOOLS before each test to avoid cross-test pollution
        TOOLS.clear()

    def test_decorator_registers_function(self):
        @register_tool("test_add", "Add two numbers", {"a": "number", "b": "number"})
        def test_add(a, b):
            return a + b

        assert "test_add" in TOOLS
        assert callable(TOOLS["test_add"]["fn"])
        assert TOOLS["test_add"]["description"] == "Add two numbers"
        assert TOOLS["test_add"]["schema"] == {"a": "number", "b": "number"}

    def test_decorator_preserves_function(self):
        @register_tool("test_echo", "Echo input", {"text": "string"})
        def test_echo(text):
            return text

        assert test_echo("hi") == "hi"

    def test_default_schema_is_empty_dict(self):
        @register_tool("test_no_schema")
        def test_no_schema():
            return "ok"

        assert TOOLS["test_no_schema"]["schema"] == {}

    def test_default_description_is_empty_string(self):
        @register_tool("test_no_desc")
        def test_no_desc():
            return "ok"

        assert TOOLS["test_no_desc"]["description"] == ""


class TestExecuteTool:
    def setup_method(self):
        TOOLS.clear()

    def test_execute_with_kwargs(self):
        @register_tool("exec_add", "Add numbers", {"a": "number", "b": "number"})
        def exec_add(a, b):
            return a + b

        result = execute_tool("exec_add", a=3, b=4)
        assert result == 7

    def test_execute_returns_error_string_for_unknown_tool(self):
        result = execute_tool("nonexistent_tool_xyz")
        assert "Error" in result
        assert "nonexistent_tool_xyz" in result

    def test_execute_propagates_function_return(self):
        @register_tool("exec_greet", "Greet someone", {"person": "string"})
        def exec_greet(person):
            return f"Hello, {person}!"

        result = execute_tool("exec_greet", person="World")
        assert result == "Hello, World!"

    def test_execute_catches_exception(self):
        @register_tool("exec_broken", "Always raises", {})
        def exec_broken():
            raise ValueError("something broke")

        result = execute_tool("exec_broken")
        assert "Error" in result
        assert "something broke" in result


class TestGetToolDescriptions:
    def setup_method(self):
        TOOLS.clear()

    def test_returns_dict(self):
        result = get_tool_descriptions()
        assert isinstance(result, dict)

    def test_dict_structure_per_tool(self):
        @register_tool("desc_test", "A tool for testing", {"x": "number"})
        def desc_test(x):
            return x

        result = get_tool_descriptions()
        assert "desc_test" in result
        assert "description" in result["desc_test"]
        assert "schema" in result["desc_test"]
        assert result["desc_test"]["description"] == "A tool for testing"
        assert result["desc_test"]["schema"] == {"x": "number"}

    def test_empty_when_no_tools(self):
        result = get_tool_descriptions()
        assert result == {}

    def test_multiple_tools(self):
        @register_tool("t1", "First tool", {"a": "string"})
        def t1(a):
            return a

        @register_tool("t2", "Second tool", {"b": "number"})
        def t2(b):
            return b

        result = get_tool_descriptions()
        assert len(result) == 2
        assert "t1" in result
        assert "t2" in result