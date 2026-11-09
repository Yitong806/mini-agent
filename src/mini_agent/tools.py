"""
Tool Calling (~35 lines)

Register tools with name, description, schema.
Execute by name with keyword arguments.
"""

TOOLS = {}


def register_tool(name, description="", schema=None):
    """Decorator to register a tool."""

    def decorator(fn):
        TOOLS[name] = {
            "fn": fn,
            "description": description,
            "schema": schema or {},
        }
        return fn

    return decorator


def execute_tool(name, **kwargs):
    """Execute a tool by name with arguments."""
    if name not in TOOLS:
        return f"Error: tool '{name}' not found."
    try:
        return TOOLS[name]["fn"](**kwargs)
    except Exception as e:
        return f"Error executing tool '{name}': {e}"


def get_tool_descriptions():
    """Return {name: {description, schema}} for prompt formatting."""
    return {
        name: {
            "description": meta["description"],
            "schema": meta["schema"],
        }
        for name, meta in TOOLS.items()
    }