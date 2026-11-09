# mini-agent

A minimal, zero-framework ReAct (Reasoning + Acting) AI agent in pure Python.

No LangChain. No LLM SDKs. Just `httpx` + `python-dotenv` + your code.

## Features

- **ReAct loop**: Thought → Action → Observation → Answer
- **Multi-turn conversations** with session state
- **Tool calling** via decorator-based registration
- **Multi-provider**: OpenAI / DeepSeek / Claude (raw HTTP, no SDK)
- **History truncation** to prevent context overflow
- **Verbose mode** for debugging the full reasoning chain

## Project Structure

```text
mini-agent/
├── src/mini_agent/
│   ├── __init__.py
│   ├── llm.py          # Multi-provider LLM layer (OpenAI/DeepSeek/Claude)
│   ├── core.py         # Agent class with ReAct loop
│   ├── prompt.py       # System prompt & tool description formatting
│   └── tools.py        # Tool registry & execution
├── examples/
│   ├── hello_agent.py  # Basic usage + interactive mode
│   ├── multi_turn.py   # Multi-turn conversation with memory tools
│   └── search_tool.py  # DuckDuckGo search + summarize
├── tests/
│   ├── test_llm.py     # LLM provider tests (mocked HTTP)
│   ├── test_core.py    # ReAct loop & session tests
│   ├── test_prompt.py  # Prompt formatting tests
│   └── test_tools.py   # Tool registration & execution tests
├── pyproject.toml
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Yitong806/mini-agent.git
cd mini-agent
pip install -e .
```

### 2. Set up API keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

> Only the provider(s) you use need keys. The others can be left empty.

### 3. Run an example

```bash
# Basic agent
python examples/hello_agent.py

# Multi-turn with memory
python examples/multi_turn.py

# Search + summarize
python examples/search_tool.py
```

## Usage

### Basic agent

```python
from mini_agent.llm import create_llm
from mini_agent.core import Agent
from mini_agent.tools import register_tool, execute_tool, get_tool_descriptions

# Define a tool
@register_tool("add", "Add two numbers", {"a": "number", "b": "number"})
def add(a, b):
    return a + b

# Create LLM & agent
llm = create_llm("deepseek")  # or "openai", "claude"
agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    tools=get_tool_descriptions(),
    verbose=True,
)

# Run
result = agent.run("What is 123 + 456?")
print(result)
```

### Multi-turn conversation

```python
agent.run("My name is Alice.")
agent.run("What's my name?")   # Agent remembers via tool calls
agent.reset()                  # Clear history, keep system prompt
```

### Custom tools

```python
@register_tool("weather", "Get weather for a city", {"city": "string"})
def weather(city):
    # Your implementation here
    return f"{city}: 22°C, sunny"

agent = Agent(llm=llm, tools=get_tool_descriptions())
agent.run("What's the weather in Beijing?")
```

## How It Works

The agent follows the **ReAct pattern**:

1. User input → appended to message history
2. LLM generates a response
3. If response contains `Action:` + `Action Input:` → execute tool → append `Observation:` → loop back to step 2
4. If response contains `Answer:` → return final answer
5. If response has no ReAct format → return as-is (fallback)
6. History auto-truncates to prevent context overflow

```
Thought: I need to add these numbers
Action: add
Action Input: {"a": 123, "b": 456}
Observation: 579
Thought: Now I have the answer
Answer: 123 + 456 = 579
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 8 | Max ReAct loop iterations per query |
| `verbose` | False | Print Thought/Action/Input/Observation/Answer |
| History limit | 14 msgs | Auto-truncates to system + last 12 |

## Development

### Install dev dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/ -v
```

### Run a single test file

```bash
pytest tests/test_core.py -v
```

## Requirements

- Python 3.9+
- `httpx>=0.27.0`
- `python-dotenv>=1.0.0`

## License

MIT
