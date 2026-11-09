"""
multi_turn.py — Multi-turn conversation demo

Agent holds state internally; just call run() repeatedly.
Demonstrates cross-turn context, follow-up questions, session reset.
"""

import argparse
from mini_agent.llm import create_llm
from mini_agent.core import Agent
from mini_agent.prompt import format_system_prompt
from mini_agent.tools import register_tool, get_tool_descriptions


# ────────────────────────────────────────
# Tool Definitions
# ────────────────────────────────────────

@register_tool("add", "Add two numbers", {"a": "number", "b": "number"})
def add(a, b):
    return a + b


@register_tool("multiply", "Multiply two numbers", {"a": "number", "b": "number"})
def multiply(a, b):
    return a * b


@register_tool("weather", "Get temperature of a city", {"city": "string"})
def weather(city):
    data = {
        "北京": 15, "beijing": 15, "Beijing": 15,
        "深圳": 26, "shenzhen": 26, "Shenzhen": 26,
        "上海": 20, "shanghai": 20, "Shanghai": 20,
    }
    if city in data:
        return f"temperature of {city} = {data[city]} (Celsius)"
    return f"ERROR: no weather data for {city}"


@register_tool("remember", "Store a key-value fact for later use", {"key": "string", "value": "string"})
def remember(key, value):
    _memory[key] = value
    return f"Stored: {key} = {value}"


@register_tool("recall", "Recall a previously stored fact by key", {"key": "string"})
def recall(key):
    if key in _memory:
        return f"{key} = {_memory[key]}"
    return f"ERROR: nothing stored under '{key}'"


_memory = {}


# ────────────────────────────────────────
# Main
# ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="mini-agent multi-turn demo")
    parser.add_argument("--provider", default="openai", choices=["openai", "deepseek", "claude"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    llm_kwargs = {}
    if args.model:
        llm_kwargs["model"] = args.model
    llm = create_llm(args.provider, **llm_kwargs)

    system_prompt = format_system_prompt(get_tool_descriptions())

    # ── Pre-scripted multi-turn demo ──
    agent = Agent(llm=llm, system_prompt=system_prompt, verbose=True)

    turns = [
        "你好！我叫小明。请用 remember 工具记住我的名字。",
        "北京今天多少度？",
        "深圳呢？",
        "帮我算一下这两个城市温度的和。",
        "我叫什么名字？用 recall 工具告诉我。",
    ]

    print(f"\n🤖 mini-agent multi-turn (provider: {args.provider})\n")
    print("=" * 60)

    for i, turn in enumerate(turns, 1):
        print(f"\n👤 Turn {i}: {turn}")
        print("-" * 40)
        answer = agent.run(turn)
        print(f"\n🤖 Agent: {answer}")
        print("=" * 60)

    # ── Interactive multi-turn ──
    print("\n💬 进入多轮交互模式（'quit' 退出，'reset' 重置）\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("🔄 对话已重置\n")
            continue

        answer = agent.run(user_input)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()