"""
hello_agent.py — End-to-end ReAct demo

Single-turn usage: fresh Agent per query (or reuse, doesn't matter).
Demonstrates tool registration, multi-provider LLM, full ReAct loop.
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


# ────────────────────────────────────────
# Main
# ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="mini-agent ReAct demo")
    parser.add_argument("--provider", default="openai", choices=["openai", "deepseek", "claude"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    llm_kwargs = {}
    if args.model:
        llm_kwargs["model"] = args.model
    llm = create_llm(args.provider, **llm_kwargs)

    system_prompt = format_system_prompt(get_tool_descriptions())

    # ── Demo queries (each uses a fresh Agent — single-turn style) ──
    queries = [
        "请计算 123 + 456 的结果",
        "帮我查一下北京的天气",
        "计算 23 * 45",
    ]

    print(f"\n🤖 mini-agent (provider: {args.provider})\n")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        agent = Agent(llm=llm, system_prompt=system_prompt, verbose=True)
        answer = agent.run(query)
        print(f"\n✅ Answer: {answer}")
        print("=" * 60)

    # ── Interactive mode (reuses Agent — multi-turn style) ──
    print("\n💬 进入交互模式（输入 'quit' 退出）\n")
    agent = Agent(llm=llm, system_prompt=system_prompt, verbose=True)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break

        answer = agent.run(user_input)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()