"""
search_tool.py — Real tool integration demo

Demonstrates:
  - Registering a real HTTP-based tool (DuckDuckGo Instant Answer API)
  - Agent autonomously decides when to search
  - Multi-step ReAct: search → parse → answer
  - No API key required (DuckDuckGo Instant Answer API is free)

Usage:
  python examples/search_tool.py
  python examples/search_tool.py --provider deepseek
"""

import argparse
import urllib.parse

import httpx
from mini_agent.llm import create_llm
from mini_agent.core import Agent
from mini_agent.prompt import format_system_prompt
from mini_agent.tools import register_tool, get_tool_descriptions


# ────────────────────────────────────────
# Search Tool (real HTTP call)
# ────────────────────────────────────────

@register_tool(
    name="search",
    description="Search the web for a query and return a brief answer or summary",
    schema={"query": "string"}
)
def search(query):
    """Call DuckDuckGo Instant Answer API (no key needed)."""
    print('search is invoked')
    encoded = urllib.parse.quote(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"

    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"ERROR: search failed: {e}"

    # DuckDuckGo Instant Answer API returns several fields
    # Priority: AbstractText > Answer > RelatedTopics

    abstract = data.get("AbstractText", "").strip()
    if abstract:
        source = data.get("AbstractSource", "")
        return f"{abstract}" + (f" (source: {source})" if source else "")

    answer = data.get("Answer", "").strip()
    if answer:
        return f"{answer}"

    # RelatedTopics fallback
    topics = data.get("RelatedTopics", [])
    if topics:
        snippets = []
        for t in topics[:3]:
            if isinstance(t, dict) and t.get("Text"):
                snippets.append(t["Text"])
        if snippets:
            return " | ".join(snippets)

    # Definition fallback
    definition = data.get("Definition", "").strip()
    if definition:
        return definition

    return f"No direct answer found for '{query}'. Try a more specific query."


# ────────────────────────────────────────
# Additional tools for composition
# ────────────────────────────────────────

@register_tool("summarize", "Summarize a piece of text into a shorter form", {"text": "string"})
def summarize(text):
    # Simple extractive "summarization" — take first 2 sentences
    sentences = text.replace("\n", " ").split(".")
    if len(sentences) <= 2:
        return text
    return ". ".join(s.strip() for s in sentences[:2] if s.strip()) + "."


# ────────────────────────────────────────
# Main
# ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="mini-agent search tool demo")
    parser.add_argument("--provider", default="openai", choices=["openai", "deepseek", "claude"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    llm_kwargs = {}
    if args.model:
        llm_kwargs["model"] = args.model
    llm = create_llm(args.provider, **llm_kwargs)

    system_prompt = format_system_prompt(get_tool_descriptions())

    agent = Agent(llm=llm, system_prompt=system_prompt)

    # ── Pre-scripted demo queries ──
    queries = [
        "搜索一下 Python 编程语言是什么时候发布的？",
        "Who won the 2024 Nobel Prize in Physics? Search for it.",
        "搜索 'ReAct agent' 是什么，然后用 summarize 工具总结成一句话。",
    ]

    print(f"\n🤖 mini-agent search demo (provider: {args.provider})\n")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        answer = agent.run(query)
        print(f"\n✅ Answer: {answer}")
        print("=" * 60)

    # ── Interactive search mode ──
    print("\n💬 进入搜索交互模式（输入 'quit' 退出，'reset' 重置）\n")

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