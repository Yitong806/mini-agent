"""
Thin Multi-Provider LLM Layer (~45 lines)

All providers called via raw HTTP + httpx.
Zero LLM SDK dependencies.
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()


class LLMProvider:
    def chat(self, messages):
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.url = "https://api.openai.com/v1/chat/completions"

    def chat(self, messages):
        resp = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class DeepSeekProvider(LLMProvider):
    def __init__(self, model="deepseek-chat"):
        self.model = model
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.url = "https://api.deepseek.com/v1/chat/completions"

    def chat(self, messages):
        resp = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class ClaudeProvider(LLMProvider):
    def __init__(self, model="claude-3-5-sonnet-20241022"):
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.url = "https://api.anthropic.com/v1/messages"

    def chat(self, messages):
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        resp = httpx.post(
            self.url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "max_tokens": 1024, "system": system, "messages": filtered},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


def create_llm(provider: str, **kwargs) -> LLMProvider:
    return {"openai": OpenAIProvider, "deepseek": DeepSeekProvider, "claude": ClaudeProvider}[provider.lower()](**kwargs)