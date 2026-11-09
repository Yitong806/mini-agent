"""
Tests for llm module: provider classes, HTTP calls, error handling, factory.

Uses monkeypatch to mock httpx.post — no real API calls, no .env needed.
"""

import json
import pytest
import httpx
from unittest.mock import Mock, patch

from mini_agent.llm import (
    LLMProvider,
    OpenAIProvider,
    DeepSeekProvider,
    ClaudeProvider,
    create_llm,
)


# ────────────────────────────────────────
# Tests: LLMProvider (base class)
# ────────────────────────────────────────

class TestLLMProviderBase:
    def test_base_class_instantiable(self):
        """LLMProvider is a plain class, not ABC."""
        provider = LLMProvider()
        assert provider is not None

    def test_chat_raises_not_implemented(self):
        provider = LLMProvider()
        with pytest.raises(NotImplementedError):
            provider.chat([{"role": "user", "content": "hi"}])


# ────────────────────────────────────────
# Tests: OpenAIProvider
# ────────────────────────────────────────

class TestOpenAIProvider:
    def test_init_default_model(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = OpenAIProvider()
            assert llm.model == "gpt-4o-mini"

    def test_init_custom_model(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = OpenAIProvider(model="gpt-4")
            assert llm.model == "gpt-4"

    def test_init_reads_env_var(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "my-secret-key"}):
            llm = OpenAIProvider()
            assert llm.api_key == "my-secret-key"

    def test_init_url(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = OpenAIProvider()
            assert "openai.com" in llm.url

    @patch("mini_agent.llm.httpx.post")
    def test_chat_success(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from OpenAI!"}}]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = OpenAIProvider()
            result = llm.chat([{"role": "user", "content": "hi"}])

        assert result == "Hello from OpenAI!"
        mock_post.assert_called_once()
        # Verify correct URL
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"

    @patch("mini_agent.llm.httpx.post")
    def test_chat_passes_messages(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = OpenAIProvider()
            messages = [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "hello"},
            ]
            llm.chat(messages)

        call_kwargs = mock_post.call_args.kwargs
        assert "json" in call_kwargs
        assert call_kwargs["json"]["messages"] == messages

    @patch("mini_agent.llm.httpx.post")
    def test_chat_passes_model(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = OpenAIProvider(model="gpt-4")
            llm.chat([{"role": "user", "content": "hi"}])

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["model"] == "gpt-4"

    @patch("mini_agent.llm.httpx.post")
    def test_chat_http_error_raises(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=Mock(),
            response=Mock(status_code=401),
        )
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "bad-key"}):
            llm = OpenAIProvider()
            with pytest.raises(httpx.HTTPStatusError):
                llm.chat([{"role": "user", "content": "hi"}])

    @patch("mini_agent.llm.httpx.post")
    def test_chat_network_error_raises(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("Connection failed")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = OpenAIProvider()
            with pytest.raises(httpx.ConnectError):
                llm.chat([{"role": "user", "content": "hi"}])


# ────────────────────────────────────────
# Tests: DeepSeekProvider
# ────────────────────────────────────────

class TestDeepSeekProvider:
    def test_init_default_model(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            llm = DeepSeekProvider()
            assert llm.model == "deepseek-chat"

    def test_init_custom_model(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            llm = DeepSeekProvider(model="deepseek-reasoner")
            assert llm.model == "deepseek-reasoner"

    def test_init_reads_env_var(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "ds-secret"}):
            llm = DeepSeekProvider()
            assert llm.api_key == "ds-secret"

    def test_init_url(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            llm = DeepSeekProvider()
            assert "deepseek.com" in llm.url

    @patch("mini_agent.llm.httpx.post")
    def test_chat_success(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from DeepSeek!"}}]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            llm = DeepSeekProvider()
            result = llm.chat([{"role": "user", "content": "hi"}])

        assert result == "Hello from DeepSeek!"
        mock_post.assert_called_once()

    @patch("mini_agent.llm.httpx.post")
    def test_chat_passes_auth_header(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "ds-key-123"}):
            llm = DeepSeekProvider()
            llm.chat([{"role": "user", "content": "hi"}])

        call_kwargs = mock_post.call_args.kwargs
        headers = call_kwargs["headers"]
        assert "Bearer ds-key-123" in headers["Authorization"]


# ────────────────────────────────────────
# Tests: ClaudeProvider
# ────────────────────────────────────────

class TestClaudeProvider:
    def test_init_default_model(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            llm = ClaudeProvider()
            assert "claude" in llm.model.lower()

    def test_init_custom_model(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            llm = ClaudeProvider(model="claude-3-opus-20240229")
            assert llm.model == "claude-3-opus-20240229"

    def test_init_reads_env_var(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "claude-secret"}):
            llm = ClaudeProvider()
            assert llm.api_key == "claude-secret"

    @patch("mini_agent.llm.httpx.post")
    def test_chat_success(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello from Claude!"}]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            llm = ClaudeProvider()
            result = llm.chat([{"role": "user", "content": "hi"}])

        assert result == "Hello from Claude!"
        mock_post.assert_called_once()

    @patch("mini_agent.llm.httpx.post")
    def test_chat_extracts_system_message(self, mock_post):
        """Claude API expects system as a top-level param, not in messages array."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "ok"}]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            llm = ClaudeProvider()
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hello"},
            ]
            llm.chat(messages)

        call_kwargs = mock_post.call_args.kwargs
        json_body = call_kwargs["json"]
        # System extracted to top-level
        assert json_body["system"] == "You are helpful"
        # System NOT in messages array
        for m in json_body["messages"]:
            assert m["role"] != "system"

    @patch("mini_agent.llm.httpx.post")
    def test_chat_no_system_message(self, mock_post):
        """When no system message, system param should be empty string."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "ok"}]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            llm = ClaudeProvider()
            messages = [{"role": "user", "content": "hello"}]
            llm.chat(messages)

        call_kwargs = mock_post.call_args.kwargs
        json_body = call_kwargs["json"]
        assert json_body["system"] == ""

    @patch("mini_agent.llm.httpx.post")
    def test_chat_passes_max_tokens(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "ok"}]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            llm = ClaudeProvider()
            llm.chat([{"role": "user", "content": "hi"}])

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["max_tokens"] == 1024

    @patch("mini_agent.llm.httpx.post")
    def test_chat_passes_anthropic_version_header(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "ok"}]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            llm = ClaudeProvider()
            llm.chat([{"role": "user", "content": "hi"}])

        call_kwargs = mock_post.call_args.kwargs
        headers = call_kwargs["headers"]
        assert headers["anthropic-version"] == "2023-06-01"


# ────────────────────────────────────────
# Tests: create_llm factory
# ────────────────────────────────────────

class TestCreateLLM:
    def test_create_openai(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = create_llm("openai")
            assert isinstance(llm, OpenAIProvider)

    def test_create_deepseek(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            llm = create_llm("deepseek")
            assert isinstance(llm, DeepSeekProvider)

    def test_create_claude(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            llm = create_llm("claude")
            assert isinstance(llm, ClaudeProvider)

    def test_create_case_insensitive(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = create_llm("OpenAI")
            assert isinstance(llm, OpenAIProvider)

    def test_create_with_kwargs(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = create_llm("openai", model="gpt-4")
            assert isinstance(llm, OpenAIProvider)
            assert llm.model == "gpt-4"

    def test_create_unknown_provider_raises_key_error(self):
        with pytest.raises(KeyError):
            create_llm("unknown_provider_xyz")