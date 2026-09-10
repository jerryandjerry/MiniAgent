"""Tests for the OpenAI-compatible client used during data generation."""

import inspect

import pytest

from data.clients import openai_compat


def test_call_llm_passes_flags_by_keyword_not_position():
    params = list(inspect.signature(openai_compat.llm_manager.call_model).parameters)
    assert params[:8] == [
        "model_name",
        "prompt",
        "system_prompt",
        "messages",
        "tools",
        "tool_call_handler",
        "stream",
        "show_thinking",
    ]
    source = inspect.getsource(openai_compat.call_llm)
    assert "stream=stream" in source
    assert "show_thinking=show_thinking" in source
    assert "messages, stream, show_thinking)" not in source


def test_every_provider_has_an_environment_key_mapping():
    for name, config in openai_compat.llm_manager.models.items():
        assert config.get("provider") in openai_compat.LLMManager._KEY_ENV, name


@pytest.mark.live
def test_call_llm_round_trips():
    output = openai_compat.call_llm(
        "qwen-plus",
        "Reply with only pong.",
        "You are a test assistant.",
        stream=False,
        show_thinking=False,
    )
    assert "pong" in output.lower()


@pytest.mark.live
def test_configured_qwen_models_receive_the_environment_key():
    qwen = [
        config
        for config in openai_compat.llm_manager.models.values()
        if config["provider"] == "qwen"
    ]
    assert qwen and all(config["api_key"] for config in qwen)
