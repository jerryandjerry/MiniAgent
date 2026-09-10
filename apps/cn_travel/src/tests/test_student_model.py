"""Smoke tests for the configured CN Travel policy endpoint."""
from __future__ import annotations

import json

import pytest

from cn_travel.agent import TravelAssistantFuncCall
from cn_travel.deployment import policy_endpoint_status
from cn_travel.service.agent import make_client

pytestmark = [pytest.mark.policy, pytest.mark.slow]


@pytest.fixture(scope="module")
def policy():
    client, model = make_client()
    assistant = TravelAssistantFuncCall(client=client, model_name=model, verbose=False)
    return client, model, assistant.system_prompt, assistant.tools


def test_configured_policy_is_served():
    status = policy_endpoint_status(timeout=3)
    assert status["ready"], status


def test_policy_returns_openai_chat_completion(policy):
    client, model, system_prompt, tools = policy
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "你好，请简单介绍你能提供的旅行帮助。"},
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    message = response.choices[0].message
    assert message.content or message.tool_calls


def test_policy_tool_calls_follow_registered_contract(policy):
    client, model, system_prompt, tools = policy
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "从北京望京到天坛公园怎么走？"},
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    calls = response.choices[0].message.tool_calls or []
    assert calls
    known = {tool["function"]["name"] for tool in tools}
    for call in calls:
        assert call.function.name in known
        assert isinstance(json.loads(call.function.arguments), dict)
