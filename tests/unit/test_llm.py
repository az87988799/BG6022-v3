from __future__ import annotations

import json
from collections.abc import Sequence
from threading import Event
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from bg6022.config import LlmSettings
from bg6022.llm import LlmClient, LlmError


class _Answer(BaseModel):
    value: int


def _response(content: Any, *, finish_reason: str, usage: dict[str, int]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "offline-response-v1",
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        },
    )


def _offline_client(
    responses: Sequence[httpx.Response], *, structured_output_corrections: int
) -> tuple[LlmClient, list[dict[str, Any]], httpx.Client]:
    remaining_responses = list(responses)
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return remaining_responses.pop(0)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = LlmSettings(
        api_key_env="TEST_BG6022_KEY",
        request_timeout_seconds=5,
        structured_output_corrections=structured_output_corrections,
    )
    return (
        LlmClient(settings, client=http_client, api_key="offline-test-key"),
        requests,
        http_client,
    )


def test_empty_stopped_json_response_gets_one_valid_correction_and_keeps_call_metadata() -> None:
    llm, requests, http_client = _offline_client(
        [
            _response(
                "",
                finish_reason="stop",
                usage={"prompt_tokens": 11, "completion_tokens": 0},
            ),
            _response(
                '{"value": 42}',
                finish_reason="stop",
                usage={"prompt_tokens": 17, "completion_tokens": 5},
            ),
        ],
        structured_output_corrections=1,
    )
    try:
        answer = llm.complete_json(
            [{"role": "user", "content": "give me a value"}],
            _Answer,
            purpose="offline-answer",
        )

        assert answer.value == 42
        assert len(requests) == 2
        assert "previous JSON was rejected" in requests[1]["messages"][-1]["content"]
        assert len(llm.calls) == 2
        assert [call.usage for call in llm.calls] == [
            {"prompt_tokens": 11, "completion_tokens": 0},
            {"prompt_tokens": 17, "completion_tokens": 5},
        ]
        assert [call.corrected for call in llm.calls] == [False, True]
        assert [call.category for call in llm.calls] == ["empty_response", "success"]
        assert [call.structured_correction_count for call in llm.calls] == [0, 1]
        assert all(call.purpose == "offline-answer" for call in llm.calls)
        assert all(call.schema_version for call in llm.calls)
        assert all(call.model == llm.settings.model for call in llm.calls)
        assert all(call.request_model == llm.settings.model for call in llm.calls)
        assert all(call.response_model == "offline-response-v1" for call in llm.calls)
        assert [call.body_length for call in llm.calls] == [0, len('{"value": 42}')]
        assert all(call.finish_reason == "stop" for call in llm.calls)
        assert all(call.transport_attempts == 1 for call in llm.calls)
        assert all(call.elapsed_seconds >= 0 for call in llm.calls)
    finally:
        http_client.close()


def test_empty_length_json_response_is_classified_as_truncated() -> None:
    llm, requests, http_client = _offline_client(
        [
            _response(
                "",
                finish_reason="length",
                usage={"prompt_tokens": 9, "completion_tokens": 0},
            )
        ],
        structured_output_corrections=0,
    )
    try:
        with pytest.raises(LlmError) as raised:
            llm.complete_json([{"role": "user", "content": "give me a value"}], _Answer)

        assert raised.value.category == "truncated"
        assert len(requests) == 1
        assert len(llm.calls) == 1
        assert llm.calls[0].usage == {"prompt_tokens": 9, "completion_tokens": 0}
        assert llm.calls[0].category == "truncated"
        assert llm.calls[0].finish_reason == "length"
    finally:
        http_client.close()


def test_nonempty_length_json_response_uses_the_same_correction_budget() -> None:
    llm, requests, http_client = _offline_client(
        [
            _response(
                '{"value":',
                finish_reason="length",
                usage={"prompt_tokens": 9, "completion_tokens": 2},
            ),
            _response(
                '{"value": 42}',
                finish_reason="stop",
                usage={"prompt_tokens": 17, "completion_tokens": 5},
            ),
        ],
        structured_output_corrections=1,
    )
    try:
        answer = llm.complete_json(
            [{"role": "user", "content": "give me a value"}],
            _Answer,
        )

        assert answer.value == 42
        assert len(requests) == 2
        assert [call.category for call in llm.calls] == ["truncated", "success"]
        assert [call.structured_correction_count for call in llm.calls] == [0, 1]
    finally:
        http_client.close()


def test_always_empty_stopped_json_response_exhausts_one_correction() -> None:
    empty = _response(
        "",
        finish_reason="stop",
        usage={"prompt_tokens": 8, "completion_tokens": 0},
    )
    llm, requests, http_client = _offline_client([empty, empty], structured_output_corrections=1)
    try:
        with pytest.raises(LlmError) as raised:
            llm.complete_json([{"role": "user", "content": "give me a value"}], _Answer)

        assert raised.value.category == "empty_response"
        assert len(requests) == 2
        assert len(llm.calls) == 2
        assert all(call.usage == {"prompt_tokens": 8, "completion_tokens": 0} for call in llm.calls)
        assert all(call.category == "empty_response" for call in llm.calls)
        assert [call.structured_correction_count for call in llm.calls] == [0, 1]
    finally:
        http_client.close()


@pytest.mark.parametrize(
    ("rejected_content", "category"),
    [("{bad", "invalid_json"), ('{"value":"wrong type"}', "schema_error")],
)
def test_invalid_json_and_schema_errors_share_the_same_correction_limit(
    rejected_content: str, category: str
) -> None:
    llm, requests, http_client = _offline_client(
        [
            _response(
                rejected_content,
                finish_reason="stop",
                usage={"prompt_tokens": 6, "completion_tokens": 2},
            ),
            _response(
                '{"value": 42}',
                finish_reason="stop",
                usage={"prompt_tokens": 9, "completion_tokens": 3},
            ),
        ],
        structured_output_corrections=1,
    )
    try:
        answer = llm.complete_json([{"role": "user", "content": "give me a value"}], _Answer)

        assert answer.value == 42
        assert len(requests) == 2
        assert [call.category for call in llm.calls] == [category, "success"]
        assert [call.structured_correction_count for call in llm.calls] == [0, 1]
    finally:
        http_client.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "offline-response-v1", "choices": [], "usage": {"completion_tokens": 0}},
        {
            "model": "offline-response-v1",
            "choices": [{"message": None, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 0},
        },
        {
            "model": "offline-response-v1",
            "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 0},
        },
    ],
)
def test_invalid_message_structure_is_not_classified_as_empty_content(
    payload: dict[str, Any],
) -> None:
    response = httpx.Response(200, json=payload)
    llm, requests, http_client = _offline_client([response], structured_output_corrections=1)
    try:
        with pytest.raises(LlmError) as raised:
            llm.complete_json(
                [{"role": "user", "content": "give me a value"}],
                _Answer,
                purpose="planner",
            )

        assert raised.value.category == "invalid_response"
        assert raised.value.purpose == "planner"
        assert len(requests) == 1
        assert len(llm.calls) == 1
        assert llm.calls[0].category == "invalid_response"
        assert llm.calls[0].body_length == len(response.content)
        assert llm.calls[0].usage == payload["usage"]
    finally:
        http_client.close()


def test_length_finish_reason_wins_over_missing_json_message() -> None:
    response = httpx.Response(
        200,
        json={
            "model": "offline-response-v1",
            "choices": [{"finish_reason": "length"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 0},
        },
    )
    llm, requests, http_client = _offline_client([response], structured_output_corrections=0)
    try:
        with pytest.raises(LlmError) as raised:
            llm.complete_json([{"role": "user", "content": "give me a value"}], _Answer)

        assert raised.value.category == "truncated"
        assert len(requests) == 1
        assert llm.calls[0].category == "truncated"
        assert llm.calls[0].finish_reason == "length"
    finally:
        http_client.close()


def test_complete_text_still_rejects_an_empty_answer() -> None:
    llm, requests, http_client = _offline_client(
        [
            _response(
                "",
                finish_reason="stop",
                usage={"prompt_tokens": 4, "completion_tokens": 0},
            )
        ],
        structured_output_corrections=1,
    )
    try:
        with pytest.raises(LlmError) as raised:
            llm.complete_text([{"role": "user", "content": "answer in text"}])

        assert raised.value.category == "empty_response"
        assert len(requests) == 1
        assert len(llm.calls) == 1
        assert llm.calls[0].category == "empty_response"
    finally:
        http_client.close()


@pytest.mark.parametrize("status_code", [401, 403])
def test_auth_failure_is_recorded_once_without_exposing_response_body(status_code: int) -> None:
    llm, requests, http_client = _offline_client(
        [httpx.Response(status_code, text="secret provider details")],
        structured_output_corrections=1,
    )
    try:
        with pytest.raises(LlmError) as raised:
            llm.complete_json(
                [{"role": "user", "content": "give me a value"}],
                _Answer,
                purpose="intake",
            )

        assert raised.value.category == "auth"
        assert raised.value.purpose == "intake"
        assert len(requests) == 1
        assert len(llm.calls) == 1
        call = llm.calls[0]
        assert call.purpose == "intake"
        assert call.category == "auth"
        assert call.request_model == llm.settings.model
        assert call.response_model is None
        assert call.body_length == len("secret provider details")
        assert call.structured_correction_count == 0
        assert "secret provider details" not in repr(call)
    finally:
        http_client.close()


@pytest.mark.parametrize("status_code", [429, 503])
def test_rate_limit_and_server_error_use_only_one_transport_retry(status_code: int) -> None:
    first_response = (
        httpx.Response(429, headers={"Retry-After": "0"})
        if status_code == 429
        else httpx.Response(503, text="temporary provider failure")
    )
    llm, requests, http_client = _offline_client(
        [
            first_response,
            _response(
                '{"value": 42}',
                finish_reason="stop",
                usage={"prompt_tokens": 9, "completion_tokens": 3},
            ),
        ],
        structured_output_corrections=1,
    )
    try:
        answer = llm.complete_json([{"role": "user", "content": "give me a value"}], _Answer)

        assert answer.value == 42
        assert len(requests) == 2
        assert len(llm.calls) == 1
        assert llm.calls[0].transport_attempts == 2
        assert llm.calls[0].category == "success"
    finally:
        http_client.close()


def test_timeout_uses_only_one_transport_retry() -> None:
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        raise httpx.ReadTimeout("offline timeout")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        LlmSettings(api_key_env="TEST_BG6022_KEY", request_timeout_seconds=5),
        client=http_client,
        api_key="offline-test-key",
    )
    try:
        with pytest.raises(LlmError) as raised:
            llm.complete_json([{"role": "user", "content": "give me a value"}], _Answer)

        assert raised.value.category == "timeout"
        assert request_count == 2
        assert len(llm.calls) == 1
        assert llm.calls[0].transport_attempts == 2
        assert llm.calls[0].category == "timeout"
    finally:
        http_client.close()


def test_cancellation_after_transport_does_not_apply_a_late_response() -> None:
    cancel = Event()

    def handler(_request: httpx.Request) -> httpx.Response:
        cancel.set()
        return _response(
            '{"value": 42}',
            finish_reason="stop",
            usage={"prompt_tokens": 9, "completion_tokens": 3},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        LlmSettings(api_key_env="TEST_BG6022_KEY", request_timeout_seconds=5),
        client=http_client,
        api_key="offline-test-key",
    )
    try:
        with pytest.raises(LlmError) as raised:
            llm.complete_json(
                [{"role": "user", "content": "give me a value"}],
                _Answer,
                cancel=cancel,
            )

        assert raised.value.category == "cancelled"
        assert len(llm.calls) == 1
        assert llm.calls[0].category == "cancelled"
    finally:
        http_client.close()
