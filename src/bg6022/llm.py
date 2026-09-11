"""One bounded DeepSeek HTTP client used by intake, planning, repair, and answers."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from threading import Event
from typing import Any

import httpx
from pydantic import BaseModel

from bg6022.config import AppConfig, LlmSettings


class LlmError(RuntimeError):
    """A user-actionable model failure without retaining credentials."""

    def __init__(self, message: str, *, category: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable


@dataclass(frozen=True)
class LlmCall:
    purpose: str
    model: str
    schema_version: str | None
    elapsed_seconds: float
    usage: dict[str, Any]
    corrected: bool = False


class LlmClient:
    """Direct Chat Completions transport with bounded retries and correction."""

    def __init__(
        self,
        config: AppConfig | LlmSettings,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        api_key: str | None = None,
    ) -> None:
        self.settings = config.llm if isinstance(config, AppConfig) else config
        self._client = client
        self._transport = transport
        self._api_key_override = api_key
        self.calls: list[LlmCall] = []

    def complete_text(
        self,
        messages: Iterable[Mapping[str, str]],
        *,
        purpose: str = "text",
        cancel: Event | None = None,
        remaining_timeout_seconds: float | None = None,
    ) -> str:
        payload = self._complete(
            list(messages),
            purpose=purpose,
            response_format=None,
            schema_version=None,
            cancel=cancel,
            remaining_timeout_seconds=remaining_timeout_seconds,
        )
        content, _usage, _finish = payload
        return content

    def complete_json(
        self,
        messages: Iterable[Mapping[str, str]],
        schema: type[BaseModel] | Mapping[str, Any] | Callable[[Any], Any],
        *,
        purpose: str = "json",
        example: Mapping[str, Any] | None = None,
        cancel: Event | None = None,
        remaining_timeout_seconds: float | None = None,
    ) -> Any:
        schema_json = _schema_json(schema)
        schema_version = _schema_version(schema_json)
        base_messages = list(messages)
        instructions = {
            "role": "system",
            "content": (
                "Return exactly one JSON object and no markdown. It must match this schema:\n"
                f"{json.dumps(schema_json, ensure_ascii=False, sort_keys=True)}"
                + (
                    "\nA valid example is:\n"
                    f"{json.dumps(example, ensure_ascii=False, sort_keys=True)}"
                    if example is not None
                    else ""
                )
            ),
        }
        messages_with_schema = [instructions, *base_messages]
        corrected = False
        correction_error: str | None = None
        corrections = self.settings.structured_output_corrections
        overall_timeout = float(self.settings.request_timeout_seconds)
        if remaining_timeout_seconds is not None:
            overall_timeout = min(overall_timeout, max(0.0, remaining_timeout_seconds))
        if overall_timeout <= 0:
            raise LlmError("language-model time budget is exhausted", category="timeout")
        correction_deadline = time.monotonic() + overall_timeout
        for correction_index in range(corrections + 1):
            if cancel is not None and cancel.is_set():
                raise LlmError("model call cancelled", category="cancelled")
            remaining = correction_deadline - time.monotonic()
            if remaining <= 0:
                raise LlmError("language-model request timed out", category="timeout")
            request_messages = messages_with_schema
            if correction_error is not None:
                request_messages = [
                    *messages_with_schema,
                    {
                        "role": "user",
                        "content": (
                            "Your previous JSON was rejected by the local validator. "
                            "Return a corrected JSON object only. Validation error: "
                            f"{correction_error}"
                        ),
                    },
                ]
            content, usage, finish_reason = self._complete(
                request_messages,
                purpose=purpose,
                response_format={"type": "json_object"},
                schema_version=schema_version,
                cancel=cancel,
                remaining_timeout_seconds=remaining,
            )
            if finish_reason == "length":
                error = LlmError(
                    "model JSON response was truncated by the token limit",
                    category="truncated",
                )
                if correction_index < corrections:
                    correction_error = str(error)
                    corrected = True
                    continue
                raise error
            try:
                payload = json.loads(content)
            except (TypeError, json.JSONDecodeError) as error:
                if correction_index < corrections:
                    correction_error = f"invalid JSON: {error}"
                    corrected = True
                    continue
                raise LlmError("model returned invalid JSON", category="invalid_json") from error
            try:
                value = _validate_schema(schema, payload)
            except (TypeError, ValueError) as error:
                if correction_index < corrections:
                    correction_error = str(error)
                    corrected = True
                    continue
                raise LlmError(
                    "model JSON failed the local schema", category="schema_error"
                ) from error
            self._replace_last_call(
                purpose=purpose,
                schema_version=schema_version,
                usage=usage,
                corrected=corrected,
            )
            return value
        raise LlmError("model JSON correction bound exhausted", category="schema_error")

    def _complete(
        self,
        messages: list[Mapping[str, str]],
        *,
        purpose: str,
        response_format: dict[str, str] | None,
        schema_version: str | None,
        cancel: Event | None,
        remaining_timeout_seconds: float | None,
    ) -> tuple[str, dict[str, Any], str | None]:
        if cancel is not None and cancel.is_set():
            raise LlmError("model call cancelled", category="cancelled")
        api_key = self._api_key_override or os.environ.get(self.settings.api_key_env)
        if not api_key:
            raise LlmError(
                f"set {self.settings.api_key_env} before using the language model",
                category="missing_api_key",
            )
        timeout_seconds = float(self.settings.request_timeout_seconds)
        if remaining_timeout_seconds is not None:
            timeout_seconds = min(timeout_seconds, max(0.0, remaining_timeout_seconds))
        if timeout_seconds <= 0:
            raise LlmError("language-model time budget is exhausted", category="timeout")
        deadline = time.monotonic() + timeout_seconds
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [dict(message) for message in messages],
            "max_tokens": self.settings.max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        owns_client = self._client is None
        client = self._client
        if client is None:
            timeout = httpx.Timeout(timeout_seconds)
            client = httpx.Client(timeout=timeout, transport=self._transport, follow_redirects=True)
        started = time.monotonic()
        try:
            for attempt in range(2):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LlmError("language-model request timed out", category="timeout")
                try:
                    response = client.post(
                        self.settings.base_url.rstrip("/") + "/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=max(0.001, remaining),
                    )
                except httpx.TimeoutException as error:
                    if attempt == 0:
                        continue
                    raise LlmError(
                        "language-model request timed out", category="timeout", retryable=True
                    ) from error
                except httpx.RequestError as error:
                    if attempt == 0:
                        continue
                    raise LlmError(
                        "language-model network request failed",
                        category="network_error",
                        retryable=True,
                    ) from error
                if response.status_code in {401, 403}:
                    raise LlmError("language-model authentication was rejected", category="auth")
                if response.status_code == 429:
                    if attempt == 0:
                        _bounded_retry_after(response, deadline=deadline)
                        continue
                    raise LlmError(
                        "language-model rate limit reached", category="rate_limited", retryable=True
                    )
                if 500 <= response.status_code <= 599:
                    if attempt == 0:
                        continue
                    raise LlmError(
                        f"language-model temporary server failure: HTTP {response.status_code}",
                        category="temporary_failure",
                        retryable=True,
                    )
                if response.status_code < 200 or response.status_code >= 300:
                    raise LlmError(
                        f"language-model request failed: HTTP {response.status_code}",
                        category="http_error",
                    )
                try:
                    data = response.json()
                except (ValueError, json.JSONDecodeError) as error:
                    raise LlmError(
                        "language-model response was not valid JSON", category="invalid_response"
                    ) from error
                try:
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                except (KeyError, IndexError, TypeError) as error:
                    raise LlmError(
                        "language-model response contained no message", category="empty_response"
                    ) from error
                if not isinstance(content, str) or not content.strip():
                    raise LlmError(
                        "language-model returned an empty message", category="empty_response"
                    )
                finish_reason = choice.get("finish_reason")
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                self.calls.append(
                    LlmCall(
                        purpose=purpose,
                        model=self.settings.model,
                        schema_version=schema_version,
                        elapsed_seconds=time.monotonic() - started,
                        usage={str(key): value for key, value in usage.items()},
                    )
                )
                return content.strip(), usage, finish_reason
            raise LlmError(
                "language-model request exhausted its retry bound", category="temporary_failure"
            )
        finally:
            if owns_client:
                client.close()

    def _replace_last_call(
        self, *, purpose: str, schema_version: str, usage: dict[str, Any], corrected: bool
    ) -> None:
        if not self.calls:
            return
        last = self.calls[-1]
        self.calls[-1] = LlmCall(
            purpose=purpose,
            model=last.model,
            schema_version=schema_version,
            elapsed_seconds=last.elapsed_seconds,
            usage={str(key): value for key, value in usage.items()},
            corrected=corrected,
        )


DeepSeekClient = LlmClient


def complete_text(
    client: LlmClient | AppConfig | LlmSettings,
    messages: Iterable[Mapping[str, str]],
    *,
    purpose: str = "text",
    cancel: Event | None = None,
) -> str:
    return _as_client(client).complete_text(messages, purpose=purpose, cancel=cancel)


def complete_json(
    client: LlmClient | AppConfig | LlmSettings,
    messages: Iterable[Mapping[str, str]],
    schema: type[BaseModel] | Mapping[str, Any] | Callable[[Any], Any],
    *,
    purpose: str = "json",
    example: Mapping[str, Any] | None = None,
    cancel: Event | None = None,
) -> Any:
    return _as_client(client).complete_json(
        messages, schema, purpose=purpose, example=example, cancel=cancel
    )


def _as_client(value: LlmClient | AppConfig | LlmSettings) -> LlmClient:
    return value if isinstance(value, LlmClient) else LlmClient(value)


def _schema_json(schema: Any) -> dict[str, Any]:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    if isinstance(schema, Mapping):
        return dict(schema)
    return {"type": "object"}


def _schema_version(schema: Mapping[str, Any]) -> str:
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _validate_schema(schema: Any, payload: Any) -> Any:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_validate(payload, strict=True)
    if callable(schema) and not isinstance(schema, Mapping):
        return schema(payload)
    if not isinstance(payload, dict):
        raise ValueError("JSON output must be an object")
    return payload


def _bounded_retry_after(response: httpx.Response, *, deadline: float | None = None) -> None:
    value = response.headers.get("Retry-After")
    if not value:
        return
    try:
        delay = min(2.0, max(0.0, float(value)))
    except ValueError:
        return
    if delay:
        if deadline is not None:
            delay = min(delay, max(0.0, deadline - time.monotonic()))
        if delay:
            time.sleep(delay)


__all__ = [
    "DeepSeekClient",
    "LlmCall",
    "LlmClient",
    "LlmError",
    "complete_json",
    "complete_text",
]
