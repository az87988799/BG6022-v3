"""One bounded DeepSeek HTTP client used by intake, planning, repair, and answers."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from threading import Event
from typing import Any

import httpx
from pydantic import BaseModel

from bg6022.config import AppConfig, LlmSettings


class LlmError(RuntimeError):
    """A user-actionable model failure without retaining credentials."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool = False,
        purpose: str | None = None,
        diagnostics: tuple[dict[str, str], ...] = (),
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.purpose = purpose
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class LlmCall:
    purpose: str
    request_model: str
    schema_version: str | None
    elapsed_seconds: float
    usage: dict[str, Any]
    response_model: str | None = None
    category: str = "response"
    finish_reason: str | None = None
    body_length: int | None = None
    transport_attempts: int = 1
    structured_correction_count: int = 0
    corrected: bool = False

    @property
    def model(self) -> str:
        """Compatibility alias for the configured/request model name."""

        return self.request_model


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
        self._replace_last_call(category="success")
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
            raise LlmError(
                "language-model time budget is exhausted", category="timeout", purpose=purpose
            )
        correction_deadline = time.monotonic() + overall_timeout
        for correction_index in range(corrections + 1):
            if cancel is not None and cancel.is_set():
                raise LlmError("model call cancelled", category="cancelled", purpose=purpose)
            remaining = correction_deadline - time.monotonic()
            if remaining <= 0:
                raise LlmError(
                    "language-model request timed out", category="timeout", purpose=purpose
                )
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
            if cancel is not None and cancel.is_set():
                self._replace_last_call(
                    category="cancelled",
                    structured_correction_count=correction_index,
                )
                raise LlmError("model call cancelled", category="cancelled", purpose=purpose)
            if finish_reason == "length":
                error = LlmError(
                    "model JSON response was truncated by the token limit",
                    category="truncated",
                    purpose=purpose,
                )
                self._replace_last_call(
                    category=error.category,
                    structured_correction_count=correction_index,
                )
                if correction_index < corrections:
                    correction_error = str(error)
                    corrected = True
                    continue
                raise error
            if not content.strip():
                error = LlmError(
                    "model returned an empty JSON response",
                    category="empty_response",
                    purpose=purpose,
                )
                self._replace_last_call(
                    category=error.category,
                    structured_correction_count=correction_index,
                )
                if correction_index < corrections:
                    correction_error = str(error)
                    corrected = True
                    continue
                raise error
            try:
                payload = json.loads(content)
            except (TypeError, json.JSONDecodeError) as error:
                self._replace_last_call(
                    category="invalid_json",
                    structured_correction_count=correction_index,
                )
                if correction_index < corrections:
                    correction_error = f"invalid JSON: {error}"
                    corrected = True
                    continue
                raise LlmError(
                    "model returned invalid JSON", category="invalid_json", purpose=purpose
                ) from error
            try:
                value = _validate_schema(schema, payload)
            except (TypeError, ValueError) as error:
                ambiguous_energy = _is_ambiguous_energy_request(payload)
                failure_category = "ambiguous_result" if ambiguous_energy else "schema_error"
                diagnostics = _schema_failure_diagnostics(error)
                self._replace_last_call(
                    category=failure_category,
                    structured_correction_count=correction_index,
                )
                if correction_index < corrections:
                    correction_error = _schema_failure_feedback(diagnostics)
                    corrected = True
                    continue
                raise LlmError(
                    "model JSON failed the local schema",
                    category=failure_category,
                    purpose=purpose,
                    diagnostics=diagnostics,
                ) from error
            self._replace_last_call(
                category="success",
                structured_correction_count=correction_index,
                corrected=corrected,
            )
            return value
        raise LlmError(
            "model JSON correction bound exhausted", category="schema_error", purpose=purpose
        )

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
            raise LlmError("model call cancelled", category="cancelled", purpose=purpose)
        api_key = self._api_key_override or os.environ.get(self.settings.api_key_env)
        if not api_key:
            raise LlmError(
                f"set {self.settings.api_key_env} before using the language model",
                category="missing_api_key",
                purpose=purpose,
            )
        timeout_seconds = float(self.settings.request_timeout_seconds)
        if remaining_timeout_seconds is not None:
            timeout_seconds = min(timeout_seconds, max(0.0, remaining_timeout_seconds))
        if timeout_seconds <= 0:
            raise LlmError(
                "language-model time budget is exhausted", category="timeout", purpose=purpose
            )
        deadline = time.monotonic() + timeout_seconds
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [dict(message) for message in messages],
            "max_tokens": self.settings.max_tokens,
        }
        if (
            purpose in {"intake", "semantic", "answer"}
            and "deepseek.com" in self.settings.base_url.casefold()
        ):
            # Intake and the public answer protocol are bounded extraction/
            # presentation calls. DeepSeek's default thinking mode can consume
            # the full output budget before emitting a usable body.
            payload["thinking"] = {"type": "disabled"}
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        owns_client = self._client is None
        client = self._client
        if client is None:
            timeout = httpx.Timeout(timeout_seconds)
            client = httpx.Client(timeout=timeout, transport=self._transport, follow_redirects=True)
        started = time.monotonic()
        transport_attempts = 0
        response_model: str | None = None
        finish_reason: str | None = None
        body_length: int | None = None
        usage: dict[str, Any] = {}
        call_recorded = False

        def record_call(category: str) -> None:
            nonlocal call_recorded
            if call_recorded:
                self._replace_last_call(category=category)
                return
            self.calls.append(
                LlmCall(
                    purpose=purpose,
                    request_model=self.settings.model,
                    response_model=response_model,
                    schema_version=schema_version,
                    elapsed_seconds=time.monotonic() - started,
                    usage={str(key): value for key, value in usage.items()},
                    category=category,
                    finish_reason=finish_reason,
                    body_length=body_length,
                    transport_attempts=max(transport_attempts, 1),
                )
            )
            call_recorded = True

        try:
            for attempt in range(2):
                transport_attempts = attempt + 1
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
                    if cancel is not None and cancel.is_set():
                        raise LlmError("model call cancelled", category="cancelled")
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
                body_length = len(response.content)
                if response.status_code in {401, 403}:
                    raise LlmError("language-model authentication was rejected", category="auth")
                if response.status_code == 429:
                    if attempt == 0:
                        _bounded_retry_after(response, deadline=deadline, cancel=cancel)
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
                if not isinstance(data, Mapping):
                    raise LlmError(
                        "language-model response was not a JSON object",
                        category="invalid_response",
                    )
                response_model = _optional_text(data.get("model"))
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise LlmError(
                        "language-model response has no valid choices",
                        category="invalid_response",
                    )
                choice = choices[0]
                if not isinstance(choice, Mapping):
                    raise LlmError(
                        "language-model response choice has an invalid structure",
                        category="invalid_response",
                    )
                finish_reason = _optional_text(choice.get("finish_reason"))
                message = choice.get("message")
                if not isinstance(message, Mapping):
                    if response_format is None or finish_reason != "length":
                        raise LlmError(
                            "language-model response message has an invalid structure",
                            category="invalid_response",
                        )
                    content = ""
                else:
                    content = message.get("content")
                    if not isinstance(content, str):
                        if response_format is None or finish_reason != "length":
                            raise LlmError(
                                "language-model response content has an invalid structure",
                                category="invalid_response",
                            )
                        content = ""
                body_length = len(content)
                category = (
                    "truncated"
                    if finish_reason == "length"
                    else "empty_response"
                    if not content.strip()
                    else "response"
                )
                record_call(category)
                if response_format is None and finish_reason == "length":
                    raise LlmError("language-model output was truncated", category="truncated")
                if not content.strip() and response_format is None:
                    raise LlmError(
                        "language-model returned an empty message", category="empty_response"
                    )
                return content.strip(), usage, finish_reason
            raise LlmError(
                "language-model request exhausted its retry bound", category="temporary_failure"
            )
        except LlmError as error:
            error.purpose = purpose
            record_call(error.category)
            raise
        finally:
            if owns_client:
                client.close()

    def _replace_last_call(
        self,
        *,
        corrected: bool | None = None,
        category: str | None = None,
        structured_correction_count: int | None = None,
    ) -> None:
        if not self.calls:
            return
        changes: dict[str, Any] = {}
        if corrected is not None:
            changes["corrected"] = corrected
        if category is not None:
            changes["category"] = category
        if structured_correction_count is not None:
            changes["structured_correction_count"] = structured_correction_count
        self.calls[-1] = replace(self.calls[-1], **changes)


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


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _is_ambiguous_energy_request(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("intent") != "chemistry_compute":
        return False
    operations = value.get("operations", [])
    requested_results = value.get("requested_results", [])
    if not isinstance(operations, list) or not isinstance(requested_results, list):
        return False
    if not all(isinstance(item, str) for item in operations):
        return False
    if not all(isinstance(item, str) for item in requested_results):
        return False
    return {"Opt", "SP"}.issubset(operations) and any(
        item in {"energy", "electronic_energy"} for item in requested_results
    )


def _validate_schema(schema: Any, payload: Any) -> Any:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_validate(payload, strict=True)
    if callable(schema) and not isinstance(schema, Mapping):
        return schema(payload)
    if not isinstance(payload, dict):
        raise ValueError("JSON output must be an object")
    return payload


def _schema_failure_diagnostics(error: BaseException) -> tuple[dict[str, str], ...]:
    """Keep field paths and messages while omitting echoed input values."""

    errors_method = getattr(error, "errors", None)
    if callable(errors_method):
        try:
            errors = errors_method(include_url=False)
        except TypeError:
            errors = errors_method()
        diagnostics = []
        for item in errors:
            location = item.get("loc", ())
            path = ".".join(str(part) for part in location) or "$"
            message = str(item.get("msg") or "value failed local validation")
            diagnostics.append({"path": path, "message": message[:512]})
        if diagnostics:
            return tuple(diagnostics[:12])
    return ({"path": "$", "message": str(error)[:512]},)


def _schema_failure_feedback(diagnostics: tuple[dict[str, str], ...]) -> str:
    return "; ".join(f"{item['path']}: {item['message']}" for item in diagnostics) or (
        "the response failed local schema validation"
    )


def _bounded_retry_after(
    response: httpx.Response, *, deadline: float | None = None, cancel: Event | None = None
) -> None:
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
            if cancel is not None:
                if cancel.wait(delay):
                    raise LlmError("model call cancelled", category="cancelled")
            else:
                time.sleep(delay)


__all__ = [
    "DeepSeekClient",
    "LlmCall",
    "LlmClient",
    "LlmError",
    "complete_json",
    "complete_text",
]
