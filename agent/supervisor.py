"""Reactive PID supervisor agent for the C2 condition."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

from .tools import TOOL_SCHEMAS, dispatch_tool_call

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_TIMEOUT_SECONDS = 30.0
CONNECTIVITY_TIMEOUT_SECONDS = 3.0

PROMPT_PATH = Path(__file__).parent / "supervisor_prompt.txt"
MAX_TOOL_ROUNDS = 2
MAX_RETRIES = 1
SUPERVISOR_TOOL_NAMES = {"get_pid_telemetry"}
SUPERVISOR_TOOL_SCHEMAS = [
    schema
    for schema in TOOL_SCHEMAS
    if schema.get("function", {}).get("name") in SUPERVISOR_TOOL_NAMES
]


class SupervisorSchemaError(Exception):
    """Raised when the supervisor JSON is missing required structure."""


class SupervisorParseError(Exception):
    """Raised when the supervisor LLM response cannot be parsed as JSON."""


@dataclass
class ReactiveSupervisorInput:
    room: str
    user_objective: str
    current_time: str
    telemetry_window: dict[str, float | int | str]
    current_kp: float
    current_ki: float
    current_kd: float


@dataclass
class ReactiveSupervisorDecision:
    action: str
    kp: float | None
    ki: float | None
    kd: float | None
    rationale: str
    source: str


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _create_ollama_client(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> OpenAI:
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        timeout=timeout,
    )


def create_ollama_client(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> OpenAI:
    return _create_ollama_client(timeout=timeout)


def check_ollama_connection(client: OpenAI | None = None) -> bool:
    if client is None:
        client = create_ollama_client(timeout=CONNECTIVITY_TIMEOUT_SECONDS)
    try:
        client.models.list()
        return True
    except Exception as exc:
        logger.warning("[C2 supervisor] Ollama unavailable at %s: %s", OLLAMA_BASE_URL, exc)
        return False


def _build_messages(supervisor_input: ReactiveSupervisorInput) -> list[dict[str, str]]:
    system_prompt = _load_system_prompt()
    user_content = (
        f"Room: {supervisor_input.room}\n"
        f"User objective: {supervisor_input.user_objective}\n"
        f"Current time: {supervisor_input.current_time}\n"
        f"Current gains: kp={supervisor_input.current_kp:.4f}, "
        f"ki={supervisor_input.current_ki:.4f}, kd={supervisor_input.current_kd:.4f}\n"
        "Recent 5-minute telemetry summary:\n"
        f"{json.dumps(supervisor_input.telemetry_window, ensure_ascii=False, indent=2)}\n\n"
        "Before deciding, call get_pid_telemetry(room) once to inspect the latest PID snapshot."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_messages_with_tool_data(
    supervisor_input: ReactiveSupervisorInput,
    tool_result: dict[str, Any] | None,
) -> list[dict[str, str]]:
    system_prompt = _load_system_prompt()
    tool_block = ""
    if tool_result is not None:
        tool_block = (
            "\nLatest get_pid_telemetry(room) result:\n"
            f"{json.dumps(tool_result, ensure_ascii=False, indent=2)}\n"
        )
    user_content = (
        f"Room: {supervisor_input.room}\n"
        f"User objective: {supervisor_input.user_objective}\n"
        f"Current time: {supervisor_input.current_time}\n"
        f"Current gains: kp={supervisor_input.current_kp:.4f}, "
        f"ki={supervisor_input.current_ki:.4f}, kd={supervisor_input.current_kd:.4f}\n"
        "Recent 5-minute telemetry summary:\n"
        f"{json.dumps(supervisor_input.telemetry_window, ensure_ascii=False, indent=2)}"
        f"{tool_block}\n"
        "Return exactly one JSON object."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    code_block_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = code_block_pattern.search(text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_pattern = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
    match = brace_pattern.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise SupervisorParseError(f"Supervisor response could not be parsed as JSON: {text[:300]}")


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SupervisorSchemaError(f"Could not parse numeric gain value: {value!r}") from exc


def _validate_decision(
    raw: dict[str, Any],
) -> ReactiveSupervisorDecision:
    action = str(raw.get("action", "hold")).strip().lower()
    if action not in {"hold", "set_pid"}:
        raise SupervisorSchemaError(f"Invalid supervisor action: {action!r}")

    kp = _coerce_optional_float(raw.get("kp"))
    ki = _coerce_optional_float(raw.get("ki"))
    kd = _coerce_optional_float(raw.get("kd"))
    rationale = str(raw.get("rationale", "")).strip()

    if action == "hold":
        kp = None
        ki = None
        kd = None
    elif kp is None and ki is None and kd is None:
        raise SupervisorSchemaError("set_pid action requires at least one gain value")

    return ReactiveSupervisorDecision(
        action=action,
        kp=kp,
        ki=ki,
        kd=kd,
        rationale=rationale,
        source="agent",
    )


def _decide_with_tool_calling(
    supervisor_input: ReactiveSupervisorInput,
    client: OpenAI,
    model: str,
) -> ReactiveSupervisorDecision:
    messages = _build_messages(supervisor_input)
    tool_call_count = 0

    for _round in range(MAX_TOOL_ROUNDS + 1):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
        }
        if tool_call_count < MAX_TOOL_ROUNDS:
            kwargs["tools"] = SUPERVISOR_TOOL_SCHEMAS

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        if not choice.message.tool_calls:
            break

        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            logger.info("[C2 supervisor] tool call: %s(%s)", fn_name, fn_args)
            result = dispatch_tool_call(fn_name, fn_args)
            tool_call_count += 1

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result if result is not None else {"error": "tool call failed"},
                        ensure_ascii=False,
                    ),
                }
            )

            if tool_call_count >= MAX_TOOL_ROUNDS:
                break

    content = choice.message.content or ""
    raw = _extract_json(content)
    return _validate_decision(raw)


def _decide_with_prompt_injection(
    supervisor_input: ReactiveSupervisorInput,
    client: OpenAI,
    model: str,
) -> ReactiveSupervisorDecision:
    tool_result = dispatch_tool_call("get_pid_telemetry", {"room": supervisor_input.room})
    messages = _build_messages_with_tool_data(supervisor_input, tool_result)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    content = response.choices[0].message.content or ""
    raw = _extract_json(content)
    decision = _validate_decision(raw)
    decision.source = "agent_prompt_injection"
    return decision


def decide_reactive_pid(
    supervisor_input: ReactiveSupervisorInput,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    strategy: str = "auto",
) -> ReactiveSupervisorDecision:
    if client is None:
        client = _create_ollama_client()

    last_error: Exception | None = None

    if strategy in {"auto", "tool_calling"}:
        for attempt in range(MAX_RETRIES + 1):
            try:
                return _decide_with_tool_calling(supervisor_input, client, model)
            except (SupervisorParseError, SupervisorSchemaError) as exc:
                last_error = exc
                logger.warning("[C2 supervisor] tool-calling attempt %d failed: %s", attempt + 1, exc)
            except (APIConnectionError, APITimeoutError, OSError) as exc:
                last_error = exc
                logger.error("[C2 supervisor] tool-calling connection failed: %s", exc)
                break
        if strategy == "tool_calling":
            raise last_error  # type: ignore[misc]

    for attempt in range(MAX_RETRIES + 1):
        try:
            return _decide_with_prompt_injection(supervisor_input, client, model)
        except (SupervisorParseError, SupervisorSchemaError) as exc:
            last_error = exc
            logger.warning("[C2 supervisor] prompt-injection attempt %d failed: %s", attempt + 1, exc)
        except (APIConnectionError, APITimeoutError, OSError) as exc:
            last_error = exc
            logger.error("[C2 supervisor] prompt-injection connection failed: %s", exc)
            break

    raise last_error  # type: ignore[misc]
