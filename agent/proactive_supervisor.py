"""Proactive PID supervisor agent for the C3 condition."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

from .supervisor import DEFAULT_MODEL, create_ollama_client
from .tools import TOOL_SCHEMAS, dispatch_tool_call

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "proactive_supervisor_prompt.txt"
MAX_TOOL_ROUNDS = 4
MAX_RETRIES = 1
SUPERVISOR_TOOL_NAMES = {
    "get_pid_telemetry",
    "get_weather_forecast",
    "get_tariff_schedule",
    "get_schedule",
}
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
class ProactiveSupervisorInput:
    room: str
    city: str
    user_id: str
    user_objective: str
    current_time: str
    telemetry_window: dict[str, float | int | str]
    forecast_summary: dict[str, float | int | str]
    current_kp: float
    current_ki: float
    current_kd: float
    current_setpoint_c: float
    current_cost_weights: dict[str, float]


@dataclass
class ProactiveSupervisorDecision:
    mode: str
    action: str
    kp: float | None
    ki: float | None
    kd: float | None
    setpoint_c: float | None
    cost_weights: dict[str, float] | None
    rationale: str
    source: str


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _extract_date(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace(" ", "T")).date().isoformat()
    except ValueError:
        return value.split(" ")[0]


def _build_messages(supervisor_input: ProactiveSupervisorInput) -> list[dict[str, str]]:
    system_prompt = _load_system_prompt()
    user_content = (
        f"Room: {supervisor_input.room}\n"
        f"City: {supervisor_input.city}\n"
        f"User ID: {supervisor_input.user_id}\n"
        f"User objective: {supervisor_input.user_objective}\n"
        f"Current time: {supervisor_input.current_time}\n"
        f"Current gains: kp={supervisor_input.current_kp:.4f}, "
        f"ki={supervisor_input.current_ki:.4f}, kd={supervisor_input.current_kd:.4f}\n"
        f"Current online setpoint: {supervisor_input.current_setpoint_c:.3f} C\n"
        "Current online cost weights:\n"
        f"{json.dumps(supervisor_input.current_cost_weights, ensure_ascii=False, indent=2)}\n\n"
        "Recent 5-minute telemetry summary:\n"
        f"{json.dumps(supervisor_input.telemetry_window, ensure_ascii=False, indent=2)}\n\n"
        "Local proactive context summary:\n"
        f"{json.dumps(supervisor_input.forecast_summary, ensure_ascii=False, indent=2)}\n\n"
        "Before deciding, call get_pid_telemetry(room), get_weather_forecast(city, hours_ahead), "
        "get_tariff_schedule(time, hours_ahead), and get_schedule(user_id, date)."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_messages_with_tool_data(
    supervisor_input: ProactiveSupervisorInput,
    tool_results: dict[str, Any],
) -> list[dict[str, str]]:
    system_prompt = _load_system_prompt()
    user_content = (
        f"Room: {supervisor_input.room}\n"
        f"City: {supervisor_input.city}\n"
        f"User ID: {supervisor_input.user_id}\n"
        f"User objective: {supervisor_input.user_objective}\n"
        f"Current time: {supervisor_input.current_time}\n"
        f"Current gains: kp={supervisor_input.current_kp:.4f}, "
        f"ki={supervisor_input.current_ki:.4f}, kd={supervisor_input.current_kd:.4f}\n"
        f"Current online setpoint: {supervisor_input.current_setpoint_c:.3f} C\n"
        "Current online cost weights:\n"
        f"{json.dumps(supervisor_input.current_cost_weights, ensure_ascii=False, indent=2)}\n\n"
        "Recent 5-minute telemetry summary:\n"
        f"{json.dumps(supervisor_input.telemetry_window, ensure_ascii=False, indent=2)}\n\n"
        "Local proactive context summary:\n"
        f"{json.dumps(supervisor_input.forecast_summary, ensure_ascii=False, indent=2)}\n\n"
        "Tool results:\n"
        f"{json.dumps(tool_results, ensure_ascii=False, indent=2)}\n\n"
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


def _coerce_optional_cost_weights(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SupervisorSchemaError("cost_weights must be an object")

    parsed: dict[str, float] = {}
    for key in ("comfort", "energy", "response"):
        raw = value.get(key)
        if raw is None:
            continue
        try:
            parsed[key] = float(raw)
        except (TypeError, ValueError) as exc:
            raise SupervisorSchemaError(f"Could not parse cost weight {key}: {raw!r}") from exc

    return parsed or None


def _validate_decision(raw: dict[str, Any]) -> ProactiveSupervisorDecision:
    mode = str(raw.get("mode", "hold")).strip().lower()
    action = str(raw.get("action", "hold")).strip().lower()
    if mode not in {"hold", "proactive", "reactive"}:
        raise SupervisorSchemaError(f"Invalid supervisor mode: {mode!r}")
    if action not in {"hold", "set_pid", "set_setpoint", "set_cost_weights", "multi_action"}:
        raise SupervisorSchemaError(f"Invalid supervisor action: {action!r}")

    kp = _coerce_optional_float(raw.get("kp"))
    ki = _coerce_optional_float(raw.get("ki"))
    kd = _coerce_optional_float(raw.get("kd"))
    setpoint_c = _coerce_optional_float(
        raw.get("setpoint_C", raw.get("setpoint_c", raw.get("setpoint")))
    )
    cost_weights = _coerce_optional_cost_weights(raw.get("cost_weights"))
    rationale = str(raw.get("rationale", "")).strip()

    if mode == "hold":
        if action != "hold":
            raise SupervisorSchemaError("hold mode requires hold action")
        kp = None
        ki = None
        kd = None
        setpoint_c = None
        cost_weights = None
    else:
        has_pid = kp is not None or ki is not None or kd is not None
        has_setpoint = setpoint_c is not None
        has_cost_weights = cost_weights is not None
        if not (has_pid or has_setpoint or has_cost_weights):
            raise SupervisorSchemaError("non-hold action requires at least one control update")

    return ProactiveSupervisorDecision(
        mode=mode,
        action=action,
        kp=kp,
        ki=ki,
        kd=kd,
        setpoint_c=setpoint_c,
        cost_weights=cost_weights,
        rationale=rationale,
        source="agent",
    )


def _decide_with_tool_calling(
    supervisor_input: ProactiveSupervisorInput,
    client: OpenAI,
    model: str,
) -> ProactiveSupervisorDecision:
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

            logger.info("[C3 supervisor] tool call: %s(%s)", fn_name, fn_args)
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


def _prefetch_tool_data(supervisor_input: ProactiveSupervisorInput) -> dict[str, Any]:
    current_date = _extract_date(supervisor_input.current_time)
    return {
        "pid_telemetry": dispatch_tool_call(
            "get_pid_telemetry",
            {"room": supervisor_input.room},
        ),
        "weather_forecast": dispatch_tool_call(
            "get_weather_forecast",
            {"city": supervisor_input.city, "hours_ahead": 12},
        ),
        "tariff_schedule": dispatch_tool_call(
            "get_tariff_schedule",
            {"time": supervisor_input.current_time, "hours_ahead": 12},
        ),
        "schedule": dispatch_tool_call(
            "get_schedule",
            {"user_id": supervisor_input.user_id, "date": current_date},
        ),
    }


def _decide_with_prompt_injection(
    supervisor_input: ProactiveSupervisorInput,
    client: OpenAI,
    model: str,
) -> ProactiveSupervisorDecision:
    tool_results = _prefetch_tool_data(supervisor_input)
    messages = _build_messages_with_tool_data(supervisor_input, tool_results)
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


def decide_proactive_pid(
    supervisor_input: ProactiveSupervisorInput,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    strategy: str = "auto",
) -> ProactiveSupervisorDecision:
    if client is None:
        client = create_ollama_client()

    last_error: Exception | None = None

    if strategy in {"auto", "tool_calling"}:
        for attempt in range(MAX_RETRIES + 1):
            try:
                return _decide_with_tool_calling(supervisor_input, client, model)
            except (SupervisorParseError, SupervisorSchemaError) as exc:
                last_error = exc
                logger.warning("[C3 supervisor] tool-calling attempt %d failed: %s", attempt + 1, exc)
            except (APIConnectionError, APITimeoutError, OSError) as exc:
                last_error = exc
                logger.error("[C3 supervisor] tool-calling connection failed: %s", exc)
                break
        if strategy == "tool_calling":
            raise last_error  # type: ignore[misc]

    for attempt in range(MAX_RETRIES + 1):
        try:
            return _decide_with_prompt_injection(supervisor_input, client, model)
        except (SupervisorParseError, SupervisorSchemaError) as exc:
            last_error = exc
            logger.warning("[C3 supervisor] prompt-injection attempt %d failed: %s", attempt + 1, exc)
        except (APIConnectionError, APITimeoutError, OSError) as exc:
            last_error = exc
            logger.error("[C3 supervisor] prompt-injection connection failed: %s", exc)
            break

    raise last_error  # type: ignore[misc]
