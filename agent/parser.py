"""Core agent parsing flow: natural language -> Ollama -> structured JSON.

Minimum-invasive change vs previous version:
  Removed _attach_required_output_data() and _unavailable_solar_payload(), and
  their two call sites in Strategy A and Strategy B. Previously the code
  forcibly injected solar_radiation into the LLM output dict; this was wrong
  because solar data is a TOOL INPUT (via tools.get_solar_radiation), not an
  OUTPUT field. All dual-strategy / retry / tool-calling logic is unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, OpenAI

from .schema import AgentInput, AgentOutput, ParseError, SchemaError, validate_and_fix
from .tools import TOOL_SCHEMAS, dispatch_tool_call

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:7b"

PROMPT_PATH = Path(__file__).parent / "prompt.txt"
MAX_TOOL_ROUNDS = 3
MAX_RETRIES = 2


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _create_ollama_client() -> OpenAI:
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        timeout=120.0,
    )


def _build_messages(agent_input: AgentInput) -> list[dict[str, str]]:
    system_prompt = _load_system_prompt()
    ctx = agent_input.current_context

    user_content = (
        f"User command: {agent_input.user_command}\n\n"
        f"Current context:\n"
        f"- room: {ctx.room}\n"
        f"- current_temperature: {ctx.current_temperature}C\n"
        f"- current_hvac_mode: {ctx.current_hvac_mode}\n"
        f"- current_fan_mode: {ctx.current_fan_mode}\n"
        f"- occupied: {ctx.occupied}\n"
        f"- window_open: {ctx.window_open}\n"
        f"- current_time: {ctx.time}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_messages_with_tool_data(
    agent_input: AgentInput,
    tool_results: dict[str, dict],
) -> list[dict[str, str]]:
    system_prompt = _load_system_prompt()
    ctx = agent_input.current_context

    tool_section = ""
    if tool_results:
        tool_section = "\n\nSystem-fetched reference data:\n"
        for tool_name, result in tool_results.items():
            tool_section += f"\n[{tool_name}]\n{json.dumps(result, ensure_ascii=False, indent=2)}\n"

    user_content = (
        f"User command: {agent_input.user_command}\n\n"
        f"Current context:\n"
        f"- room: {ctx.room}\n"
        f"- current_temperature: {ctx.current_temperature}C\n"
        f"- current_hvac_mode: {ctx.current_hvac_mode}\n"
        f"- current_fan_mode: {ctx.current_fan_mode}\n"
        f"- occupied: {ctx.occupied}\n"
        f"- window_open: {ctx.window_open}\n"
        f"- current_time: {ctx.time}"
        f"{tool_section}\n\n"
        f"Use all relevant information above and output exactly one JSON object."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _extract_json(text: str) -> dict:
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

    raise ParseError(f"LLM response could not be parsed as JSON: {text[:300]}")


def _parse_with_tool_calling(
    agent_input: AgentInput,
    client: OpenAI,
    model: str,
) -> AgentOutput:
    messages = _build_messages(agent_input)
    tool_call_count = 0

    for _round in range(MAX_TOOL_ROUNDS + 1):
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
        }
        if tool_call_count < MAX_TOOL_ROUNDS:
            kwargs["tools"] = TOOL_SCHEMAS

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

            logger.info("[Strategy A] tool call: %s(%s)", fn_name, fn_args)
            result = dispatch_tool_call(fn_name, fn_args)
            tool_call_count += 1

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    result if result is not None else {"error": "tool call failed"},
                    ensure_ascii=False,
                ),
            })

            if tool_call_count >= MAX_TOOL_ROUNDS:
                logger.info("Reached tool call limit (%d)", MAX_TOOL_ROUNDS)
                break

    content = choice.message.content or ""
    logger.debug("[Strategy A] raw LLM output:\n%s", content)
    raw_output = _extract_json(content)
    return validate_and_fix(raw_output, agent_input.current_context)


def _auto_collect_tool_data(agent_input: AgentInput) -> dict[str, dict]:
    ctx = agent_input.current_context
    results: dict[str, dict] = {}

    room_data = dispatch_tool_call("get_room_status", {"room": ctx.room})
    if room_data:
        results["get_room_status"] = room_data

    weather_data = dispatch_tool_call("get_weather", {"city": "Pittsburgh"})
    if weather_data:
        results["get_weather"] = weather_data

    price_data = dispatch_tool_call("get_energy_price", {"time": ctx.time})
    if price_data:
        results["get_energy_price"] = price_data

    solar_data = dispatch_tool_call("get_solar_radiation", {"time": ctx.time})
    if solar_data:
        results["get_solar_radiation"] = solar_data

    hour = 12
    try:
        hour = int(ctx.time.split("T")[1].split(":")[0])
    except (IndexError, ValueError):
        pass

    if hour < 12:
        time_of_day = "morning"
    elif hour < 18:
        time_of_day = "afternoon"
    else:
        time_of_day = "night"

    habits_data = dispatch_tool_call(
        "get_user_habits",
        {"user_id": "default", "time_of_day": time_of_day},
    )
    if habits_data:
        results["get_user_habits"] = habits_data

    return results


def _parse_with_prompt_injection(
    agent_input: AgentInput,
    client: OpenAI,
    model: str,
) -> AgentOutput:
    logger.info("[Strategy B] using prompt injection mode")

    tool_results = _auto_collect_tool_data(agent_input)
    logger.info(
        "[Strategy B] collected %d tool results: %s",
        len(tool_results),
        list(tool_results.keys()),
    )

    messages = _build_messages_with_tool_data(agent_input, tool_results)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )

    content = response.choices[0].message.content or ""
    logger.debug("[Strategy B] raw LLM output:\n%s", content)
    raw_output = _extract_json(content)
    return validate_and_fix(raw_output, agent_input.current_context)


def check_ollama_connection(client: OpenAI | None = None) -> bool:
    if client is None:
        client = _create_ollama_client()
    try:
        client.models.list()
        return True
    except Exception as exc:
        logger.error("Could not connect to Ollama (%s): %s", OLLAMA_BASE_URL, exc)
        return False


def parse(
    agent_input: AgentInput,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    strategy: str = "auto",
) -> AgentOutput:
    if client is None:
        client = _create_ollama_client()

    last_error: Exception | None = None

    if strategy in ("auto", "tool_calling"):
        for attempt in range(MAX_RETRIES + 1):
            try:
                result = _parse_with_tool_calling(agent_input, client, model)
                logger.info("[Strategy A] success on attempt %d", attempt + 1)
                return result
            except (ParseError, SchemaError) as exc:
                last_error = exc
                logger.warning("[Strategy A] attempt %d failed: %s", attempt + 1, exc)
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                logger.error("[Strategy A] API connection failed: %s", exc)
                break

        if strategy == "tool_calling":
            raise last_error  # type: ignore[misc]
        logger.info("[Strategy A] all attempts failed, falling back to Strategy B")

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = _parse_with_prompt_injection(agent_input, client, model)
            logger.info("[Strategy B] success on attempt %d", attempt + 1)
            return result
        except (ParseError, SchemaError) as exc:
            last_error = exc
            logger.warning("[Strategy B] attempt %d failed: %s", attempt + 1, exc)

    raise last_error  # type: ignore[misc]
