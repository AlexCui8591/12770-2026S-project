"""C2 Supervisor: call LLM every 5 minutes, apply decision to controller.

Minimal implementation for Demo Version A (time-pressure version):
- LLM sees only current telemetry + user's original text
- 3 actions: hold / set_pid / set_setpoint (no set_weights)
- Falls back to "hold" on any failure (parse error, API error, invalid params)

Design: this module is independent of demo.py so it can be tested separately.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, OpenAI

from agent.mock_pid import MockPIDController


logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:7b"

SUPERVISOR_PROMPT_PATH = Path(__file__).parent / "supervisor_prompt.txt"

# Safety bounds for LLM-produced params (mirrors MockPID.set_gains limits)
KP_MIN, KP_MAX = 0.5, 5.0
KI_MIN, KI_MAX = 0.0, 0.1
KD_MIN, KD_MAX = 0.0, 2.0
SETPOINT_MIN, SETPOINT_MAX = 16.0, 30.0


def _load_prompt() -> str:
    return SUPERVISOR_PROMPT_PATH.read_text(encoding="utf-8")


def _create_client() -> OpenAI:
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        timeout=60.0,
    )


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction (same approach as parser.py)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Grab the first balanced {...}
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse supervisor JSON: {text[:200]}")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _validate_and_clamp(decision: dict) -> dict:
    """Normalize LLM decision. Any validation failure → fallback to hold."""
    action = decision.get("action", "").lower()
    params = decision.get("params", {}) or {}
    rationale = decision.get("rationale", "")

    if action == "hold":
        return {"action": "hold", "params": {}, "rationale": rationale}

    if action == "set_pid":
        try:
            kp = _clamp(params["kp"], KP_MIN, KP_MAX)
            ki = _clamp(params["ki"], KI_MIN, KI_MAX)
            kd = _clamp(params["kd"], KD_MIN, KD_MAX)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("set_pid params invalid (%s), falling back to hold", e)
            return {"action": "hold", "params": {}, "rationale": f"(fallback: invalid set_pid params)"}
        return {
            "action": "set_pid",
            "params": {"kp": kp, "ki": ki, "kd": kd},
            "rationale": rationale,
        }

    if action == "set_setpoint":
        try:
            sp = _clamp(params["setpoint"], SETPOINT_MIN, SETPOINT_MAX)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("set_setpoint params invalid (%s), falling back to hold", e)
            return {"action": "hold", "params": {}, "rationale": f"(fallback: invalid setpoint)"}
        return {
            "action": "set_setpoint",
            "params": {"setpoint": sp},
            "rationale": rationale,
        }

    # Unknown action → hold
    logger.warning("Unknown action %r, falling back to hold", action)
    return {"action": "hold", "params": {}, "rationale": f"(fallback: unknown action {action!r})"}


def ask_supervisor(
    telemetry: dict,
    user_text: str,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Ask the LLM supervisor for a decision. Always returns a valid decision
    (falls back to hold on any failure)."""
    if client is None:
        client = _create_client()

    system_prompt = _load_prompt()

    user_content = (
        f"User's original intent: {user_text}\n\n"
        f"Current controller telemetry:\n"
        f"- kp={telemetry['kp']}, ki={telemetry['ki']}, kd={telemetry['kd']}\n"
        f"- setpoint={telemetry['setpoint']}°C\n"
        f"- indoor_temp={telemetry['indoor_temp']}°C\n"
        f"- tracking_error={telemetry['tracking_error']}°C\n"
        f"- control_signal={telemetry['control_signal']}W\n"
        f"- oscillation_count={telemetry['oscillation_count']}\n"
        f"- cost_J={telemetry['cost_J']}\n"
        f"- cumulative_energy_kwh={telemetry['cumulative_energy_kwh']}\n\n"
        f"Output exactly one JSON object with your decision."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
    except (APIConnectionError, APITimeoutError) as e:
        logger.error("Supervisor LLM call failed: %s", e)
        return {"action": "hold", "params": {}, "rationale": f"(fallback: API error {e})"}

    content = response.choices[0].message.content or ""
    try:
        raw = _extract_json(content)
    except ValueError as e:
        logger.warning("Supervisor JSON parse failed: %s", e)
        return {"action": "hold", "params": {}, "rationale": "(fallback: JSON parse failed)"}

    return _validate_and_clamp(raw)


def apply_decision(controller: MockPIDController, decision: dict) -> None:
    """Execute the decision on the controller."""
    action = decision["action"]
    params = decision["params"]

    if action == "hold":
        return
    if action == "set_pid":
        controller.set_gains(kp=params["kp"], ki=params["ki"], kd=params["kd"])
        return
    if action == "set_setpoint":
        controller.set_setpoint(params["setpoint"])
        return
    # Unknown actions already filtered by validate; defensive no-op
    logger.warning("apply_decision: unknown action %r, ignored", action)