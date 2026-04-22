"""
Phase 1: Intent parser.

Single-shot LLM call: natural-language query -> StructuredDirective.

Design decisions (per Phase 1 plan):
  - No retry on failure. Record the failure with a typed error category so
    Phase 3 dual-strategy can decide whether to fallback to prompt-injection.
  - Failure types are *classified*, not just logged. This matters because:
      * JSON_INVALID  -> likely fixable by format coercion or re-call
      * SCHEMA_VIOLATION -> model "understood the task" but output wrong shape
                           -> prompt-injection fallback is the right move
      * VALUE_OUT_OF_RANGE -> Phase 4 validator can clamp and still proceed
      * MISSING_FIELD -> Phase 4 validator can fill with context-aware default
      * LLM_ERROR -> infrastructure failure, different response needed
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import ollama
from pydantic import ValidationError

from agent.schema import StructuredDirective


DEFAULT_MODEL = "qwen2.5:7b"
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "intent_parser_v1.txt"


class ParseErrorType(str, Enum):
    NONE = "none"
    LLM_ERROR = "llm_error"  # Ollama call failed (network/process)
    JSON_INVALID = "json_invalid"  # Could not extract JSON from response
    MISSING_FIELD = "missing_field"  # Required field absent
    VALUE_OUT_OF_RANGE = "value_out_of_range"  # e.g., target_temperature=45
    SCHEMA_VIOLATION = "schema_violation"  # Wrong types, bad enum values, etc.


@dataclass
class ParseResult:
    success: bool
    directive: Optional[StructuredDirective]
    raw_output: str
    error: Optional[str]
    error_type: ParseErrorType
    latency_ms: float
    json_repaired: bool = False

    def to_log_dict(self) -> dict:
        """Flatten for SQLite/JSONL logging."""
        return {
            "success": self.success,
            "error_type": self.error_type.value,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "json_repaired": self.json_repaired,
            "raw_output": self.raw_output,
            "directive": self.directive.model_dump() if self.directive else None,
        }


def _load_system_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"System prompt not found at {PROMPT_PATH}. "
            f"Create prompts/intent_parser_v1.txt first."
        )
    return PROMPT_PATH.read_text(encoding="utf-8")


def _extract_json(raw: str) -> tuple[Optional[dict], bool]:
    """Best-effort JSON extraction. Returns (parsed_dict, was_repaired).

    No re-calling the LLM; just local string fix-up. This does NOT count as
    a retry — we're repairing what the model already said, not asking again.
    """
    # Direct parse
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    stripped = raw.strip()
    for fence in ("```json", "```"):
        if stripped.startswith(fence):
            stripped = stripped[len(fence):].lstrip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    try:
        return json.loads(stripped), True
    except json.JSONDecodeError:
        pass

    # Extract first {...} block
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start:end + 1]
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            pass

    # Optional: json-repair library (pip install json-repair)
    try:
        from json_repair import repair_json
        repaired = repair_json(raw, return_objects=True)
        if isinstance(repaired, dict):
            return repaired, True
    except ImportError:
        pass
    except Exception:
        pass

    return None, False


def _classify_validation_error(exc: ValidationError) -> ParseErrorType:
    """Map Pydantic errors to our error taxonomy."""
    errors = exc.errors()
    if any(err["type"] == "missing" for err in errors):
        return ParseErrorType.MISSING_FIELD
    range_markers = ("greater_than", "less_than", "greater_than_equal", "less_than_equal")
    if any(any(m in err["type"] for m in range_markers) for err in errors):
        return ParseErrorType.VALUE_OUT_OF_RANGE
    return ParseErrorType.SCHEMA_VIOLATION


def parse_intent(
    nl_query: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
) -> ParseResult:
    """Parse a natural-language HVAC directive.

    Single LLM call, no retry. Returns a ParseResult that is ALWAYS valid
    (success or typed failure) — callers never see exceptions from the LLM.
    """
    system_prompt = system_prompt or _load_system_prompt()
    start = time.time()

    # 1. LLM call (JSON mode enforces well-formed top-level JSON in Qwen2.5)
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": nl_query},
            ],
            format="json",
            options={"temperature": temperature},
        )
        raw = response["message"]["content"]
    except Exception as e:
        return ParseResult(
            success=False, directive=None, raw_output="",
            error=f"Ollama call failed: {e}",
            error_type=ParseErrorType.LLM_ERROR,
            latency_ms=(time.time() - start) * 1000,
        )

    # 2. JSON extraction (with local repair)
    parsed, repaired = _extract_json(raw)
    if parsed is None:
        return ParseResult(
            success=False, directive=None, raw_output=raw,
            error="Could not extract valid JSON from model output",
            error_type=ParseErrorType.JSON_INVALID,
            latency_ms=(time.time() - start) * 1000,
        )

    # 3. Schema validation
    try:
        directive = StructuredDirective(**parsed)
    except ValidationError as e:
        return ParseResult(
            success=False, directive=None, raw_output=raw,
            error=str(e),
            error_type=_classify_validation_error(e),
            latency_ms=(time.time() - start) * 1000,
            json_repaired=repaired,
        )
    except TypeError as e:
        # Wrong top-level type (e.g. list instead of dict)
        return ParseResult(
            success=False, directive=None, raw_output=raw,
            error=f"Unexpected JSON structure: {e}",
            error_type=ParseErrorType.SCHEMA_VIOLATION,
            latency_ms=(time.time() - start) * 1000,
            json_repaired=repaired,
        )

    return ParseResult(
        success=True, directive=directive, raw_output=raw,
        error=None, error_type=ParseErrorType.NONE,
        latency_ms=(time.time() - start) * 1000,
        json_repaired=repaired,
    )


if __name__ == "__main__":
    # Quick sanity check — run `python -m agent.intent_parser`
    test_query = "Make the bedroom cooler, but keep electricity costs low."
    print(f"Query: {test_query}\n")
    result = parse_intent(test_query)
    print(f"Success: {result.success}")
    print(f"Error type: {result.error_type.value}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    if result.directive:
        print(f"Directive: {result.directive.model_dump_json(indent=2)}")
    else:
        print(f"Error: {result.error}")
        print(f"Raw output: {result.raw_output[:500]}")
