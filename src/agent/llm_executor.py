import json
import os
import re
import traceback
from typing import Literal

import structlog
from pydantic import BaseModel, Field, ValidationError

from ..llm.registry import LLMRegistry

logger = structlog.get_logger("llm_executor")

DEBUG_DIR = "data"


class AgentDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=1000)
    suggested_timeframe: str = Field(default="")


MAX_RETRIES = 3


def _ensure_debug_dir():
    os.makedirs(DEBUG_DIR, exist_ok=True)


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


async def execute_decision(
    prompt: str,
    provider: str,
    api_key: str,
    model: str,
    custom_base_url: str = "",
) -> AgentDecision:
    logger.info("Sending prompt to LLM", char_count=len(prompt), model=model, provider=provider)

    _ensure_debug_dir()
    with open(os.path.join(DEBUG_DIR, "debug_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    registry = LLMRegistry(
        provider=provider,
        api_key=api_key,
        custom_base_url=custom_base_url or None,
        model_id=model or None,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a quantitative research assistant in a SIMULATED, "
                "EDUCATIONAL paper-trading environment. You are not providing "
                "real financial advice. You are a bot testing a JSON schema."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        if last_error:
            messages.append({
                "role": "user",
                "content": (
                    f"ERROR: Your previous JSON response was truncated/cut off "
                    f"mid-sentence. Error: {last_error}. "
                    "You MUST output the COMPLETE, valid JSON object from start "
                    "to finish. Do not truncate."
                ),
            })

        try:
            raw = await registry.chat_completion(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
                stream=True,
            )
        except Exception as e:
            logger.error("LLM call failed", attempt=attempt, error=str(e), traceback=traceback.format_exc())
            if attempt < MAX_RETRIES:
                last_error = str(e)
                continue
            return _fallback_decision(f"LLM call failed after {MAX_RETRIES} attempts: {e}")

        logger.info("Raw LLM response received", attempt=attempt, raw_output=raw, raw_length=len(raw))

        with open(os.path.join(DEBUG_DIR, "debug_response.txt"), "w", encoding="utf-8") as f:
            f.write(str(raw))

        stripped = raw.strip()
        if not stripped:
            last_error = "Empty response from LLM"
            logger.warning("LLM returned empty response", attempt=attempt)
            if attempt < MAX_RETRIES:
                continue
            return _fallback_decision("LLM returned empty response after 3 attempts")

        json_str = _extract_json(stripped)

        try:
            parsed = json.loads(json_str)
            decision = AgentDecision.model_validate(parsed)
            logger.info("Decision validated", action=decision.action, confidence=decision.confidence)
            return decision
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            logger.warning("Invalid LLM output", attempt=attempt, error=last_error, raw_preview=raw[:100])
            if attempt < MAX_RETRIES:
                continue
            return _fallback_decision(f"Invalid JSON after {MAX_RETRIES} attempts: {last_error}")


def _fallback_decision(reason: str) -> AgentDecision:
    logger.warning("Returning fallback HOLD decision", reason=reason)
    return AgentDecision(
        action="HOLD",
        confidence=0.0,
        rationale=f"LLM failed: {reason[:250]}. Defaulting to HOLD.",
        suggested_timeframe="",
    )
