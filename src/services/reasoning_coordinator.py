from __future__ import annotations

import structlog

from src.agent.prompt_compiler import render
from src.core.models import CandidateTrade
from src.intelligence.models import PromptContext
from src.models.reasoning import LLMDecision, MarketContext, PortfolioSnapshot
from src.services.decision_validator import validate
from src.services.llm_scheduler import LLMScheduler

logger = structlog.get_logger("reasoning_coordinator")


class ReasoningCoordinator:
    def __init__(self, llm_scheduler: LLMScheduler):
        self._llm_scheduler = llm_scheduler

    async def evaluate_candidate(
        self,
        candidate: CandidateTrade,
        market: MarketContext,
        portfolio: PortfolioSnapshot,
        evidence: PromptContext,
    ) -> LLMDecision:
        if self._llm_scheduler.is_degraded():
            logger.warning(
                "LLM degraded mode active, abstaining",
                symbol=candidate.symbol,
            )
            return LLMDecision(
                action="ABSTAIN",
                confidence=0.0,
                rationale="LLM_DEGRADED_MODE",
                risk_assessment="HALT",
            )

        try:
            prompt = render(candidate, market, portfolio, evidence)

            system_prompt = (
                "You are a quantitative reasoning engine in a SIMULATED, "
                "EDUCATIONAL paper-trading environment. You are not providing "
                "real financial advice. Output ONLY valid JSON matching the "
                "specified schema."
            )

            raw = await self._llm_scheduler.request_completion(
                system_prompt=system_prompt,
                user_prompt=prompt,
            )

            return validate(raw, evidence)

        except Exception as e:
            logger.warning(
                "LLM reasoning failed, defaulting to ABSTAIN",
                symbol=candidate.symbol,
                error=str(e),
            )
            return LLMDecision(
                action="ABSTAIN",
                confidence=0.0,
                rationale=f"LLM_UNAVAILABLE: {str(e)}",
                risk_assessment="HALT",
            )
