from __future__ import annotations

import hashlib
import structlog

from src.agent.prompt_compiler import render
from src.core.events import EventBus
from src.core.models import CandidateTrade, SystemEvent
from src.intelligence.models import PromptContext
from src.models.learning.reasoning_episode import ReasoningEpisode
from src.models.reasoning import LLMDecision, MarketContext, PortfolioSnapshot
from src.services.decision_validator import validate, apply_ceiling
from src.services.llm_scheduler import LLMScheduler
from src.services.self_critique import CritiqueVerdict, SelfCritiqueEngine, SelfCritiqueResult
from src.storage.learning.learning_corpus import LearningCorpus

logger = structlog.get_logger("reasoning_coordinator")


class ReasoningCoordinator:
    def __init__(self, llm_scheduler: LLMScheduler, event_bus: EventBus, corpus: LearningCorpus):
        self._llm_scheduler = llm_scheduler
        self._event_bus = event_bus
        self._corpus = corpus
        self._self_critique = SelfCritiqueEngine()

    async def evaluate_candidate(
        self,
        candidate: CandidateTrade,
        market: MarketContext,
        portfolio: PortfolioSnapshot,
        evidence: PromptContext,
        show_confidence: bool = False,
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
            prompt = render(candidate, market, portfolio, evidence, show_confidence)

            # Log evidence being fed into the LLM so the feedback loop is observable
            evidence_source = evidence.evidence_source if evidence else "NONE"
            evidence_token_count = evidence.token_count if evidence else 0
            if evidence and evidence.context_string:
                summary_line = evidence.context_string.split("\n")[0] if evidence.context_string else ""
            else:
                summary_line = "no evidence"
            logger.info(
                "LLM_PROMPT_EVIDENCE",
                symbol=candidate.symbol,
                evidence_source=evidence_source,
                evidence_tokens=evidence_token_count,
                evidence_summary=summary_line,
                _force_log=True,
            )

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

            decision = validate(raw, evidence)

            episode = ReasoningEpisode(
                decision_id=candidate.opportunity_id or candidate.symbol,
                symbol=candidate.symbol,
                timeframe=market.timeframe if market else "",
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
                prompt_preview=prompt[:200],
                market_summary=market.model_dump(mode="json") if market else {},
                evidence_summary={"source": evidence.evidence_source if evidence else "NONE", "tokens": evidence.token_count if evidence else 0},
                portfolio_summary=portfolio.model_dump(mode="json") if portfolio else {},
                action=decision.action,
                confidence=decision.confidence,
                rationale=decision.rationale,
                risk_assessment=decision.risk_assessment,
                llm_response_raw=raw,
                execution_id="",
                correlation_id="",
                opportunity_id=candidate.opportunity_id or "",
                strategy_version="",
            )
            logger.info(
                "REASONING_EPISODE_CAPTURED",
                episode_id=episode.episode_id,
                symbol=episode.symbol,
                action=episode.action,
                confidence=episode.confidence,
                evidence_source=episode.evidence_summary.get("source"),
                prompt_hash=episode.prompt_hash,
                _force_log=True,
            )
            self._corpus.save_reasoning_episode(episode)
            logger.info(
                "REASONING_EPISODE_SAVED",
                episode_id=episode.episode_id,
                symbol=episode.symbol,
                _force_log=True,
            )
            self._event_bus.publish_nowait(SystemEvent(
                event_type="REASONING_RECORDED",
                service_name="ReasoningCoordinator",
                payload={"episode_id": episode.episode_id, "symbol": candidate.symbol, "action": decision.action},
            ))
            logger.info(
                "REASONING_RECORDED_PUBLISHED",
                episode_id=episode.episode_id,
                symbol=candidate.symbol,
                action=decision.action,
                _force_log=True,
            )

            # Phase 5C — Self-critique
            logger.info(
                "SELF_CRITIQUE_STARTED",
                episode_id=episode.episode_id,
                symbol=candidate.symbol,
                action=decision.action,
                confidence=decision.confidence,
                _force_log=True,
            )
            critique = self._self_critique.critique(
                decision, portfolio,
                evidence_source=evidence_source,
                evidence_tokens=evidence_token_count,
            )

            if critique.verdict == CritiqueVerdict.OVERTURN:
                self._corpus.update_reasoning_episode(
                    episode.episode_id,
                    action=critique.revised_action,
                    confidence=critique.revised_confidence,
                    rationale=critique.revised_rationale,
                    risk_assessment=critique.revised_risk_assessment,
                    metadata={"critique_verdict": "OVERTURN", "critique_rationale": critique.critique_rationale},
                )
                overturned = LLMDecision(
                    action=critique.revised_action or "HOLD",
                    confidence=critique.revised_confidence or 0.0,
                    rationale=critique.revised_rationale or decision.rationale,
                    risk_assessment=critique.revised_risk_assessment or decision.risk_assessment,
                )
                logger.info(
                    "SELF_CRITIQUE_VERDICT",
                    episode_id=episode.episode_id,
                    verdict="OVERTURN",
                    original_action=decision.action,
                    revised_action=overturned.action,
                    critique_rationale=critique.critique_rationale,
                    _force_log=True,
                )
                logger.info(
                    "DECISION_FINALIZED",
                    episode_id=episode.episode_id,
                    action=overturned.action,
                    confidence=overturned.confidence,
                    source="critique_overturned",
                    _force_log=True,
                )
                return overturned

            if critique.verdict == CritiqueVerdict.REVISE:
                revised = LLMDecision(
                    action=decision.action,
                    confidence=critique.revised_confidence or decision.confidence,
                    rationale=critique.revised_rationale or decision.rationale,
                    risk_assessment=critique.revised_risk_assessment or decision.risk_assessment,
                )
                self._corpus.update_reasoning_episode(
                    episode.episode_id,
                    confidence=revised.confidence,
                    rationale=revised.rationale,
                    risk_assessment=revised.risk_assessment,
                    metadata={"critique_verdict": "REVISE", "critique_rationale": critique.critique_rationale},
                )
                revised = apply_ceiling(revised, evidence)
                logger.info(
                    "SELF_CRITIQUE_VERDICT",
                    episode_id=episode.episode_id,
                    verdict="REVISE",
                    original_action=decision.action,
                    revised_confidence=revised.confidence,
                    critique_rationale=critique.critique_rationale,
                    _force_log=True,
                )
                logger.info(
                    "DECISION_FINALIZED",
                    episode_id=episode.episode_id,
                    action=revised.action,
                    confidence=revised.confidence,
                    source="critique_revised",
                    _force_log=True,
                )
                return revised

            if critique.verdict == CritiqueVerdict.PASS:
                decision = apply_ceiling(decision, evidence)
                logger.info(
                    "SELF_CRITIQUE_VERDICT",
                    episode_id=episode.episode_id,
                    verdict="PASS",
                    _force_log=True,
                )
                logger.info(
                    "DECISION_FINALIZED",
                    episode_id=episode.episode_id,
                    action=decision.action,
                    confidence=decision.confidence,
                    source="llm",
                    _force_log=True,
                )

            return decision

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
