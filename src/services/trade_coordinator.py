from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Optional

import structlog

from src.core.events import EventBus
from src.core.models import (
    CandidateTrade,
    ExecutionContext,
    RiskDecision,
    SystemEvent,
)
from src.intelligence.models import PromptContext
from src.models.reasoning import LLMDecision, MarketContext, PortfolioSnapshot
from src.recommendations.store import ConfigurationStore
from src.services.evidence_resolver import EvidenceResolver
from src.services.execution import ExecutionService
from src.services.market_context import MarketContextService
from src.services.portfolio_manager import PortfolioManager
from src.services.risk_manager import RiskManager
from src.services.reasoning_coordinator import ReasoningCoordinator

logger = structlog.get_logger("trade_coordinator")


class TradeCoordinator:
    def __init__(
        self,
        risk_manager: RiskManager,
        execution_svc: ExecutionService,
        portfolio_mgr: PortfolioManager,
        event_bus: EventBus,
        config: dict,
        reasoning_coordinator: ReasoningCoordinator,
        market_context_svc: MarketContextService,
        evidence_resolver: EvidenceResolver,
        config_store: Optional[ConfigurationStore] = None,
        session_id: Optional[str] = None,
    ):
        self._risk = risk_manager
        self._execution = execution_svc
        self._portfolio = portfolio_mgr
        self._event_bus = event_bus
        self._config = config
        self._reasoning_coordinator = reasoning_coordinator
        self._market_context = market_context_svc
        self._evidence_resolver = evidence_resolver
        self._config_store = config_store
        self._session_id = session_id
        self._shadow_enabled = config.get("shadow", {}).get("enabled", True)

        self._event_bus.subscribe("CANDIDATE_DISCOVERED", self._on_candidate_discovered)
        logger.info("TradeCoordinator initialized", shadow_enabled=self._shadow_enabled)

    async def _on_candidate_discovered(self, event: SystemEvent) -> None:
        payload = event.payload
        candidate_data = payload.get("candidate", {})
        candidate = CandidateTrade(**candidate_data)
        correlation_id = payload.get("correlation_id", str(uuid.uuid4()))
        strategy_version = payload.get("strategy_version", "1.0")
        opportunity_id = payload.get("opportunity_id", "")
        timeframe = payload.get("timeframe", "5m")

        quote_filter = self._config.get("universe", {}).get("quote_filter", "USDT")
        if quote_filter != "all":
            symbol_quote = "USDC" if candidate.symbol.endswith("USDC") else "USDT"
            if symbol_quote != quote_filter:
                logger.info(
                    "CANDIDATE_SKIPPED_QUOTE_FILTER",
                    symbol=candidate.symbol,
                    symbol_quote=symbol_quote,
                    quote_filter=quote_filter,
                )
                return

        t0 = time.perf_counter()
        logger.info(
            "CANDIDATE_PROCESSING_STARTED",
            symbol=candidate.symbol,
            proposed_side=candidate.proposed_side,
            anchor_symbol=candidate.anchor_symbol,
            correlation_id=correlation_id,
            opportunity_id=opportunity_id,
        )

        risk_cfg = self._config.get("risk", {})
        max_positions = risk_cfg.get("max_positions", 3)
        min_confidence = risk_cfg.get("min_llm_confidence", 0.3)
        max_exposure = risk_cfg.get("max_live_exposure_usdt", 10000.0)

        market = await self._market_context.get_context(candidate.symbol, timeframe)
        snapshot = await self._portfolio.build_snapshot(
            max_positions=max_positions,
            min_llm_confidence=min_confidence,
            max_live_exposure_usdt=max_exposure,
        )
        evidence = self._evidence_resolver.resolve(candidate)

        logger.info(
            "Calling ReasoningCoordinator",
            symbol=candidate.symbol,
            correlation_id=correlation_id,
        )
        llm_decision = await self._reasoning_coordinator.evaluate_candidate(
            candidate=candidate,
            market=market,
            portfolio=snapshot,
            evidence=evidence,
        )
        logger.info(
            "LLM decision received",
            symbol=candidate.symbol,
            action=llm_decision.action,
            confidence=llm_decision.confidence,
            correlation_id=correlation_id,
        )

        if llm_decision.action == "ABSTAIN":
            logger.info(
                "Trade aborted: LLM abstained",
                symbol=candidate.symbol,
                rationale=llm_decision.rationale,
                correlation_id=correlation_id,
            )
            return

        logger.info(
            "Evaluating candidate with RiskManager",
            symbol=candidate.symbol,
            llm_confidence=llm_decision.confidence,
            correlation_id=correlation_id,
        )
        try:
            risk_decision, risk_reason = await self._risk.evaluate_candidate(
                candidate, self._portfolio, llm_decision.confidence,
            )
        except Exception as e:
            logger.critical(
                "RiskManager.evaluate_candidate threw exception",
                symbol=candidate.symbol,
                error=str(e),
                exc_info=True,
                correlation_id=correlation_id,
            )
            return
        logger.info(
            "RiskManager result",
            symbol=candidate.symbol,
            decision=risk_decision.value,
            reason=risk_reason,
            correlation_id=correlation_id,
        )

        candidate_id = str(uuid.uuid4())
        trade_group_id = str(uuid.uuid4())
        self._log_decision(
            candidate, llm_decision, risk_decision, risk_reason,
            correlation_id, candidate_id,
        )

        eval_event = SystemEvent(
            event_type="CANDIDATE_EVALUATED",
            service_name="TradeCoordinator",
            payload={
                "candidate": candidate_data,
                "risk_decision": risk_decision.value,
                "risk_decision_reason": risk_reason,
                "llm_decision": llm_decision.model_dump(),
                "evidence_source": evidence.evidence_source,
                "evidence_tier": evidence.evidence_tier,
                "correlation_id": correlation_id,
                "candidate_id": candidate_id,
            },
        )
        await self._event_bus.publish(eval_event)

        if risk_decision == RiskDecision.REJECTED_QUALITY:
            logger.info(
                "Candidate discarded (quality)",
                symbol=candidate.symbol,
                reason=risk_reason,
                execution_stage="risk_quality",
            )
            return

        if risk_decision == RiskDecision.DEFERRED:
            logger.info(
                "Candidate deferred",
                symbol=candidate.symbol,
                reason=risk_reason,
                execution_stage="risk_deferred",
            )
            return

        if risk_decision == RiskDecision.REJECTED_CONSTRAINT and not self._shadow_enabled:
            logger.info(
                "Candidate discarded (shadow disabled)",
                symbol=candidate.symbol,
                reason=risk_reason,
                execution_stage="risk_constraint",
            )
            return

        exec_mode = "LIVE" if risk_decision == RiskDecision.APPROVED else "SHADOW"
        origin = "NORMAL" if risk_decision == RiskDecision.APPROVED else "CONSTRAINT"

        active_profile_id: Optional[str] = None
        if self._config_store:
            profile = self._config_store.get_active_profile()
            if profile:
                active_profile_id = profile.profile_id

        context = ExecutionContext(
            correlation_id=correlation_id,
            timeframe=timeframe,
            execution_id=str(uuid.uuid4()),
            trade_group_id=trade_group_id,
            candidate_id=candidate_id,
            strategy_version=strategy_version,
            active_profile_id=active_profile_id,
            execution_mode=exec_mode,
            origin=origin,
            symbol=candidate.symbol,
            side=candidate.proposed_side,
            quantity=candidate.proposed_quantity or 0.0,
            anchor_symbol=candidate.anchor_symbol,
            correlation_score=candidate.correlation_score,
            entry_thesis=(
                f"LLM decision: {llm_decision.action} "
                f"(confidence: {llm_decision.confidence:.2f}) — "
                f"{llm_decision.rationale[:100]}"
            ),
            llm_confidence=llm_decision.confidence,
            risk_decision=risk_decision.value,
            risk_decision_reason=risk_reason,
            opportunity_id=opportunity_id,
            session_id=self._session_id,
            execution_model=self._config.get("execution", {}).get("model", "fixed_friction_v1"),
            execution_model_version=self._config.get("execution", {}).get("model_version", "1.0"),
            execution_parameters=self._config.get("execution", {}).get("parameters", {
                "spread_bps": 2.0,
                "fee_bps": 4.0,
                "slippage_bps": 3.0,
            }),
            entry_timestamp=datetime.utcnow(),
        )

        logger.info(
            "Publishing EXECUTE_TRADE event",
            symbol=candidate.symbol,
            execution_mode=exec_mode,
            origin=origin,
            correlation_id=correlation_id,
            trade_group_id=trade_group_id,
        )
        exec_event = SystemEvent(
            event_type="EXECUTE_TRADE",
            service_name="TradeCoordinator",
            payload={"context": context.model_dump()},
        )
        await self._event_bus.publish(exec_event)

        logger.info(
            "EXECUTE_TRADE_PUBLISHED",
            symbol=candidate.symbol,
            execution_mode=exec_mode,
            origin=origin,
            risk_reason=risk_reason,
            correlation_id=correlation_id,
            trade_group_id=trade_group_id,
            opportunity_id=opportunity_id,
            llm_action=llm_decision.action,
            llm_confidence=llm_decision.confidence,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def _log_decision(
        self,
        candidate: CandidateTrade,
        llm_decision: LLMDecision,
        decision: RiskDecision,
        reason: str,
        correlation_id: str,
        candidate_id: str,
    ) -> None:
        logger.info(
            "Risk decision",
            symbol=candidate.symbol,
            llm_action=llm_decision.action,
            llm_confidence=llm_decision.confidence,
            decision=decision.value,
            reason=reason,
            correlation_id=correlation_id,
            candidate_id=candidate_id,
        )
