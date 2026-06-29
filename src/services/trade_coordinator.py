from __future__ import annotations

import uuid
from datetime import datetime

import structlog

from src.core.events import EventBus
from src.core.models import (
    CandidateTrade,
    ExecutionContext,
    RiskDecision,
    SystemEvent,
)
from src.services.execution import ExecutionService
from src.services.portfolio_manager import PortfolioManager
from src.services.risk_manager import RiskManager

logger = structlog.get_logger("trade_coordinator")


class TradeCoordinator:
    def __init__(
        self,
        risk_manager: RiskManager,
        execution_svc: ExecutionService,
        portfolio_mgr: PortfolioManager,
        event_bus: EventBus,
        config: dict,
    ):
        self._risk = risk_manager
        self._execution = execution_svc
        self._portfolio = portfolio_mgr
        self._event_bus = event_bus
        self._config = config
        self._shadow_enabled = config.get("shadow", {}).get("enabled", True)

        self._event_bus.subscribe("CANDIDATE_DISCOVERED", self._on_candidate_discovered)
        logger.info("TradeCoordinator initialized", shadow_enabled=self._shadow_enabled)

    async def _on_candidate_discovered(self, event: SystemEvent) -> None:
        payload = event.payload
        candidate_data = payload.get("candidate", {})
        candidate = CandidateTrade(**candidate_data)
        llm_confidence = payload.get("llm_confidence", 0.0)
        correlation_id = payload.get("correlation_id", str(uuid.uuid4()))
        strategy_version = payload.get("strategy_version", "1.0")
        llm_request_id = payload.get("llm_request_id")

        risk_decision, risk_reason = await self._risk.evaluate_candidate(
            candidate, self._portfolio, llm_confidence
        )

        candidate_id = str(uuid.uuid4())
        trade_group_id = str(uuid.uuid4())
        self._log_decision(
            candidate, risk_decision, risk_reason, correlation_id, candidate_id
        )

        eval_event = SystemEvent(
            event_type="CANDIDATE_EVALUATED",
            service_name="TradeCoordinator",
            payload={
                "candidate": candidate_data,
                "risk_decision": risk_decision.value,
                "risk_decision_reason": risk_reason,
                "llm_confidence": llm_confidence,
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
            )
            return

        if risk_decision == RiskDecision.DEFERRED:
            logger.info(
                "Candidate deferred",
                symbol=candidate.symbol,
                reason=risk_reason,
            )
            return

        if risk_decision == RiskDecision.REJECTED_CONSTRAINT and not self._shadow_enabled:
            logger.info(
                "Candidate discarded (shadow disabled)",
                symbol=candidate.symbol,
                reason=risk_reason,
            )
            return

        exec_mode = "LIVE" if risk_decision == RiskDecision.APPROVED else "SHADOW"
        origin = "NORMAL" if risk_decision == RiskDecision.APPROVED else "CONSTRAINT"

        context = ExecutionContext(
            correlation_id=correlation_id,
            execution_id=str(uuid.uuid4()),
            trade_group_id=trade_group_id,
            candidate_id=candidate_id,
            strategy_version=strategy_version,
            llm_request_id=llm_request_id,
            execution_mode=exec_mode,
            origin=origin,
            symbol=candidate.symbol,
            side=candidate.proposed_side,
            quantity=candidate.proposed_quantity or 0.0,
            anchor_symbol=candidate.anchor_symbol,
            correlation_score=candidate.correlation_score,
            entry_thesis=(
                f"Scanner signal: {candidate.signal_strength:.2f} confidence "
                f"on {candidate.anchor_symbol}"
            ),
            llm_confidence=llm_confidence,
            risk_decision=risk_decision.value,
            risk_decision_reason=risk_reason,
            execution_model=self._config.get("execution", {}).get("model", "fixed_friction_v1"),
            execution_model_version=self._config.get("execution", {}).get("model_version", "1.0"),
            execution_parameters=self._config.get("execution", {}).get("parameters", {
                "spread_bps": 2.0,
                "fee_bps": 4.0,
                "slippage_bps": 3.0,
            }),
            entry_timestamp=datetime.utcnow(),
        )

        exec_event = SystemEvent(
            event_type="EXECUTE_TRADE",
            service_name="TradeCoordinator",
            payload={"context": context.model_dump()},
        )
        await self._event_bus.publish(exec_event)

        logger.info(
            "Trade routed for execution",
            symbol=candidate.symbol,
            execution_mode=exec_mode,
            origin=origin,
            risk_reason=risk_reason,
            correlation_id=correlation_id,
            trade_group_id=trade_group_id,
        )

    def _log_decision(
        self,
        candidate: CandidateTrade,
        decision: RiskDecision,
        reason: str,
        correlation_id: str,
        candidate_id: str,
    ) -> None:
        logger.info(
            "Risk decision",
            symbol=candidate.symbol,
            decision=decision.value,
            reason=reason,
            correlation_id=correlation_id,
            candidate_id=candidate_id,
        )
