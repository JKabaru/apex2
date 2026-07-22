from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Optional

import structlog

from src.core.models import Position
from src.db.tim_store import TimStore
from src.db.write_coordinator import DatabaseWriteCoordinator
from src.models.tim.enums import JournalEventType, OriginQuality, TIMMode, ThesisStatus
from src.models.tim.review import ReviewConditions
from src.models.tim.trade_memory import TradeJournalEntry, TradeOrigin, WorkingMemory
from src.services.execution import safe_float

logger = structlog.get_logger("tim_bridge")


class TimMemoryBridge:
    def __init__(
        self,
        tim_store: Optional[TimStore],
        coordinator: Optional[DatabaseWriteCoordinator],
        tim_mode: TIMMode,
    ) -> None:
        self._tim_store = tim_store
        self._coordinator = coordinator
        try:
            self._tim_mode = tim_mode if isinstance(tim_mode, TIMMode) else TIMMode(tim_mode)
        except (ValueError, KeyError, TypeError):
            self._tim_mode = TIMMode.OFF

    def is_enabled(self) -> bool:
        return self._tim_store is not None and self._tim_mode != TIMMode.OFF

    async def on_position_filled(
        self,
        position: Position,
        state: Optional[dict] = None,
    ) -> bool:
        if not self.is_enabled():
            return False
        if position.trade_memory_id is not None:
            return False

        memory_id = str(uuid.uuid4())
        origin_episode_id = str(uuid.uuid4())

        origin_quality, conviction, conviction_source, metadata = self._build_origin_metadata(position, state)

        origin = TradeOrigin(
            memory_id=memory_id,
            position_id=position.position_id,
            origin_episode_id=origin_episode_id,
            origin_quality=origin_quality,
            entry_thesis=position.entry_thesis or "",
            entry_price=position.avg_fill_price,
            entry_atr=self._get_entry_atr(position, state),
            entry_timestamp=position.entry_timestamp,
            symbol=position.symbol,
            side=position.side,
            anchor_symbol=position.anchor_symbol or "",
            timeframe=position.timeframe or "",
            metadata=metadata,
        )

        ref_ts = position.entry_timestamp or datetime.utcnow()

        review_conditions = ReviewConditions(
            reference_price=position.avg_fill_price,
            reference_atr=safe_float(origin.entry_atr) if origin.entry_atr is not None else None,
            reference_stop=position.current_stop or position.initial_stop_loss or None,
            reference_target=position.current_target or position.initial_take_profit or None,
            reference_trend=(
                str(state.get("trend_regime")) if state and "trend_regime" in state else None
            ),
            reference_volatility=(
                str(state.get("volatility_regime")) if state and "volatility_regime" in state else None
            ),
            reference_timestamp=ref_ts,
        )

        wm = WorkingMemory(
            memory_id=memory_id,
            position_id=position.position_id,
            version=1,
            thesis_status=ThesisStatus.INTACT,
        )
        wm.metadata["initial_conviction"] = conviction
        wm.metadata["current_conviction"] = conviction
        wm.metadata["conviction_source"] = conviction_source
        wm.metadata["reference_timestamp"] = ref_ts.isoformat() if hasattr(ref_ts, "isoformat") else str(ref_ts)
        wm.next_review_conditions = review_conditions.model_dump_json()
        wm.checksum = self._compute_checksum(wm)

        try:
            if self._coordinator is not None:
                with self._coordinator.exclusive_transaction() as conn:
                    self._tim_store.insert_origin(origin)
                    self._tim_store.upsert_working_memory(wm)
            else:
                self._tim_store.insert_origin(origin)
                self._tim_store.upsert_working_memory(wm)
        except Exception:
            logger.error(
                "TIM_MEMORY_WRITE_FAILED",
                position_id=position.position_id,
                operation="origin_and_working_memory",
                tim_mode=self._tim_mode.value,
                exc_info=True,
            )
            return False

        try:
            j1 = TradeJournalEntry(
                position_id=position.position_id,
                memory_id=memory_id,
                version=1,
                event_type=JournalEventType.ORIGIN_SET,
                event_data=json.dumps({
                    "source": "ENTRY_FILL",
                    "origin_quality": origin_quality.value,
                }),
            )
            j2 = TradeJournalEntry(
                position_id=position.position_id,
                memory_id=memory_id,
                version=2,
                event_type=JournalEventType.WORKING_MEMORY_INITIALIZED,
                event_data=json.dumps({
                    "conviction": conviction,
                    "conviction_source": conviction_source,
                }),
            )
            if self._coordinator is not None:
                with self._coordinator.exclusive_transaction() as conn:
                    self._tim_store.append_journal_entry(j1)
                    self._tim_store.append_journal_entry(j2)
            else:
                self._tim_store.append_journal_entry(j1)
                self._tim_store.append_journal_entry(j2)
        except Exception:
            logger.error(
                "TIM_MEMORY_WRITE_FAILED",
                position_id=position.position_id,
                operation="journal_entries",
                tim_mode=self._tim_mode.value,
                exc_info=True,
            )
            return False

        position.trade_memory_id = memory_id
        position.origin_episode_id = origin_episode_id

        logger.info(
            "TRADE_ORIGIN_CREATED",
            position_id=position.position_id,
            origin_id=memory_id,
            tim_mode=self._tim_mode.value,
            origin_quality=origin_quality.value,
        )
        logger.info(
            "WORKING_MEMORY_INITIALIZED",
            memory_id=memory_id,
            origin_id=memory_id,
            position_id=position.position_id,
            version=1,
        )
        return True

    def _build_origin_metadata(
        self, position: Position, state: Optional[dict],
    ) -> tuple[OriginQuality, float, str, dict[str, Any]]:
        metadata: dict[str, Any] = {}

        if position.trade_context is not None:
            tc = position.trade_context
            metadata["expected_catalyst"] = tc.expected_catalyst or ""
            metadata["invalidation_conditions"] = tc.expected_invalidation or ""
            metadata["expected_horizon_hours"] = tc.expected_holding_horizon_hours
            metadata["core_hypothesis"] = tc.thesis or ""
            metadata["direction"] = tc.direction or ""
            metadata["scanner_name"] = tc.scanner_name or ""
            metadata["strategy_name"] = tc.strategy_name or ""
            metadata["anchor_info"] = {
                "anchor_symbol": tc.anchor_symbol,
                "target_symbol": tc.target_symbol or "",
                "relationship": tc.relationship or "",
            }
        else:
            metadata["expected_catalyst"] = ""
            metadata["invalidation_conditions"] = ""
            metadata["expected_horizon_hours"] = 0.0

        if position.initial_evidence is not None:
            ie = position.initial_evidence
            metadata["entry_trend_regime"] = ie.trend_regime
            metadata["entry_volatility_regime"] = ie.volatility_regime
            metadata["entry_volume_profile"] = ie.volume_profile
            metadata["entry_momentum"] = ie.momentum
            metadata["entry_correlation_regime"] = ie.correlation_regime
            metadata["entry_integrity"] = ie.integrity
            metadata["entry_evidence_source"] = ie.source

        metadata["llm_request_id"] = position.llm_request_id
        metadata["risk_decision"] = position.risk_decision or ""
        metadata["risk_decision_reason"] = position.risk_decision_reason or ""
        metadata["opportunity_id"] = position.opportunity_id or ""
        metadata["strategy_version"] = position.strategy_version or ""
        metadata["execution_model"] = position.execution_model or ""

        if state:
            metadata["entry_market_context"] = {
                "trend_regime": state.get("trend_regime"),
                "volatility_regime": state.get("volatility_regime"),
                "volume_profile": state.get("volume_profile"),
            }

        has_llm = bool(position.llm_request_id)
        has_context = position.trade_context is not None
        has_thesis = bool(position.entry_thesis)

        # Stage 2 limitation: no ReasoningEpisode verification available.
        # HIGH quality requires full episode verification — deferred to Stage 3.
        # llm_request_id existence alone is insufficient for HIGH.
        if has_llm and (has_context or has_thesis):
            quality = OriginQuality.MEDIUM
            conviction = 0.3
            conviction_source = "THESIS_ONLY"
        elif has_llm:
            quality = OriginQuality.MEDIUM
            conviction = 0.0
            conviction_source = "UNKNOWN"
        elif has_context or has_thesis:
            quality = OriginQuality.MEDIUM
            conviction = 0.3
            conviction_source = "THESIS_ONLY"
        else:
            quality = OriginQuality.LOW
            conviction = 0.0
            conviction_source = "SYNTHETIC"

        metadata["initial_conviction"] = conviction
        metadata["conviction_source"] = conviction_source

        return quality, conviction, conviction_source, metadata

    def _get_entry_atr(self, position: Position, state: Optional[dict]) -> Optional[float]:
        if position.initial_evidence is not None and position.initial_evidence.atr is not None:
            return safe_float(position.initial_evidence.atr, None)
        if state and "indicators" in state:
            return safe_float(state["indicators"].get("atr"), None)
        return None

    def _compute_checksum(self, wm: WorkingMemory) -> str:
        import hashlib
        import json

        data = {
            "memory_id": wm.memory_id,
            "position_id": wm.position_id,
            "version": wm.version,
            "thesis_status": wm.thesis_status.value if wm.thesis_status else "",
            "protection_mode": wm.protection_mode.value if wm.protection_mode else "",
            "review_count": wm.review_count,
            "failed_review_count": wm.failed_review_count,
            "metadata": wm.metadata,
        }
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
