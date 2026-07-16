from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone

from typing import Any

import structlog

from src.core.events import EventBus
from src.core.models import (
    Difference,
    EvidenceEpisode,
    InitialEvidence,
    MarketEvidence,
    Position,
    PositionState,
    ProtectionOrders,
    SystemEvent,
    TradeContext,
)
from src.services.execution import ExecutionService
from src.services.llm_scheduler import LLMScheduler
from src.services.market_context import MarketContextService
from src.services.portfolio_manager import PortfolioManager

logger = structlog.get_logger("position_manager")

MONITOR_INTERVAL = 5
HEARTBEAT_INTERVAL = 300  # 5 min
PROTECTION_AUDIT_INTERVAL = 60  # 1 min
INTERIM_LEARNING_INTERVAL = 900  # 15 min — throttle for periodic interim snapshots
INTERIM_MFE_MAE_ATR_THRESHOLD = 0.5  # min ATR multiple change to trigger on MFE/MAE

TRACKED_STATE_FIELDS = [
    ("price", lambda v: f"{v:.2f}"),
    ("rsi", lambda v: f"{v:.1f}"),
    ("macd_histogram", lambda v: f"{v:.4f}"),
    ("atr", lambda v: f"{v:.4f}"),
    ("trend_regime", str),
    ("momentum", str),
    ("volatility_regime", str),
    ("volume_profile", str),
    ("correlation_regime", str),
    ("correlation_score", lambda v: f"{v:.2f}"),
]


class PositionManager:
    def __init__(
        self,
        portfolio: PortfolioManager,
        execution_svc: ExecutionService,
        llm_scheduler: LLMScheduler,
        event_bus: EventBus,
        context: MarketContextService,
        calibration_enabled: bool = True,
    ):
        self._portfolio = portfolio
        self._execution = execution_svc
        self._llm = llm_scheduler
        self._event_bus = event_bus
        self._context = context
        self._calibration_enabled = calibration_enabled
        self._last_save_time: dict[str, float] = {}
        self._last_protection_audit_time: dict[str, float] = {}
        self._last_interim_time: dict[str, float] = {}
        self._last_mfe_value: dict[str, float] = {}
        self._last_mae_value: dict[str, float] = {}
        self._event_bus.subscribe("ORDER_FILLED", self._on_order_filled)
        logger.info("PositionManager initialized", calibration_enabled=calibration_enabled)

    # ── Event Handlers ──

    async def _on_order_filled(self, event: SystemEvent) -> None:
        payload = event.payload
        fill_type = payload.get("type")
        if fill_type == "entry":
            await self._handle_entry_fill(payload)
        elif fill_type == "exit":
            await self._handle_exit_fill(payload)

    async def _handle_entry_fill(self, payload: dict) -> None:
        try:
            entry_ts = payload.get("entry_timestamp")
            if entry_ts and isinstance(entry_ts, str):
                try:
                    entry_ts = datetime.fromisoformat(entry_ts)
                except (ValueError, TypeError):
                    entry_ts = datetime.utcnow()
            else:
                entry_ts = datetime.utcnow()

            exec_id = payload.get("execution_id")
            pos_kwargs = dict(
                symbol=payload["symbol"],
                side=payload["side"],
                quantity=payload["executed_qty"],
                avg_fill_price=payload["avg_price"],
                fees=payload.get("commission", 0.0),
                exchange_order_ids=[str(payload.get("order_id", ""))],
                anchor_symbol=payload.get("anchor_symbol", ""),
                correlation_score=payload.get("correlation_score", 0.0),
                initial_stop_loss=payload.get("initial_stop_loss", 0.0),
                initial_take_profit=payload.get("initial_take_profit", 0.0),
                current_stop=payload.get("initial_stop_loss", 0.0),
                current_target=payload.get("initial_take_profit", 0.0),
                entry_thesis=payload.get("entry_thesis", ""),
                lifecycle_state=PositionState.OPEN,
                entry_timestamp=entry_ts,
                execution_mode=payload.get("execution_mode", "LIVE"),
                origin=payload.get("origin", "NORMAL"),
                execution_id=payload.get("execution_id"),
                trade_group_id=payload.get("trade_group_id"),
                candidate_id=payload.get("candidate_id"),
                correlation_id=payload.get("correlation_id"),
                llm_request_id=payload.get("llm_request_id"),
                strategy_version=payload.get("strategy_version", "1.0"),
                execution_model=payload.get("execution_model", "fixed_friction_v1"),
                execution_model_version=payload.get("execution_model_version", "1.0"),
                execution_parameters=payload.get("execution_parameters", {}),
                risk_decision=payload.get("risk_decision", ""),
                risk_decision_reason=payload.get("risk_decision_reason", ""),
                created_by=payload.get("created_by", "SCANNER"),
                opportunity_source=payload.get("opportunity_source", "SCANNER"),
                opportunity_id=payload.get("opportunity_id", ""),
                timeframe=payload.get("timeframe", "5m"),
                max_holding_period_minutes=payload.get("max_holding_period_minutes", 0.0),
                active_profile_id=payload.get("active_profile_id"),
                session_id=payload.get("session_id"),
            )
            if exec_id:
                pos_kwargs["position_id"] = exec_id
            position = Position(**pos_kwargs)

            vf_data = payload.get("virtual_fill")
            if vf_data:
                from src.core.models import VirtualFill
                position.virtual_fill = VirtualFill(**vf_data)

            po_data = payload.get("protection_orders")
            if po_data:
                position.protection_orders = ProtectionOrders(**po_data)

            existing_positions = self._portfolio.get_open_positions()
            dup = next((p for p in existing_positions if p.symbol == position.symbol), None)
            if dup is not None:
                logger.warning(
                    "Duplicate position detected for symbol — removing existing",
                    existing_position_id=dup.position_id,
                    existing_mode=dup.execution_mode,
                    existing_state=dup.lifecycle_state.value,
                    new_position_id=position.position_id,
                    symbol=position.symbol,
                )
                # Remove the old position from the store (CLOSED) and in-memory
                dup_previous_state = dup.lifecycle_state
                dup.lifecycle_state = PositionState.CLOSED
                dup.exit_timestamp = datetime.utcnow()
                dup_exit_reason = f"REPLACED_BY_NEW_ENTRY_{position.position_id[:8]}"
                self._portfolio._store.save_position(dup)
                self._portfolio._positions.pop(dup.position_id, None)
                await self._portfolio._event_bus.publish(SystemEvent(
                    event_type="POSITION_UPDATED",
                    service_name="PositionManager",
                    payload={
                        "position_id": dup.position_id,
                        "symbol": position.symbol,
                        "previous_state": dup_previous_state.value,
                        "new_state": PositionState.CLOSED.value,
                        "execution_mode": dup.execution_mode,
                        "exit_reason": dup_exit_reason,
                    },
                ))

            state = await self._context.get_state(position.symbol)
            self._capture_initial_evidence(position, state, payload, source="original", integrity="HIGH")

            await self._portfolio.add_position(position)
            logger.info(
                "Position opened from fill",
                position_id=position.position_id,
                symbol=position.symbol,
                side=position.side,
                execution_mode=position.execution_mode,
                origin=position.origin,
                max_holding_period_minutes=position.max_holding_period_minutes,
                entry_timestamp=position.entry_timestamp.isoformat(),
            )
        except Exception as e:
            logger.error("Failed to create position from entry fill", error=str(e))

    async def _handle_exit_fill(self, payload: dict) -> None:
        try:
            position_id = payload["position_id"]
            reason = payload.get("reason", "manual")

            position = self._portfolio.get_position_by_id(position_id)
            if position is None:
                logger.error(
                    "Position not found for exit fill",
                    position_id=position_id,
                    symbol=payload.get("symbol"),
                )
                return

            if position.lifecycle_state == PositionState.CLOSED:
                logger.debug(
                    "Exit fill ignored — position already closed",
                    position_id=position_id,
                )
                return

            position.exit_price = payload.get("exit_price")
            position.exit_fees = payload.get("commission", 0.0)

            if position.lifecycle_state not in (PositionState.CLOSING, PositionState.CLOSED):
                if position.lifecycle_state in (PositionState.OPEN, PositionState.UNMANAGED_ADOPTED, PositionState.UNDER_REVIEW):
                    await self._portfolio.update_position_state(
                        position_id, PositionState.CLOSING,
                    )

            await self._portfolio.update_position_state(
                position_id,
                PositionState.CLOSED,
                exit_reason=reason,
            )

            # Close sibling positions for the same symbol (e.g. an adopted
            # duplicate left over from reconciler). This ensures any orphaned
            # position is cleaned up and its timeline can be closed.
            symbol = payload.get("symbol", "")
            if symbol:
                sibling_positions = [
                    p for p in self._portfolio.get_open_positions()
                    if p.symbol == symbol and p.position_id != position_id
                ]
                for sibling in sibling_positions:
                    logger.warning(
                        "Closing sibling position for same symbol",
                        sibling_id=sibling.position_id,
                        symbol=symbol,
                        reason=reason,
                    )
                    await self._portfolio.update_position_state(
                        sibling.position_id,
                        PositionState.CLOSED,
                        exit_reason=f"SIBLING_CLOSED_{reason}",
                    )

            if self._calibration_enabled:
                await self._compute_calibration(position_id, payload)

            logger.info(
                "POSITION_CLOSED_FROM_FILL",
                position_id=position_id,
                reason=reason,
                symbol=payload.get("symbol"),
                _force_log=True,
            )
        except Exception as e:
            logger.error("Failed to close position from exit fill", error=str(e))

    # ── Evidence Capture ──

    def _capture_initial_evidence(
        self, position: Position, state: dict, payload: dict,
        source: str = "original", integrity: str = "HIGH",
    ) -> None:
        indicators = state.get("indicators", {})
        position.initial_evidence = InitialEvidence(
            price=position.avg_fill_price,
            rsi=indicators.get("rsi"),
            macd_histogram=indicators.get("histogram"),
            atr=indicators.get("atr"),
            trend_regime=state.get("trend_regime", "UNKNOWN"),
            volume_profile=state.get("volume_profile", "UNKNOWN"),
            volatility_regime=state.get("volatility_regime", "UNKNOWN"),
            momentum=state.get("momentum", "UNKNOWN"),
            correlation_regime=state.get("correlation_regime", "UNKNOWN"),
            correlation_score=position.correlation_score,
            entry_timestamp=position.entry_timestamp,
            integrity=integrity,
            source=source,
        )
        position.trade_context = TradeContext(
            anchor_symbol=position.anchor_symbol,
            target_symbol=position.symbol,
            direction=position.side,
            thesis=position.entry_thesis,
        )
        logger.info(
            "initial_evidence_captured",
            position_id=position.position_id,
            symbol=position.symbol,
            source=source,
            trend_regime=position.initial_evidence.trend_regime,
            correlation_regime=position.initial_evidence.correlation_regime,
        )

    def _build_evidence_from_state(self, state: dict, episode_id: str) -> MarketEvidence:
        return MarketEvidence(
            episode_id=episode_id,
            timestamp=datetime.utcnow(),
            price=state["current_price"],
            rsi=state.get("indicators", {}).get("rsi"),
            macd_histogram=state.get("indicators", {}).get("histogram"),
            atr=state.get("indicators", {}).get("atr"),
            trend_regime=state.get("trend_regime", "UNKNOWN"),
            volume_profile=state.get("volume_profile", "UNKNOWN"),
            volatility_regime=state.get("volatility_regime", "UNKNOWN"),
            momentum=state.get("momentum", "UNKNOWN"),
            correlation_regime=state.get("correlation_regime", "UNKNOWN"),
            correlation_score=state.get("correlation_score", 0.0),
        )

    def _categorical_profile(self, evidence: MarketEvidence | InitialEvidence) -> str:
        return (
            f"{evidence.trend_regime}|{evidence.momentum}|"
            f"{evidence.volatility_regime}|{evidence.volume_profile}|"
            f"{evidence.correlation_regime}"
        )

    def _compute_differences(
        self, prev: MarketEvidence | InitialEvidence | None, curr: MarketEvidence,
    ) -> list[Difference]:
        if prev is None:
            return []
        diffs = []
        for field_name, _ in TRACKED_STATE_FIELDS:
            pv = getattr(prev, field_name, None)
            cv = getattr(curr, field_name, None)
            if pv != cv:
                diffs.append(Difference(field=field_name, previous=pv, current=cv))
        return diffs

    def _generate_episode_summary(self, diffs: list[Difference]) -> str:
        if not diffs:
            return "State stable"
        return " | ".join(
            f"{d.field}: {d.previous} -> {d.current}"
            for d in diffs
        )

    # ── Calibration ──

    def _find_live_sibling(self, trade_group_id: str) -> Position | None:
        matches = [
            p for p in self._portfolio._positions.values()
            if p.trade_group_id == trade_group_id
            and p.execution_mode == "LIVE"
        ]
        return matches[0] if matches else None

    def _find_mirror_sibling(self, trade_group_id: str) -> Position | None:
        matches = [
            p for p in self._portfolio._positions.values()
            if p.trade_group_id == trade_group_id
            and p.execution_mode == "SHADOW"
            and p.origin == "MIRROR"
        ]
        return matches[0] if matches else None

    async def _compute_calibration(self, closed_position_id: str, exit_payload: dict) -> None:
        closed_pos = self._portfolio.get_position_by_id(closed_position_id)
        if closed_pos is None:
            return

        trade_group_id = closed_pos.trade_group_id
        if not trade_group_id:
            return

        # Determine which role this position plays in the mirror pair
        if closed_pos.origin == "MIRROR" and closed_pos.execution_mode == "SHADOW":
            shadow_pos = closed_pos
            live_pos = self._find_live_sibling(trade_group_id)
            if live_pos is None or live_pos.lifecycle_state != PositionState.CLOSED:
                return
        elif closed_pos.origin == "NORMAL" and closed_pos.execution_mode == "LIVE":
            live_pos = closed_pos
            shadow_pos = self._find_mirror_sibling(trade_group_id)
            if shadow_pos is None or shadow_pos.lifecycle_state != PositionState.CLOSED:
                return
        else:
            return  # Not part of a mirror pair

        if live_pos.avg_fill_price <= 0:
            return

        shadow_entry = shadow_pos.avg_fill_price
        shadow_exit = exit_payload.get("exit_price", 0.0)
        shadow_fees = exit_payload.get("commission", 0.0) + shadow_pos.fees
        live_entry = live_pos.avg_fill_price
        live_exit = exit_payload.get("exit_price", 0.0)
        live_fees = live_pos.fees + exit_payload.get("commission", 0.0)

        direction = 1.0 if shadow_pos.side == "LONG" else -1.0
        shadow_pnl = (shadow_exit - shadow_entry) * direction * shadow_pos.quantity - shadow_fees
        live_pnl = (live_exit - live_entry) * direction * live_pos.quantity - live_fees

        calibration_data = {
            "entry_error_bps": round((shadow_entry - live_entry) / live_entry * 10000, 2) if live_entry else 0.0,
            "exit_error_bps": round((shadow_exit - live_exit) / live_exit * 10000, 2) if live_exit else 0.0,
            "fee_error_usdt": round(shadow_fees - live_fees, 4),
            "return_error_usdt": round(shadow_pnl - live_pnl, 4),
            "live_entry": live_entry,
            "shadow_entry": shadow_entry,
            "live_exit": live_exit,
            "shadow_exit": shadow_exit,
            "live_fees": live_fees,
            "shadow_fees": shadow_fees,
            "live_pnl": round(live_pnl, 4),
            "shadow_pnl": round(shadow_pnl, 4),
        }

        shadow_pos.calibration_model = "entry_exit_v1"
        shadow_pos.calibration_version = "1.0"
        shadow_pos.calibration_data = calibration_data
        self._portfolio._store.save_position(shadow_pos)

        live_pos.calibration_model = "entry_exit_v1"
        live_pos.calibration_version = "1.0"
        live_pos.calibration_data = calibration_data
        self._portfolio._store.save_position(live_pos)

        calib_event = SystemEvent(
            event_type="CALIBRATION_COMPUTED",
            service_name="PositionManager",
            payload={
                "live_position_id": live_pos.position_id,
                "mirror_position_id": shadow_pos.position_id,
                "trade_group_id": trade_group_id,
                "calibration_data": calibration_data,
                "calibration_model": "entry_exit_v1",
                "calibration_version": "1.0",
            },
        )
        await self._event_bus.publish(calib_event)

        logger.info(
            "Calibration computed for mirror pair",
            trade_group_id=trade_group_id,
            return_error_usdt=round(shadow_pnl - live_pnl, 4),
        )

    # ── Monitor Loop ──

    async def monitor_positions(self) -> None:
        logger.info("Position monitor loop started")
        while True:
            try:
                await asyncio.sleep(MONITOR_INTERVAL)
                open_positions = self._portfolio.get_open_positions()

                for pos in open_positions:
                    try:
                        # State guard: skip if already under review to prevent state machine spam
                        if pos.lifecycle_state == PositionState.UNDER_REVIEW:
                            logger.debug(
                                "Position already under review, skipping monitor cycle",
                                position_id=pos.position_id,
                                symbol=pos.symbol,
                            )
                            continue
                        await self._monitor_single_position(pos)
                    except Exception as e:
                        logger.error(
                            "Monitor check error for position",
                            position_id=pos.position_id,
                            symbol=pos.symbol,
                            error=str(e),
                        )

            except asyncio.CancelledError:
                logger.info("Position monitor loop cancelled")
                break
            except Exception as e:
                logger.error("Position monitor loop error", error=str(e))
                await asyncio.sleep(1)

    def _emit_observation(
        self, category: str, importance: float, symbol: str,
        data: dict, position_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "source": "position_manager",
            "category": category,
            "importance": importance,
            "symbol": symbol,
            "data": data,
        }
        if position_id:
            payload["context"] = {"position_id": position_id}
        self._event_bus.publish_nowait(SystemEvent(
            event_type="OBSERVATION_EMITTED",
            service_name="PositionManager",
            payload=payload,
        ))

    async def _trigger_interim_learning(self, pos: Position, trigger: str) -> None:
        self._event_bus.publish_nowait(SystemEvent(
            event_type="LEARNING_INTERIM",
            service_name="PositionManager",
            payload={
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "trigger": trigger,
                "execution_mode": pos.execution_mode,
                "lifecycle_state": pos.lifecycle_state.value,
                "entry_timestamp": pos.entry_timestamp.isoformat(),
            },
        ))

    async def _monitor_single_position(self, pos: Position) -> None:
        # ── STEP 0: Time-based exit for SHADOW/MIRROR positions (no price needed) ──
        if pos.execution_mode != "LIVE" and pos.max_holding_period_minutes > 0:
            elapsed_minutes = (datetime.utcnow() - pos.entry_timestamp).total_seconds() / 60.0
            if elapsed_minutes >= pos.max_holding_period_minutes:
                logger.info(
                    "TIME_BASED_EXIT",
                    position_id=pos.position_id,
                    symbol=pos.symbol,
                    elapsed_minutes=round(elapsed_minutes, 1),
                    max_holding_period_minutes=pos.max_holding_period_minutes,
                )
                await self._portfolio.update_position_state(
                    pos.position_id, PositionState.CLOSING
                )
                self._emit_observation(
                    "position", 0.70, pos.symbol,
                    {"event": "time_based_exit", "elapsed_minutes": elapsed_minutes,
                     "max_holding_period_minutes": pos.max_holding_period_minutes},
                    pos.position_id,
                )
                await self._execution.execute_exit(pos, "TIME_BASED_EXIT")
                return

        # ── STEP 1: HARD RISK — SL/TP Check ──
        state = await self._context.get_state(pos.symbol)
        current_price = state.get("current_price", 0.0)
        if current_price <= 0:
            return

        stop_price = pos.protection_orders.stop_price
        tp_price = pos.protection_orders.tp_price
        sl_hit = False
        tp_hit = False
        if pos.side == "LONG":
            if stop_price > 0 and current_price <= stop_price:
                sl_hit = True
            if tp_price > 0 and current_price >= tp_price:
                tp_hit = True
        else:
            if stop_price > 0 and current_price >= stop_price:
                sl_hit = True
            if tp_price > 0 and current_price <= tp_price:
                tp_hit = True

        if sl_hit:
            logger.info(
                "Stop loss hit",
                position_id=pos.position_id,
                symbol=pos.symbol,
                side=pos.side,
                price=current_price,
                stop=pos.current_stop,
            )
            await self._portfolio.update_position_state(
                pos.position_id, PositionState.CLOSING
            )
            self._emit_observation(
                "position", 0.85, pos.symbol,
                {"event": "stop_loss_hit", "price": current_price, "stop": stop_price},
                pos.position_id,
            )
            await self._execution.execute_exit(pos, "SL_HIT")
            return

        if tp_hit:
            logger.info(
                "Take profit hit",
                position_id=pos.position_id,
                symbol=pos.symbol,
                side=pos.side,
                price=current_price,
                target=tp_price,
            )
            await self._portfolio.update_position_state(
                pos.position_id, PositionState.CLOSING
            )
            self._emit_observation(
                "position", 0.80, pos.symbol,
                {"event": "take_profit_hit", "price": current_price, "target": tp_price},
                pos.position_id,
            )
            await self._execution.execute_exit(pos, "TP_HIT")
            return

        # ── STEP 1.2: Continuous Protection Audit (throttled to 60s/position) ──
        now_ts = time.time()
        last_audit = self._last_protection_audit_time.get(pos.position_id, 0)
        if now_ts - last_audit >= float(
            self._execution._config.get("protection", {}).get("audit_interval_seconds", 60.0)
        ):
            self._last_protection_audit_time[pos.position_id] = now_ts
            try:
                if pos.execution_mode == "LIVE" and pos.protection_orders.status not in ("REMOVED", "FAILED"):
                    order_ids = await self._execution.get_open_protection_ids(pos.symbol)
                    stop_refs = {
                        str(v) for v in (
                            pos.protection_orders.stop_client_order_id,
                            pos.protection_orders.stop_order_id,
                        ) if v
                    }
                    tp_refs = {
                        str(v) for v in (
                            pos.protection_orders.tp_client_order_id,
                            pos.protection_orders.tp_order_id,
                        ) if v
                    }
                    stop_on_exchange = bool(stop_refs & order_ids)
                    tp_on_exchange = bool(tp_refs & order_ids)
                    expected_stop = pos.protection_orders.stop_client_order_id is not None
                    expected_tp = pos.protection_orders.tp_client_order_id is not None
                    if (expected_stop and not stop_on_exchange) or (expected_tp and not tp_on_exchange):
                        logger.warning(
                            "PROTECTION_AUDIT_MISMATCH",
                            position_id=pos.position_id,
                            symbol=pos.symbol,
                            stop_on_exchange=stop_on_exchange,
                            tp_on_exchange=tp_on_exchange,
                            expected_stop=expected_stop,
                            expected_tp=expected_tp,
                        )
                        repair_event = SystemEvent(
                            event_type="PROTECTION_REPAIR_REQUESTED",
                            service_name="PositionManager",
                            payload={
                                "position_id": pos.position_id,
                                "symbol": pos.symbol,
                                "side": pos.side,
                                "stop_price": pos.protection_orders.stop_price,
                                "tp_price": pos.protection_orders.tp_price,
                                "quantity": pos.quantity,
                                "execution_id": pos.position_id,
                            },
                        )
                        self._event_bus.publish_nowait(repair_event)
                        audit_event = SystemEvent(
                            event_type="PROTECTION_LOST",
                            service_name="PositionManager",
                            payload={
                                "position_id": pos.position_id,
                                "symbol": pos.symbol,
                                "stop_on_exchange": stop_on_exchange,
                                "tp_on_exchange": tp_on_exchange,
                                "reason": "protection_audit_mismatch",
                            },
                        )
                        self._emit_observation(
                            "risk", 0.75, pos.symbol,
                            {"event": "protection_lost", "reason": "audit_mismatch",
                             "stop_on_exchange": stop_on_exchange, "tp_on_exchange": tp_on_exchange},
                            pos.position_id,
                        )
                        self._event_bus.publish_nowait(audit_event)
                    else:
                        audit_event = SystemEvent(
                            event_type="PROTECTION_VERIFIED",
                            service_name="PositionManager",
                            payload={
                                "position_id": pos.position_id,
                                "symbol": pos.symbol,
                                "status": "active",
                            },
                        )
                        self._event_bus.publish_nowait(audit_event)
            except Exception as e:
                logger.error(
                    "PROTECTION_AUDIT_ERROR",
                    position_id=pos.position_id,
                    symbol=pos.symbol,
                    error=str(e),
                )

        # ── STEP 1.5: Trailing Stop Update ──
        if pos.execution_mode == "LIVE" and pos.protection_orders.status == "PENDING":
            logger.debug(
                "Protection PENDING — skipping trailing stop update",
                position_id=pos.position_id, symbol=pos.symbol,
            )
        else:
            indicators = state.get("indicators", {})
            atr = indicators.get("atr")
            if atr and atr > 0 and stop_price > 0:
                atr_multiplier = 2.0
                atr_offset = atr * atr_multiplier
                if pos.side == "LONG":
                    candidate_stop = current_price - atr_offset
                else:
                    candidate_stop = current_price + atr_offset

                if pos.side == "LONG":
                    is_improvement = candidate_stop > stop_price
                else:
                    is_improvement = candidate_stop < stop_price

                if is_improvement:
                    logger.info(
                        "Trailing stop candidate",
                        position_id=pos.position_id,
                        symbol=pos.symbol,
                        old_stop=stop_price,
                        new_stop=candidate_stop,
                        atr=atr,
                    )
                    self._emit_observation(
                        "position", 0.55, pos.symbol,
                        {"event": "trailing_stop_update", "old_stop": stop_price,
                         "new_stop": candidate_stop, "atr": atr},
                        pos.position_id,
                    )
                    await self._execution.update_trailing_stop(pos, candidate_stop)

        # ── STEP 2: MFE/MAE Tracking ──
        if pos.side == "LONG":
            unrealized_pnl = (current_price - pos.avg_fill_price) * pos.quantity
            drawdown = (pos.avg_fill_price - current_price) * pos.quantity
        else:
            unrealized_pnl = (pos.avg_fill_price - current_price) * pos.quantity
            drawdown = (current_price - pos.avg_fill_price) * pos.quantity
        indicators = state.get("indicators", {})
        current_atr = indicators.get("atr")
        mfe_changed = False
        mae_changed = False
        if unrealized_pnl > pos.highest_unrealized_profit:
            prev_mfe = pos.highest_unrealized_profit
            pos.highest_unrealized_profit = unrealized_pnl
            self._emit_observation(
                "position", 0.50, pos.symbol,
                {"event": "mfe_update", "mfe": unrealized_pnl, "price": current_price},
                pos.position_id,
            )
            if prev_mfe > 0 and current_atr and current_atr > 0:
                mfe_delta_atr = (unrealized_pnl - prev_mfe) / (current_atr * pos.quantity)
                mfe_changed = mfe_delta_atr >= INTERIM_MFE_MAE_ATR_THRESHOLD
        if drawdown > pos.maximum_drawdown:
            prev_mae = pos.maximum_drawdown
            pos.maximum_drawdown = drawdown
            if drawdown > pos.maximum_drawdown * 0.5:
                self._emit_observation(
                    "risk", 0.60, pos.symbol,
                    {"event": "drawdown_increased", "drawdown": drawdown, "price": current_price},
                    pos.position_id,
                )
            if prev_mae > 0 and current_atr and current_atr > 0:
                mae_delta_atr = (drawdown - prev_mae) / (current_atr * pos.quantity)
                mae_changed = mae_delta_atr >= INTERIM_MFE_MAE_ATR_THRESHOLD

        # ── STEP 3: EVIDENCE EVOLUTION ──
        needs_save = await self._process_evidence(pos, state)

        # ── STEP 3a: INTERIM LEARNING TRIGGER ──
        now_ts = time.time()
        last_interim = self._last_interim_time.get(pos.position_id, 0)
        periodic_due = (now_ts - last_interim) >= INTERIM_LEARNING_INTERVAL
        if periodic_due or mfe_changed or mae_changed:
            self._last_interim_time[pos.position_id] = now_ts
            if mfe_changed:
                await self._trigger_interim_learning(pos, "mfe_milestone")
            elif mae_changed:
                await self._trigger_interim_learning(pos, "mae_milestone")
            elif periodic_due:
                await self._trigger_interim_learning(pos, "periodic")

        # ── STEP 4: LLM REVIEW (DISABLED — Phase 4.3 hook) ──
        # When enabled, add these guards BEFORE review logic:
        # 1. State guard: if pos.lifecycle_state == PositionState.UNDER_REVIEW: continue
        # 2. Empty response guard + JSONDecodeError handling
        pass  # interface preserved, implementation deferred

        # ── STEP 5: THROTTLED PERSISTENCE ──
        now = time.time()
        last_save = self._last_save_time.get(pos.position_id, 0)
        if needs_save or (now - last_save) >= HEARTBEAT_INTERVAL:
            self._portfolio._store.save_position(pos)
            self._last_save_time[pos.position_id] = now

    async def _process_evidence(self, pos: Position, state: dict) -> bool:
        # 3a: Ensure initial evidence exists (lazy capture for adopted/recovered)
        if pos.initial_evidence is None:
            self._capture_initial_evidence(
                pos, state, {}, source="reconstructed", integrity="LOW",
            )

        # 3b: Build current evidence snapshot
        episode_id = pos.evidence_episodes[-1].episode_id if pos.evidence_episodes else self._new_episode_id()
        current_evidence = self._build_evidence_from_state(state, episode_id)

        # 3c: Check if categorical state changed
        current_profile = self._categorical_profile(current_evidence)
        last_profile = (
            self._categorical_profile(pos.current_evidence)
            if pos.current_evidence
            else self._categorical_profile(pos.initial_evidence)
        )

        if current_profile == last_profile and pos.current_evidence is not None:
            # No categorical change — no evidence generated
            return False

        # 3d: Compute diffs
        current_evidence.drift_from_entry = self._compute_differences(
            pos.initial_evidence, current_evidence,
        )
        current_evidence.change_since_last_cycle = self._compute_differences(
            pos.current_evidence or pos.initial_evidence, current_evidence,
        )

        # 3e: Start new episode if profile changed
        if not pos.evidence_episodes or current_profile != (
            pos.evidence_episodes[-1].state_profile if pos.evidence_episodes else ""
        ):
            episode_id = self._new_episode_id()
            current_evidence.episode_id = episode_id
            episode = EvidenceEpisode(
                episode_id=episode_id,
                index=len(pos.evidence_episodes),
                started_at=datetime.utcnow(),
                state_profile=current_profile,
                summary=self._generate_episode_summary(current_evidence.change_since_last_cycle),
                evidence=[current_evidence],
            )
            pos.evidence_episodes.append(episode)
            await self._trigger_interim_learning(pos, "evidence_episode_transition")
        else:
            pos.evidence_episodes[-1].evidence.append(current_evidence)
            pos.evidence_episodes[-1].ended_at = datetime.utcnow()
            if current_evidence.change_since_last_cycle:
                pos.evidence_episodes[-1].summary = self._generate_episode_summary(
                    current_evidence.change_since_last_cycle,
                )

        # 3f: Update current evidence cache
        pos.current_evidence = current_evidence

        logger.info(
            "evidence_observation",
            position_id=pos.position_id,
            symbol=pos.symbol,
            episode_id=episode_id,
            drift_count=len(current_evidence.drift_from_entry),
            change_count=len(current_evidence.change_since_last_cycle),
        )
        return True

    @staticmethod
    def _new_episode_id() -> str:
        return f"ep_{uuid.uuid4().hex[:12]}"
