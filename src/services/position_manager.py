from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone

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

            position = Position(
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
                active_profile_id=payload.get("active_profile_id"),
                session_id=payload.get("session_id"),
            )

            vf_data = payload.get("virtual_fill")
            if vf_data:
                from src.core.models import VirtualFill
                position.virtual_fill = VirtualFill(**vf_data)

            po_data = payload.get("protection_orders")
            if po_data:
                position.protection_orders = ProtectionOrders(**po_data)

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
            )
        except Exception as e:
            logger.error("Failed to create position from entry fill", error=str(e))

    async def _handle_exit_fill(self, payload: dict) -> None:
        try:
            position_id = payload["position_id"]
            reason = payload.get("reason", "manual")

            position = self._portfolio.get_position_by_id(position_id)
            if position is not None:
                position.exit_price = payload.get("exit_price")
                position.exit_fees = payload.get("commission", 0.0)

            await self._portfolio.update_position_state(
                position_id,
                PositionState.CLOSED,
                exit_reason=reason,
            )

            if self._calibration_enabled:
                await self._compute_calibration(position_id, payload)

            logger.info(
                "Position closed from fill",
                position_id=position_id,
                reason=reason,
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

    async def _monitor_single_position(self, pos: Position) -> None:
        state = await self._context.get_state(pos.symbol)
        current_price = state.get("current_price", 0.0)
        if current_price <= 0:
            return

        # ── STEP 1: HARD RISK — SL/TP Check ──
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
                    stop_on_exchange = (
                        pos.protection_orders.stop_client_order_id in order_ids
                    )
                    tp_on_exchange = (
                        pos.protection_orders.tp_client_order_id in order_ids
                    )
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
                await self._execution.update_trailing_stop(pos, candidate_stop)

        # ── STEP 2: MFE/MAE Tracking ──
        if pos.side == "LONG":
            unrealized_pnl = (current_price - pos.avg_fill_price) * pos.quantity
            drawdown = (pos.avg_fill_price - current_price) * pos.quantity
        else:
            unrealized_pnl = (pos.avg_fill_price - current_price) * pos.quantity
            drawdown = (current_price - pos.avg_fill_price) * pos.quantity
        if unrealized_pnl > pos.highest_unrealized_profit:
            pos.highest_unrealized_profit = unrealized_pnl
        if drawdown > pos.maximum_drawdown:
            pos.maximum_drawdown = drawdown

        # ── STEP 3: EVIDENCE EVOLUTION ──
        needs_save = await self._process_evidence(pos, state)

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
