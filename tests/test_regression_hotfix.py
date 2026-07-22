from __future__ import annotations

import math
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import Position, PositionState, ProtectionOrders
from src.services.execution import safe_float


# ── D. safe_float helper behavior ─────────────────────────────────────

class TestSafeFloat:
    def test_nominal_value(self):
        assert safe_float(3.14) == 3.14
        assert safe_float("3.14") == 3.14
        assert safe_float(0) == 0.0
        assert safe_float(-1.5) == -1.5

    def test_none_returns_default(self):
        assert safe_float(None) == 0.0
        assert safe_float(None, 1.0) == 1.0
        assert safe_float(None, -1.0) == -1.0

    def test_missing_key_returns_default(self):
        d = {}
        assert safe_float(d.get("atr"), 0.0) == 0.0
        assert safe_float(d.get("atr"), 42.0) == 42.0

    def test_key_with_none_value(self):
        d = {"atr": None}
        assert safe_float(d.get("atr"), 0.0) == 0.0

    def test_key_with_invalid_string(self):
        assert safe_float("not_a_number") == 0.0
        assert safe_float("") == 0.0
        assert safe_float("NaN", 0.0) == 0.0

    def test_nan_value(self):
        assert safe_float(float("nan")) == 0.0
        assert safe_float(math.nan) == 0.0

    def test_infinity_passes_through(self):
        result = safe_float(float("inf"))
        assert result == float("inf")
        assert math.isinf(result)

    def test_zero_returns_zero(self):
        assert safe_float(0) == 0.0
        assert safe_float(0.0) == 0.0
        assert safe_float("0") == 0.0

    def test_negative_values(self):
        assert safe_float(-100.0) == -100.0
        assert safe_float("-50.5") == -50.5


# ── A. execution.py trailing stop path with atr=None ──────────────────
# This tests that _build_executable_trade does not crash when
# indicators={"atr": None} by mocking up the minimal call path.

class TestBuildExecutableTradeNoneAtr:
    @pytest.fixture
    def exec_service(self):
        """Minimal ExecutionService with mocked dependencies."""
        from src.services.execution import ExecutionService

        mock_client = MagicMock()
        mock_event_bus = AsyncMock()
        mock_context = AsyncMock()
        mock_portfolio = MagicMock()

        mock_virtual = MagicMock()

        service = ExecutionService(
            live_executor=mock_client,
            virtual_executor=mock_virtual,
            event_bus=mock_event_bus,
            market_context=mock_context,
            portfolio_mgr=mock_portfolio,
            config={"execution": {}},
        )
        return service

    def test_atr_none_does_not_crash(self, exec_service):
        """_build_executable_trade must handle indicators={"atr": None}."""
        from src.core.models import ExecutionContext
        from src.models.execution import ExecutionPlan

        context = ExecutionContext(
            correlation_id="corr-1",
            execution_id="exec-1",
            trade_group_id="grp-1",
            candidate_id="cand-1",
            execution_mode="LIVE",
            origin="TEST",
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            anchor_symbol="BTC",
        )
        plan = ExecutionPlan(
            symbol="BTCUSDT",
            side="LONG",
        )

        trade = exec_service._build_executable_trade(
            context=context,
            qty=0.5,
            qty_str="0.50",
            current_price=50000.0,
            trade_side="LONG",
            proxy_stop=49000.0,
            proxy_tp=55000.0,
            exec_cfg={"sizing_value": "2.0", "stop_loss_pct": "0.98", "take_profit_pct": "1.04", "max_risk_pct": "0.02", "slippage_bps": "3.0"},
            exchange_filters={},
            taker_fee_rate=0.0004,
            available_balance=1000.0,
            indicators={"atr": None},  # the regression case
            plan=plan,
        )
        assert trade is not None
        assert trade.atr == 0.0

    def test_atr_missing_does_not_crash(self, exec_service):
        """_build_executable_trade must handle empty indicators dict."""
        from src.core.models import ExecutionContext
        from src.models.execution import ExecutionPlan

        context = ExecutionContext(
            correlation_id="corr-2",
            execution_id="exec-2",
            trade_group_id="grp-2",
            candidate_id="cand-2",
            execution_mode="LIVE",
            origin="TEST",
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            anchor_symbol="BTC",
        )
        plan = ExecutionPlan(
            symbol="BTCUSDT",
            side="LONG",
        )

        trade = exec_service._build_executable_trade(
            context=context,
            qty=0.5,
            qty_str="0.50",
            current_price=50000.0,
            trade_side="LONG",
            proxy_stop=49000.0,
            proxy_tp=55000.0,
            exec_cfg={"sizing_value": "2.0", "stop_loss_pct": "0.98", "take_profit_pct": "1.04", "max_risk_pct": "0.02", "slippage_bps": "3.0"},
            exchange_filters={},
            taker_fee_rate=0.0004,
            available_balance=1000.0,
            indicators={},  # atr key is missing entirely
            plan=plan,
        )
        assert trade is not None
        assert trade.atr == 0.0


# ── E. Position save/load with new TIM fields (already tested) ───────
# Covered by tests/test_tim_migration.py:
#   test_save_and_load_with_tim_fields
#   test_save_and_load_without_tim_fields
#   test_old_position_without_tim_fields_loads_safely

# ── F. ProtectionOrders authority_mode default (already tested) ──────
# Covered by tests/test_tim_migration.py:
#   test_old_json_without_authority_mode_defaults
#   test_new_json_with_authority_mode
#   test_protection_orders_authority_mode_roundtrip
#   test_protection_orders_default_roundtrip
