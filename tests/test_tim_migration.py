from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pytest

from src.core.models import Position, PositionState, ProtectionOrders
from src.db.portfolio_store import PORTFOLIO_DB, PortfolioStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_apex.duckdb")
    s = PortfolioStore(db_path)
    s.create_schema()
    yield s
    s.close()


class TestMigrationColumns:
    def test_migration_adds_tim_columns(self, store):
        cols = {
            row[0]
            for row in store._conn.execute("DESCRIBE positions").fetchall()
        }
        assert "trade_memory_id" in cols
        assert "origin_episode_id" in cols

    def test_migration_idempotent(self, store):
        store._apply_migration()
        store._apply_migration()

    def test_column_types(self, store):
        col_info = {
            row[0]: row[1]
            for row in store._conn.execute("DESCRIBE positions").fetchall()
        }
        assert col_info["trade_memory_id"] == "VARCHAR"
        assert col_info["origin_episode_id"] == "VARCHAR"


class TestOldPositionLoad:
    def test_old_position_without_tim_fields_loads_safely(self, store):
        store._conn.execute(
            """
            INSERT INTO positions (
                position_id, symbol, side, quantity, avg_fill_price,
                exchange_order_ids, entry_timestamp, anchor_symbol,
                review_count, lifecycle_state, current_stop, current_target,
                highest_unrealized_profit, maximum_drawdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "old-pos-1", "BTCUSDT", "LONG", 0.5, 50000.0,
                [], datetime(2025, 1, 1, 0, 0, 0), "BTC",
                0, "OPEN", 49000.0, 55000.0, 1000.0, 500.0,
            ],
        )
        pos = store.get_position_by_id("old-pos-1")
        assert pos is not None
        assert pos.position_id == "old-pos-1"
        assert pos.trade_memory_id is None
        assert pos.origin_episode_id is None

    def test_old_position_without_protection_orders(self, store):
        store._conn.execute(
            """
            INSERT INTO positions (
                position_id, symbol, side, quantity, avg_fill_price,
                exchange_order_ids, entry_timestamp, anchor_symbol,
                review_count, lifecycle_state, current_stop, current_target,
                highest_unrealized_profit, maximum_drawdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "old-pos-2", "ETHUSDT", "SHORT", 2.0, 3000.0,
                [], datetime(2025, 1, 1, 0, 0, 0), "ETH",
                0, "OPEN", 3100.0, 2800.0, 200.0, 100.0,
            ],
        )
        pos = store.get_position_by_id("old-pos-2")
        assert pos is not None
        assert pos.protection_orders is not None
        assert pos.protection_orders.authority_mode == "MECHANICAL_ONLY"


class TestProtectionOrdersAuthorityMode:
    def test_old_json_without_authority_mode_defaults(self, store):
        old_json = json.dumps({
            "stop_order_id": "stop-1",
            "tp_order_id": "tp-1",
            "stop_price": 48000.0,
            "tp_price": 55000.0,
        })
        po = ProtectionOrders(**json.loads(old_json))
        assert po.authority_mode == "MECHANICAL_ONLY"

    def test_new_json_with_authority_mode(self, store):
        new_json = json.dumps({
            "stop_order_id": "stop-1",
            "authority_mode": "TIM_SUPERVISED",
        })
        po = ProtectionOrders(**json.loads(new_json))
        assert po.authority_mode == "TIM_SUPERVISED"


class TestSaveLoadRoundtrip:
    def test_save_and_load_with_tim_fields(self, store):
        pos = Position(
            position_id="pos-tim-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            avg_fill_price=50000.0,
            anchor_symbol="BTC",
            trade_memory_id="mem-1",
            origin_episode_id="ep-1",
        )
        store.save_position(pos)
        loaded = store.get_position_by_id("pos-tim-1")
        assert loaded is not None
        assert loaded.trade_memory_id == "mem-1"
        assert loaded.origin_episode_id == "ep-1"

    def test_save_and_load_without_tim_fields(self, store):
        pos = Position(
            position_id="pos-no-tim-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            avg_fill_price=50000.0,
            anchor_symbol="BTC",
        )
        store.save_position(pos)
        loaded = store.get_position_by_id("pos-no-tim-1")
        assert loaded is not None
        assert loaded.trade_memory_id is None
        assert loaded.origin_episode_id is None

    def test_protection_orders_authority_mode_roundtrip(self, store):
        po = ProtectionOrders(
            stop_price=48000.0,
            tp_price=55000.0,
            authority_mode="TIM_SUPERVISED",
        )
        pos = Position(
            position_id="pos-auth-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            avg_fill_price=50000.0,
            anchor_symbol="BTC",
            protection_orders=po,
        )
        store.save_position(pos)
        loaded = store.get_position_by_id("pos-auth-1")
        assert loaded is not None
        assert loaded.protection_orders.authority_mode == "TIM_SUPERVISED"

    def test_protection_orders_default_roundtrip(self, store):
        pos = Position(
            position_id="pos-auth-2",
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            avg_fill_price=50000.0,
            anchor_symbol="BTC",
        )
        store.save_position(pos)
        loaded = store.get_position_by_id("pos-auth-2")
        assert loaded is not None
        assert loaded.protection_orders.authority_mode == "MECHANICAL_ONLY"
