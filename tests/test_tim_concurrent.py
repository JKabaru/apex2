from __future__ import annotations

import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pytest

from src.db.portfolio_store import PortfolioStore
from src.db.tim_store import TimStore
from src.models.tim.enums import TIMMode, OriginQuality
from src.models.tim.trade_memory import TradeOrigin, WorkingMemory

DB_LOCK = threading.Lock()


@pytest.fixture
def db_path():
    path = os.path.join(tempfile.gettempdir(), f"tim_concurrent_{uuid.uuid4().hex}.duckdb")
    yield path
    if os.path.exists(path):
        try:
            os.unlink(path)
        except PermissionError:
            pass
    wal = path + ".wal"
    if os.path.exists(wal):
        try:
            os.unlink(wal)
        except PermissionError:
            pass


def _build_origin(pos_id: str, mem_id: str) -> TradeOrigin:
    return TradeOrigin(
        position_id=pos_id,
        origin_episode_id=f"ep-{pos_id}",
        symbol="BTCUSDT",
        side="LONG",
        memory_id=mem_id,
        entry_price=50000.0,
        entry_atr=1000.0,
        origin_quality=OriginQuality.LOW,
    )


def _build_working(mem_id: str, pos_id: str) -> WorkingMemory:
    return WorkingMemory(
        memory_id=mem_id,
        position_id=pos_id,
        version=1,
    )


def _tim_write(tim_store: TimStore, pos_id: str) -> None:
    mem_id = f"mem-{uuid.uuid4().hex[:8]}"
    origin = _build_origin(pos_id, mem_id)
    tim_store.insert_origin(origin)
    wm = _build_working(mem_id, pos_id)
    tim_store.upsert_working_memory(wm)


class TestConcurrentReadWrite:
    def test_concurrent_portfolio_and_tim_writes(self, db_path):
        with DB_LOCK:
            ps = PortfolioStore(db_path=db_path)
            ps.create_schema()
            conn_ts = duckdb.connect(db_path)
            ts = TimStore(connection=conn_ts)
            ts.create_schema()

            n_writes = 10
            tim_results = []
            port_results = []

            with ThreadPoolExecutor(max_workers=8) as executor:
                tim_futures = []
                port_futures = []
                for i in range(n_writes):
                    pos_id = f"conc-pos-{i:04d}"
                    tim_futures.append(executor.submit(_tim_write, ts, pos_id))

                for i in range(n_writes):
                    pos_id = f"conc-pos-{i:04d}"
                    port_futures.append(executor.submit(ps.save_position, _make_concurrent_pos(pos_id)))

                for f in as_completed(tim_futures):
                    exc = f.exception()
                    if exc:
                        tim_results.append(("error", str(exc)))
                    else:
                        tim_results.append(("ok", ""))

                for f in as_completed(port_futures):
                    exc = f.exception()
                    if exc:
                        port_results.append(("error", str(exc)))
                    else:
                        port_results.append(("ok", ""))

            tim_errors = [r for r in tim_results if r[0] == "error"]
            port_errors = [r for r in port_results if r[0] == "error"]
            assert len(tim_errors) == 0, f"TimStore errors: {tim_errors}"
            assert len(port_errors) == 0, f"PortfolioStore errors: {port_errors}"

            origin_count = conn_ts.execute(
                "SELECT COUNT(*) FROM trade_memory_origin"
            ).fetchone()[0]
            working_count = conn_ts.execute(
                "SELECT COUNT(*) FROM trade_memory_working"
            ).fetchone()[0]
            assert origin_count == n_writes
            assert working_count == n_writes

            conn_ts.close()
            ps._conn.close()


def _make_concurrent_pos(pos_id: str):
    from src.core.models import Position, PositionState, ProtectionOrders
    return Position(
        position_id=pos_id,
        symbol="BTCUSDT",
        side="LONG",
        quantity=0.5,
        avg_fill_price=50000.0,
        anchor_symbol="BTC",
        entry_timestamp="2024-01-01T00:00:00Z",
        timeframe="5m",
        lifecycle_state=PositionState.OPEN,
        protection_orders=ProtectionOrders(stop_price=49000.0, tp_price=55000.0),
    )
