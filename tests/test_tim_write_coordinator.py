from __future__ import annotations

import threading
import time

import duckdb
import pytest

from src.db.write_coordinator import DatabaseWriteCoordinator, WriteCoordinatorTimeoutError


@pytest.fixture
def conn():
    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE IF NOT EXISTS test_data (id INTEGER PRIMARY KEY, value VARCHAR)")
    yield db
    db.close()


@pytest.fixture
def coordinator(conn):
    return DatabaseWriteCoordinator(conn, timeout_seconds=5.0)


class TestCommit:
    def test_commit_persists_writes(self, coordinator, conn):
        with coordinator.exclusive_transaction() as c:
            c.execute("INSERT INTO test_data VALUES (1, 'hello')")
        row = conn.execute("SELECT * FROM test_data WHERE id = 1").fetchone()
        assert row is not None
        assert row[1] == "hello"


class TestRollback:
    def test_rollback_discards_writes(self, coordinator, conn):
        try:
            with coordinator.exclusive_transaction() as c:
                c.execute("INSERT INTO test_data VALUES (2, 'rollback_me')")
                raise RuntimeError("force rollback")
        except RuntimeError:
            pass
        row = conn.execute("SELECT * FROM test_data WHERE id = 2").fetchone()
        assert row is None


class TestTimeout:
    def test_timeout_raises_error(self, conn):
        fast_coord = DatabaseWriteCoordinator(conn, timeout_seconds=0.3)
        fast_coord._lock.acquire()
        start = time.time()
        with pytest.raises(WriteCoordinatorTimeoutError):
            with fast_coord.exclusive_transaction() as c:
                pass
        elapsed = time.time() - start
        assert elapsed < 1.0
        fast_coord._lock.release()


class TestConcurrency:
    def test_concurrent_writes_serialized(self, conn):
        coord = DatabaseWriteCoordinator(conn, timeout_seconds=10.0)
        results = []

        def writer_a():
            with coord.exclusive_transaction() as c:
                time.sleep(0.3)
                c.execute("INSERT INTO test_data VALUES (10, 'from_a')")
                results.append("a_done")

        def writer_b():
            time.sleep(0.05)
            with coord.exclusive_transaction() as c:
                c.execute("INSERT INTO test_data VALUES (11, 'from_b')")
                results.append("b_done")

        t1 = threading.Thread(target=writer_a)
        t2 = threading.Thread(target=writer_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert "a_done" in results
        assert "b_done" in results
        row_a = conn.execute("SELECT * FROM test_data WHERE id = 10").fetchone()
        row_b = conn.execute("SELECT * FROM test_data WHERE id = 11").fetchone()
        assert row_a is not None
        assert row_b is not None
