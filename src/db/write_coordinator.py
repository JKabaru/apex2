from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generator

import duckdb
import structlog

logger = structlog.get_logger("write_coordinator")


class WriteCoordinatorTimeoutError(Exception):
    pass


class DatabaseWriteCoordinator:
    """Exclusive write access coordinator for PortfolioStore + TimStore.

    Stage 1: class exists and is tested in isolation.
    Stage 2+: wired into runtime services for coordinated writes.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection, timeout_seconds: float = 30.0):
        self._conn = connection
        self._timeout = timeout_seconds
        self._lock = threading.Lock()

    @contextmanager
    def exclusive_transaction(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Acquires exclusive lock, begins transaction, yields connection.

        On success: commits.
        On exception: rolls back.
        On timeout: raises WriteCoordinatorTimeoutError.
        """
        acquired = self._lock.acquire(timeout=self._timeout)
        if not acquired:
            raise WriteCoordinatorTimeoutError(
                f"Could not acquire write lock within {self._timeout}s"
            )
        try:
            self._conn.begin()
            yield self._conn
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
