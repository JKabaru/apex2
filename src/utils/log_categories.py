"""Log category routing for focus/summary observability modes."""
from __future__ import annotations

import re
import threading
from collections import defaultdict
from typing import Any, Optional

import structlog

# Logger name → category mapping
LOGGER_CATEGORY_MAP: dict[str, str] = {
    "scanner": "scanner",
    "scanner_loop": "scanner",
    "market_context": "scanner",
    "execution_service": "execution",
    "protection_coordinator": "execution",
    "execution_router": "execution",
    "trade_monitor": "execution",
    "trade_manager": "execution",
    "trade_coordinator": "execution",
    "learning_pipeline": "learning",
    "learning_corpus": "learning",
    "experience_extractor": "learning",
    "experience_validator": "learning",
    "experience_normalizer": "learning",
    "system_recovery": "learning",
    "config_store": "learning",
    "research": "learning",
    "analytics_service": "learning",
    "metrics_service": "learning",
    "importance_scorer": "learning",
    "timeline_manager": "learning",
    "pattern_detector": "learning",
    "hypothesis_extractor": "learning",
    "knowledge_promoter": "learning",
    "observation_compressor": "learning",
    "prediction_lifecycle": "learning",
    "observation_ingestor": "learning",
    "reconciler": "reconciliation",
    "portfolio_manager": "reconciliation",
    "position_manager": "reconciliation",
    "event_bus": "system",
    "main": "system",
    "operator_cli": "system",
    "ws_ingestor": "data",
    "ingestor": "data",
    "aggregator": "data",
    "binance_client": "data",
    "correlation_dashboard": "data",
}

# Event key prefix → category (for operational events without logger mapping)
EVENT_PREFIX_CATEGORY: dict[str, str] = {
    "LEARNING_": "learning",
    "MEMORY_": "learning",
    "SYSTEM_RECOVERY": "learning",
    "ADAPTIVE_": "learning",
    "PROTECTION_": "execution",
    "EVENT_": "system",
    "CALLBACK_": "system",
    "POSITION_CLOSED": "learning",
    "POSITION_": "execution",
    "ORDER_": "execution",
    "CANDIDATE_": "scanner",
    "OBSERVATION_": "learning",
    "TIMELINE_": "learning",
    "PATTERN_": "learning",
    "HYPOTHESIS_": "learning",
    "KNOWLEDGE_": "learning",
    "PREDICTION_": "learning",
    "COMPRESSOR_": "learning",
    "MEMORY_": "learning",
}

OPERATIONAL_EVENT_RE = re.compile(r"^[A-Z][A-Z0-9_]+$")

ALL_CATEGORIES = ("scanner", "execution", "learning", "reconciliation", "data", "system")


def resolve_category(logger_name: str, event: str) -> str:
    if logger_name in LOGGER_CATEGORY_MAP:
        return LOGGER_CATEGORY_MAP[logger_name]
    for prefix, category in EVENT_PREFIX_CATEGORY.items():
        if event.startswith(prefix):
            return category
    return "system"


def is_operational_event(event: str) -> bool:
    return bool(OPERATIONAL_EVENT_RE.match(str(event)))


class SummaryAggregator:
    """Buffers non-focus INFO logs and emits periodic one-line summaries."""

    def __init__(self, interval_seconds: int = 60):
        self._interval = max(15, interval_seconds)
        self._lock = threading.Lock()
        self._buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "last_event": "", "last_detail": ""}
        )
        self._timer: Optional[threading.Timer] = None
        self._logger = structlog.get_logger("log_summary")

    def start(self) -> None:
        self._schedule_flush()

    def stop(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None
        self.flush()

    def record(self, category: str, event: str, detail: str = "") -> None:
        with self._lock:
            bucket = self._buckets[category]
            bucket["count"] += 1
            bucket["last_event"] = event
            if detail:
                bucket["last_detail"] = detail

    def flush(self) -> None:
        with self._lock:
            buckets = dict(self._buckets)
            self._buckets.clear()

        for category, data in buckets.items():
            if data["count"] <= 0:
                continue
            detail = f" — last: {data['last_detail']}" if data["last_detail"] else ""
            self._logger.info(
                "LOG_SUMMARY",
                category=category,
                events=data["count"],
                last_event=data["last_event"],
                detail=detail.strip(" —"),
            )

    def _schedule_flush(self) -> None:
        self._timer = threading.Timer(self._interval, self._on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _on_timer(self) -> None:
        self.flush()
        self._schedule_flush()


_summary_aggregator: Optional[SummaryAggregator] = None
_focus_categories: set[str] = set(ALL_CATEGORIES)
_log_level_name = "INFO"


def configure_category_logging(
    focus: Optional[list[str]] = None,
    level: str = "INFO",
    summary_interval_seconds: int = 60,
) -> None:
    global _summary_aggregator, _focus_categories, _log_level_name
    _log_level_name = level.upper()
    if focus:
        _focus_categories = {c.lower() for c in focus if c.lower() in ALL_CATEGORIES}
    else:
        _focus_categories = set(ALL_CATEGORIES)

    if _summary_aggregator:
        _summary_aggregator.stop()
    _summary_aggregator = SummaryAggregator(interval_seconds=summary_interval_seconds)
    _summary_aggregator.start()


def get_focus_categories() -> set[str]:
    return set(_focus_categories)


def category_filter_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any],
) -> dict[str, Any]:
    """structlog processor: full detail for focus categories, summary for others."""
    if event_dict.get("_force_log"):
        return event_dict

    logger_name = event_dict.get("logger", "")
    event = str(event_dict.get("event", ""))
    level = event_dict.get("level", method_name).upper()
    category = resolve_category(logger_name, event)
    event_dict["_log_category"] = category

    # Always pass warnings and above
    level_rank = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    if level_rank.get(level, 20) >= level_rank["WARNING"]:
        return event_dict

    # Focus category → full detail
    if category in _focus_categories:
        return event_dict

    # Non-focus operational events → one-line summary
    if is_operational_event(event):
        event_dict["event"] = f"[{category}] {event}"
        keep = {"event", "logger", "level", "timestamp", "_log_category"}
        summary_keys = (
            "position_id", "symbol", "status", "result", "decision",
            "stored", "experience_count", "recovered", "outcome",
        )
        for key in summary_keys:
            if key in event_dict:
                keep.add(key)
        return {k: v for k, v in event_dict.items() if k in keep}

    # Non-focus routine INFO → buffer for periodic summary
    if _summary_aggregator and level == "INFO":
        detail = ""
        for key in ("symbol", "position_id", "status", "error"):
            if key in event_dict:
                detail = f"{key}={event_dict[key]}"
                break
        _summary_aggregator.record(category, event, detail)
        raise structlog.DropEvent

    return event_dict


def stop_category_logging() -> None:
    global _summary_aggregator
    if _summary_aggregator:
        _summary_aggregator.stop()
        _summary_aggregator = None
