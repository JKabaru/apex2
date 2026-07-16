import logging
from typing import Any, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from src.utils.log_categories import category_filter_processor, configure_category_logging

console = Console(legacy_windows=False)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


def init_logging(config: Optional[dict[str, Any]] = None) -> None:
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        if sys.platform != "win32" or not sys.stdout.isatty():
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
    log_cfg = (config or {}).get("logging", {})
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)],
        level=level,
        force=True,
    )

    focus = log_cfg.get("focus")
    if isinstance(focus, str):
        focus = [c.strip() for c in focus.split(",") if c.strip()]
    summary_interval = int(log_cfg.get("summary_interval_seconds", 60))

    configure_category_logging(
        focus=focus,
        level=level_name,
        summary_interval_seconds=summary_interval,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            category_filter_processor,
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
