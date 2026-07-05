from __future__ import annotations

import structlog

from src.models.reasoning import MarketContext
from src.services.market_context import MarketContextService

logger = structlog.get_logger("state_builder")


async def build_state(symbol: str, timeframe: str, anchors: list = None) -> dict:
    logger.warning(
        "build_state() is deprecated and delegates to MarketContextService. "
        "Use MarketContextService.get_context() or get_state() directly.",
        symbol=symbol,
        timeframe=timeframe,
    )
    return {}


async def build_context(
    symbol: str,
    timeframe: str,
    context_service: MarketContextService,
) -> MarketContext:
    return await context_service.get_context(symbol, timeframe)
