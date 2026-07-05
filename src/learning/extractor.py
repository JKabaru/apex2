from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from src.models.learning.trade_experience import (
    LearningExperience,
    PositionSnapshot,
)

logger = structlog.get_logger("experience_extractor")


class ExperienceExtractor:
    """Stage 1 of the learning pipeline.
    Pure field mapping from PositionSnapshot to LearningExperience.
    Zero computation. Zero OHLCV reads. Zero judgment."""

    extraction_version: str = "1.0"

    def extract(
        self,
        snapshot: PositionSnapshot,
        opportunity_id: str = "",
    ) -> LearningExperience:
        evidence_summary = []
        for ep in (snapshot.evidence_episodes or []):
            entry = {
                "episode_id": ep.get("episode_id", ""),
                "index": ep.get("index", 0),
                "state_profile": ep.get("state_profile", ""),
                "started_at": ep.get("started_at", ""),
                "ended_at": ep.get("ended_at"),
            }
            duration = None
            if ep.get("started_at") and ep.get("ended_at"):
                try:
                    start = datetime.fromisoformat(ep["started_at"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(ep["ended_at"].replace("Z", "+00:00"))
                    duration = (end - start).total_seconds() / 60.0
                except (ValueError, TypeError):
                    pass
            entry["duration_minutes"] = duration
            evidence_summary.append(entry)

        return LearningExperience(
            position_id=snapshot.position_id,
            opportunity_id=opportunity_id or snapshot.opportunity_id,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            entry_price=snapshot.avg_fill_price,
            exit_price=snapshot.exit_price,
            fees=snapshot.fees,
            exit_fees=snapshot.exit_fees,
            highest_unrealized_profit=snapshot.highest_unrealized_profit,
            maximum_drawdown=snapshot.maximum_drawdown,
            slippage_bps=snapshot.slippage_bps,
            spread_bps=snapshot.spread_bps,
            entry_atr=snapshot.entry_atr,
            entry_rsi=snapshot.entry_rsi,
            exit_atr=snapshot.exit_atr,
            exit_rsi=snapshot.exit_rsi,
            trend_regime=snapshot.trend_regime,
            volatility_regime=snapshot.volatility_regime,
            correlation_regime=snapshot.correlation_regime,
            calibration_data=snapshot.calibration_data,
            evidence_episodes_summary=evidence_summary,
            episode_count=len(evidence_summary),
        )
