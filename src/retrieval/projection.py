from __future__ import annotations

from src.models.learning.trade_experience import LearningManifest
from src.retrieval.models import RetrievalQuery, RetrievalRecord


class CorpusProjection:
    """Anti-corruption layer between the storage schema and the retrieval layer.
    This is the ONLY class that knows the internal structure of LearningManifest.
    It converts nested manifest objects into flat, comparable RetrievalRecords."""

    @staticmethod
    def project_manifest(manifest: LearningManifest) -> RetrievalRecord:
        experience = manifest.learning_experience
        validation = manifest.validation_report
        metrics = manifest.normalized_metrics
        opportunity = manifest.opportunity_identity

        record_source = "interim" if manifest.experience_type == "interim" else "finalized"

        return RetrievalRecord(
            experience_id=manifest.experience_id,
            position_id=manifest.position_id,
            schema_version=manifest.schema_version,
            pipeline_version=manifest.pipeline_version,
            created_at=manifest.created_at,
            record_source=record_source,
            hash=manifest.hash,
            symbol=experience.symbol,
            timeframe=experience.timeframe,
            side=experience.side,
            opportunity_id=experience.opportunity_id or "",
            market_state_hash=opportunity.market_state_hash if opportunity else "",
            experience_type=manifest.experience_type,
            trend_regime=experience.trend_regime,
            volatility_regime=experience.volatility_regime,
            correlation_regime=experience.correlation_regime,
            integrity_score=validation.integrity_score,
            normalized_entry_atr_multiple=metrics.normalized_entry_atr_multiple,
            normalized_exit_atr_multiple=metrics.normalized_exit_atr_multiple,
            pnl_atr_multiple=metrics.pnl_atr_multiple,
            mfe_atr_multiple=metrics.mfe_atr_multiple,
            mae_atr_multiple=metrics.mae_atr_multiple,
            entry_rsi_percentile=metrics.entry_rsi_percentile,
            entry_volatility_percentile=metrics.entry_volatility_percentile,
            holding_duration_minutes=metrics.holding_duration_minutes,
            bars_held=metrics.bars_held,
            total_slippage_bps=metrics.total_slippage_bps,
            total_fees_bps=metrics.total_fees_bps,
            realized_rr=metrics.realized_rr,
            initial_risk_atr_multiple=metrics.initial_risk_atr_multiple,
            evidence_episodes_summary=experience.evidence_episodes_summary,
            episode_count=experience.episode_count,
        )

    @staticmethod
    def project_query(query: RetrievalQuery) -> dict:
        result: dict = {}
        if query.symbol is not None:
            result["symbol"] = query.symbol
        if query.timeframe is not None:
            result["timeframe"] = query.timeframe
        if query.trend_regime is not None:
            result["trend_regime"] = query.trend_regime
        if query.volatility_regime is not None:
            result["volatility_regime"] = query.volatility_regime
        if query.correlation_regime is not None:
            result["correlation_regime"] = query.correlation_regime
        if query.opportunity_id is not None:
            result["opportunity_id"] = query.opportunity_id
        if query.market_state_hash is not None:
            result["market_state_hash"] = query.market_state_hash
        if query.episode_count > 0:
            result["evidence.episode_count"] = query.episode_count
        return result
