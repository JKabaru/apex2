from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

CONFIDENCE_LEVEL = Literal["CANONICAL", "DERIVED", "ESTIMATED"]


class ProvenanceRecord(BaseModel, frozen=True):
    """Field-level lineage record.
    Describes where a field originated, how it was computed,
    and how trustworthy the value is."""
    field: str
    component: str
    calculation: str
    dependencies: list[str] = Field(default_factory=list)
    schema_version: str = "1.0"
    confidence: CONFIDENCE_LEVEL = "DERIVED"
    indicator_versions: dict[str, str] = Field(default_factory=dict)


class ProvenanceRegistry:
    """Registry mapping field names to their full provenance lineage.
    Supports field-level lookup, not just version strings."""
    def __init__(self, records: dict[str, ProvenanceRecord]):
        self._records = records

    @property
    def version(self) -> str:
        versions = {r.schema_version for r in self._records.values()}
        return max(versions) if versions else "1.0"

    def lookup(self, field: str) -> Optional[ProvenanceRecord]:
        return self._records.get(field)

    def get_all_records(self) -> list[ProvenanceRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)


def _build_default_registry() -> ProvenanceRegistry:
    records: list[ProvenanceRecord] = [
        # ── CANONICAL ──
        ProvenanceRecord(
            field="entry_price", component="ExecutionService",
            calculation="ORDER_FILLED.avgPrice", confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="exit_price", component="ExecutionService",
            calculation="ORDER_FILLED.exit_price", confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="fees", component="ExecutionService",
            calculation="fill.commission sum", confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="exit_fees", component="ExecutionService",
            calculation="exit_fill.commission", confidence="CANONICAL",
        ),

        # ── DERIVED ──
        ProvenanceRecord(
            field="quantity", component="ExecutionService",
            calculation="sizing * leverage / confidence", confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="entry_atr", component="MarketContextService",
            calculation="ATR(14) from OHLCV series", dependencies=["ohlcv_1m"],
            confidence="DERIVED", indicator_versions={"ATR": "1.0"},
        ),
        ProvenanceRecord(
            field="entry_rsi", component="MarketContextService",
            calculation="RSI(14) from OHLCV series", dependencies=["ohlcv_1m"],
            confidence="DERIVED", indicator_versions={"RSI": "1.0"},
        ),
        ProvenanceRecord(
            field="exit_atr", component="MarketContextService",
            calculation="ATR(14) from OHLCV series", dependencies=["ohlcv_1m"],
            confidence="DERIVED", indicator_versions={"ATR": "1.0"},
        ),
        ProvenanceRecord(
            field="exit_rsi", component="MarketContextService",
            calculation="RSI(14) from OHLCV series", dependencies=["ohlcv_1m"],
            confidence="DERIVED", indicator_versions={"RSI": "1.0"},
        ),
        ProvenanceRecord(
            field="trend_regime", component="MarketContextService",
            calculation="SMA200 crossover classification", dependencies=["ohlcv_1m"],
            confidence="DERIVED", indicator_versions={"SMA": "1.0"},
        ),
        ProvenanceRecord(
            field="volatility_regime", component="MarketContextService",
            calculation="ATR percentile classification", dependencies=["ohlcv_1m"],
            confidence="DERIVED", indicator_versions={"ATR_percentile": "1.0"},
        ),
        ProvenanceRecord(
            field="correlation_regime", component="CorrelationEngine",
            calculation="Pearson rolling window (500, lag=15)", dependencies=["close"],
            confidence="DERIVED", indicator_versions={"Pearson": "1.0"},
        ),
        ProvenanceRecord(
            field="initial_stop_loss", component="ExecutionService",
            calculation="entry_price * stop_loss_pct", dependencies=["entry_price"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="initial_take_profit", component="ExecutionService",
            calculation="entry_price * take_profit_pct", dependencies=["entry_price"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="highest_unrealized_profit", component="PositionManager",
            calculation="max(unrealized_pnl) during trade",
            dependencies=["current_price", "avg_fill_price"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="maximum_drawdown", component="PositionManager",
            calculation="max(drawdown) during trade",
            dependencies=["current_price", "avg_fill_price"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="evidence_episodes", component="PositionManager",
            calculation="categorical state change detection",
            dependencies=["MarketContext", "state_profile"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="normalized_entry_atr_multiple", component="ExperienceNormalizer",
            calculation="entry_price / atr",
            dependencies=["entry_price", "entry_atr"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="pnl_atr_multiple", component="ExperienceNormalizer",
            calculation="pnl_usdt / (atr * qty)",
            dependencies=["entry_price", "exit_price", "entry_atr"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="mfe_atr_multiple", component="ExperienceNormalizer",
            calculation="mfe / (atr * qty)",
            dependencies=["highest_unrealized_profit", "entry_atr"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="mae_atr_multiple", component="ExperienceNormalizer",
            calculation="mae / (atr * qty)",
            dependencies=["maximum_drawdown", "entry_atr"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="bars_held", component="ExperienceNormalizer",
            calculation="holding_minutes / timeframe_minutes",
            dependencies=["holding_duration_minutes", "timeframe"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="holding_duration_minutes", component="ExperienceNormalizer",
            calculation="exit_timestamp - entry_timestamp",
            dependencies=["entry_timestamp", "exit_timestamp"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="calibration_data", component="PositionManager",
            calculation="shadow_vs_live comparison",
            dependencies=["shadow_entry", "live_entry", "shadow_exit", "live_exit"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="side", component="TradeCoordinator",
            calculation="candidate.proposed_side mapped to LONG/SHORT",
            confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="exit_reason", component="PositionManager",
            calculation="SL_HIT/TP_HIT/MANUAL from monitor or fill",
            confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="correlation_score", component="CorrelationEngine",
            calculation="Pearson rolling window (500, lag=15)",
            dependencies=["close"], confidence="DERIVED",
            indicator_versions={"Pearson": "1.0"},
        ),
        ProvenanceRecord(
            field="symbol", component="Scanner",
            calculation="from market universe alternates list",
            confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="timeframe", component="Scanner",
            calculation="default 5m, set per candidate",
            confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="execution_mode", component="TradeCoordinator",
            calculation="LIVE if APPROVED, SHADOW otherwise",
            confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="origin", component="TradeCoordinator",
            calculation="NORMAL if APPROVED, CONSTRAINT if rejected, MIRROR if mirrored",
            confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="avg_fill_price", component="ExecutionService",
            calculation="ORDER_FILLED.avgPrice (weighted average)",
            confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="anchor_symbol", component="CorrelationEngine",
            calculation="best anchor from correlation analysis",
            confidence="CANONICAL",
        ),
        ProvenanceRecord(
            field="episode_count", component="PositionManager",
            calculation="len(evidence_episodes)",
            dependencies=["evidence_episodes"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="episodes_summary", component="PositionManager",
            calculation="categorical state change detection serialized",
            dependencies=["MarketContext", "state_profile"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="initial_risk_atr_multiple", component="ExperienceNormalizer",
            calculation="(entry_price - stop_loss) / atr",
            dependencies=["entry_price", "initial_stop_loss", "entry_atr"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="realized_rr", component="ExperienceNormalizer",
            calculation="(exit_price - entry_price) / (entry_price - stop_loss) * direction",
            dependencies=["entry_price", "exit_price", "initial_stop_loss", "side"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="entry_error_bps", component="PositionManager",
            calculation="(shadow_entry - live_entry) / live_entry * 10000",
            dependencies=["shadow_entry", "live_entry"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="exit_error_bps", component="PositionManager",
            calculation="(shadow_exit - live_exit) / live_exit * 10000",
            dependencies=["shadow_exit", "live_exit"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="fee_error_usdt", component="PositionManager",
            calculation="shadow_fees - live_fees",
            dependencies=["shadow_fees", "live_fees"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="entry_rsi_percentile", component="ExperienceNormalizer",
            calculation="percentile rank of entry_rsi in RSI(14) series",
            dependencies=["entry_rsi", "ohlcv_1m"],
            confidence="DERIVED", indicator_versions={"RSI": "1.0", "Percentile": "1.0"},
        ),
        ProvenanceRecord(
            field="entry_volatility_percentile", component="ExperienceNormalizer",
            calculation="percentile rank of entry_atr in ATR(14) series",
            dependencies=["entry_atr", "ohlcv_1m"],
            confidence="DERIVED", indicator_versions={"ATR": "1.0", "Percentile": "1.0"},
        ),
        ProvenanceRecord(
            field="normalized_entry_atr_multiple", component="ExperienceNormalizer",
            calculation="entry_price / atr",
            dependencies=["entry_price", "entry_atr"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="normalized_exit_atr_multiple", component="ExperienceNormalizer",
            calculation="(exit_price - entry_price) * direction / atr",
            dependencies=["exit_price", "entry_price", "entry_atr", "side"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="total_slippage_bps", component="ExperienceNormalizer",
            calculation="slippage_bps + spread_bps",
            dependencies=["slippage_bps", "spread_bps"],
            confidence="DERIVED",
        ),
        ProvenanceRecord(
            field="total_fees_bps", component="ExperienceNormalizer",
            calculation="(fees + exit_fees) / (entry_price * qty) * 10000",
            dependencies=["fees", "exit_fees", "entry_price"],
            confidence="DERIVED",
        ),

        # ── ESTIMATED ──
        ProvenanceRecord(
            field="slippage_bps", component="VirtualExecutor/LiveExecutor",
            calculation="synthetic friction parameter or exchange fill data",
            confidence="ESTIMATED",
        ),
        ProvenanceRecord(
            field="spread_bps", component="VirtualExecutor",
            calculation="synthetic friction parameter",
            confidence="ESTIMATED",
        ),
        ProvenanceRecord(
            field="entry_thesis", component="Scanner/TradeCoordinator",
            calculation="LLM-generated rationale",
            confidence="ESTIMATED",
        ),
    ]

    return ProvenanceRegistry({r.field: r for r in records})
