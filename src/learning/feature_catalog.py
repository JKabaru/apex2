from __future__ import annotations

import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

FEATURE_CATEGORY = Literal[
    "MARKET", "POSITION", "EXECUTION", "RISK",
    "EVIDENCE", "OUTCOME", "TEMPORAL", "CALIBRATION", "CONTEXT",
]


COMPARISON_TYPE = Literal["continuous", "categorical", "boolean"]
DISTANCE_FUNCTION = Literal["normalized_absolute", "exact_match"]


def _canonical_json(obj) -> str:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"),
    )


class FeatureDefinition(BaseModel, frozen=True):
    """Metadata for a single observable feature in the learning corpus.
    This is NOT a value — it is a description of a value.
    The catalog never sees runtime objects."""
    feature_id: str
    name: str
    description: str
    category: FEATURE_CATEGORY
    data_type: str
    unit: str
    nullable: bool
    produced_by: str
    depends_on: list[str] = Field(default_factory=list)
    normalizer: str = "none"
    is_learnable: bool = False
    is_configurable: bool = False
    introduced_version: str = "1.0"

    # ── Similarity comparison metadata ──
    comparison_type: COMPARISON_TYPE = "continuous"
    comparison_range: float = 1.0
    distance_function: DISTANCE_FUNCTION = "normalized_absolute"


class FeatureCatalog:
    """Pure metadata registry of every observable feature.
    Never accepts runtime types. Content-addressable via catalog_hash."""
    def __init__(self, features: dict[str, FeatureDefinition]):
        self._features = features

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def catalog_hash(self) -> str:
        definitions = sorted(self._features.values(), key=lambda f: f.feature_id)
        serialized = "".join(
            _canonical_json(d.model_dump(mode="json")) for d in definitions
        )
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get_feature(self, feature_id: str) -> Optional[FeatureDefinition]:
        return self._features.get(feature_id)

    def has_feature(self, feature_id: str) -> bool:
        return feature_id in self._features

    def get_learnable_features(self) -> list[FeatureDefinition]:
        return [f for f in self._features.values() if f.is_learnable]

    def list_features(self, category: Optional[str] = None) -> list[FeatureDefinition]:
        if category is None:
            return list(self._features.values())
        return [f for f in self._features.values() if f.category == category]

    def get_all_features(self) -> list[FeatureDefinition]:
        return list(self._features.values())

    def __contains__(self, feature_id: str) -> bool:
        return feature_id in self._features

    def __len__(self) -> int:
        return len(self._features)


def _build_default_catalog() -> FeatureCatalog:
    """Construct the default FeatureCatalog with all known features."""
    features: list[FeatureDefinition] = [
        # ── MARKET ──
        FeatureDefinition(
            feature_id="market.entry_price",
            name="Entry Price",
            description="Average fill price at position entry",
            category="MARKET", data_type="float", unit="quote_asset", nullable=False,
            produced_by="ExecutionService", is_learnable=False, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="market.exit_price",
            name="Exit Price",
            description="Average fill price at position exit",
            category="MARKET", data_type="float", unit="quote_asset", nullable=True,
            produced_by="ExecutionService", is_learnable=False, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="market.entry_atr",
            name="Entry ATR",
            description="Average True Range at entry candle (14-period)",
            category="MARKET", data_type="float", unit="quote_asset", nullable=True,
            produced_by="MarketContextService", depends_on=["ohlcv_1m"],
            normalizer="atr_multiple", is_learnable=True, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="market.entry_rsi",
            name="Entry RSI",
            description="Relative Strength Index at entry candle (14-period)",
            category="MARKET", data_type="float", unit="index", nullable=True,
            produced_by="MarketContextService", depends_on=["ohlcv_1m"],
            normalizer="percentile", is_learnable=True, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="market.exit_atr",
            name="Exit ATR",
            description="Average True Range at exit candle (14-period)",
            category="MARKET", data_type="float", unit="quote_asset", nullable=True,
            produced_by="MarketContextService", depends_on=["ohlcv_1m"],
            is_learnable=True, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="market.exit_rsi",
            name="Exit RSI",
            description="Relative Strength Index at exit candle (14-period)",
            category="MARKET", data_type="float", unit="index", nullable=True,
            produced_by="MarketContextService", depends_on=["ohlcv_1m"],
            is_learnable=True, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="market.trend_regime",
            name="Trend Regime",
            description="Categorical trend regime at entry",
            category="MARKET", data_type="string", unit="", nullable=True,
            produced_by="MarketContextService", depends_on=["ohlcv_1m", "SMA200"],
            is_learnable=True, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),
        FeatureDefinition(
            feature_id="market.volatility_regime",
            name="Volatility Regime",
            description="Categorical volatility regime at entry",
            category="MARKET", data_type="string", unit="", nullable=True,
            produced_by="MarketContextService", depends_on=["ohlcv_1m", "ATR_percentile"],
            is_learnable=True, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),
        FeatureDefinition(
            feature_id="market.correlation_regime",
            name="Correlation Regime",
            description="Categorical correlation regime at entry",
            category="MARKET", data_type="string", unit="", nullable=True,
            produced_by="CorrelationEngine", depends_on=["close", "Pearson"],
            is_learnable=True, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),
        FeatureDefinition(
            feature_id="market.correlation_score",
            name="Correlation Score",
            description="Numerical correlation score at entry",
            category="MARKET", data_type="float", unit="index", nullable=False,
            produced_by="CorrelationEngine", depends_on=["close", "Pearson"],
            is_learnable=True, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="market.normalized_entry_atr_multiple",
            name="Normalized Entry ATR Multiple",
            description="Entry price expressed as multiple of ATR at entry",
            category="MARKET", data_type="float", unit="atr_units", nullable=True,
            produced_by="ExperienceNormalizer", depends_on=["entry_price", "entry_atr"],
            normalizer="none", is_learnable=True, is_configurable=False,
            comparison_range=5.0,
        ),
        FeatureDefinition(
            feature_id="market.normalized_exit_atr_multiple",
            name="Normalized Exit ATR Multiple",
            description="Price change from entry to exit in ATR units",
            category="MARKET", data_type="float", unit="atr_units", nullable=True,
            produced_by="ExperienceNormalizer", depends_on=["exit_price", "entry_price", "entry_atr"],
            normalizer="atr_multiple", is_learnable=True, is_configurable=False,
            comparison_range=5.0,
        ),
        FeatureDefinition(
            feature_id="market.entry_rsi_percentile",
            name="Entry RSI Percentile",
            description="Percentile rank of entry RSI in recent RSI series",
            category="MARKET", data_type="float", unit="percentile", nullable=True,
            produced_by="ExperienceNormalizer", depends_on=["entry_rsi", "ohlcv_1m"],
            normalizer="percentile", is_learnable=True, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="market.entry_volatility_percentile",
            name="Entry Volatility Percentile",
            description="Percentile rank of entry ATR in recent ATR series",
            category="MARKET", data_type="float", unit="percentile", nullable=True,
            produced_by="ExperienceNormalizer", depends_on=["entry_atr", "ohlcv_1m"],
            normalizer="percentile", is_learnable=True, is_configurable=False,
        ),

        # ── POSITION ──
        FeatureDefinition(
            feature_id="position.side",
            name="Position Side",
            description="Direction of the position (LONG or SHORT)",
            category="POSITION", data_type="string", unit="", nullable=False,
            produced_by="TradeCoordinator",
            is_learnable=True, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),
        FeatureDefinition(
            feature_id="position.quantity",
            name="Position Quantity",
            description="Executed quantity of the position",
            category="POSITION", data_type="float", unit="base_asset", nullable=False,
            produced_by="ExecutionService",
            is_learnable=False, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="position.avg_fill_price",
            name="Average Fill Price",
            description="Weighted average price of entry fills",
            category="POSITION", data_type="float", unit="quote_asset", nullable=False,
            produced_by="ExecutionService",
            is_learnable=False, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="position.fees",
            name="Entry Fees",
            description="Total commission paid at entry",
            category="POSITION", data_type="float", unit="quote_asset", nullable=False,
            produced_by="ExecutionService",
            is_learnable=False, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="position.exit_fees",
            name="Exit Fees",
            description="Total commission paid at exit",
            category="POSITION", data_type="float", unit="quote_asset", nullable=True,
            produced_by="ExecutionService",
            is_learnable=False, is_configurable=False,
        ),

        # ── EXECUTION ──
        FeatureDefinition(
            feature_id="execution.slippage_bps",
            name="Entry Slippage",
            description="Estimated or actual slippage at entry in basis points",
            category="EXECUTION", data_type="float", unit="bps", nullable=True,
            produced_by="VirtualExecutor/LiveExecutor",
            is_learnable=True, is_configurable=True,
        ),
        FeatureDefinition(
            feature_id="execution.spread_bps",
            name="Entry Spread",
            description="Estimated spread at entry in basis points",
            category="EXECUTION", data_type="float", unit="bps", nullable=True,
            produced_by="VirtualExecutor",
            is_learnable=True, is_configurable=True,
        ),
        FeatureDefinition(
            feature_id="execution.total_slippage_bps",
            name="Total Slippage",
            description="Sum of entry slippage and spread in basis points",
            category="EXECUTION", data_type="float", unit="bps", nullable=True,
            produced_by="ExperienceNormalizer", depends_on=["slippage_bps", "spread_bps"],
            is_learnable=True, is_configurable=False,
            comparison_range=50.0,
        ),
        FeatureDefinition(
            feature_id="execution.total_fees_bps",
            name="Total Fees",
            description="Total fees (entry + exit) as percentage of notional in bps",
            category="EXECUTION", data_type="float", unit="bps", nullable=True,
            produced_by="ExperienceNormalizer", depends_on=["fees", "exit_fees", "entry_price"],
            is_learnable=True, is_configurable=False,
            comparison_range=50.0,
        ),

        # ── RISK ──
        FeatureDefinition(
            feature_id="risk.initial_stop_loss",
            name="Initial Stop Loss",
            description="Initial stop loss price set at position open",
            category="RISK", data_type="float", unit="quote_asset", nullable=False,
            produced_by="ExecutionService",
            is_learnable=False, is_configurable=True,
        ),
        FeatureDefinition(
            feature_id="risk.initial_take_profit",
            name="Initial Take Profit",
            description="Initial take profit price set at position open",
            category="RISK", data_type="float", unit="quote_asset", nullable=False,
            produced_by="ExecutionService",
            is_learnable=False, is_configurable=True,
        ),
        FeatureDefinition(
            feature_id="risk.initial_risk_atr_multiple",
            name="Initial Risk ATR Multiple",
            description="Distance from entry to initial stop in ATR units",
            category="RISK", data_type="float", unit="atr_units", nullable=True,
            produced_by="ExperienceNormalizer",
            depends_on=["entry_price", "initial_stop_loss", "entry_atr"],
            normalizer="atr_multiple", is_learnable=True, is_configurable=True,
            comparison_range=5.0,
        ),
        FeatureDefinition(
            feature_id="risk.realized_rr",
            name="Realized Risk-Reward",
            description="Realized reward relative to initial risk (stop distance)",
            category="RISK", data_type="float", unit="ratio", nullable=True,
            produced_by="ExperienceNormalizer",
            depends_on=["entry_price", "exit_price", "initial_stop_loss"],
            normalizer="ratio", is_learnable=True, is_configurable=False,
            comparison_range=5.0,
        ),

        # ── EVIDENCE ──
        FeatureDefinition(
            feature_id="evidence.episode_count",
            name="Evidence Episode Count",
            description="Number of categorical market regime episodes during the trade",
            category="EVIDENCE", data_type="integer", unit="episodes", nullable=False,
            produced_by="PositionManager", depends_on=["evidence_episodes"],
            is_learnable=True, is_configurable=False,
            comparison_range=20.0,
        ),
        FeatureDefinition(
            feature_id="evidence.episodes_summary",
            name="Evidence Episodes Summary",
            description="List of categorical regime episodes with duration",
            category="EVIDENCE", data_type="list", unit="", nullable=False,
            produced_by="PositionManager", depends_on=["MarketContext", "state_profile"],
            is_learnable=False, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),

        # ── OUTCOME ──
        FeatureDefinition(
            feature_id="outcome.exit_reason",
            name="Exit Reason",
            description="Reason the position was closed (TP_HIT, SL_HIT, MANUAL, etc.)",
            category="OUTCOME", data_type="string", unit="", nullable=True,
            produced_by="PositionManager",
            is_learnable=True, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),
        FeatureDefinition(
            feature_id="outcome.highest_unrealized_profit",
            name="Maximum Favorable Excursion (MFE)",
            description="Highest unrealized profit during the trade",
            category="OUTCOME", data_type="float", unit="quote_asset", nullable=False,
            produced_by="PositionManager",
            is_learnable=True, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="outcome.maximum_drawdown",
            name="Maximum Adverse Excursion (MAE)",
            description="Largest unrealized loss during the trade",
            category="OUTCOME", data_type="float", unit="quote_asset", nullable=False,
            produced_by="PositionManager",
            is_learnable=True, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="outcome.pnl_atr_multiple",
            name="PnL ATR Multiple",
            description="Total realized PnL normalized by ATR and quantity",
            category="OUTCOME", data_type="float", unit="atr_units", nullable=True,
            produced_by="ExperienceNormalizer",
            depends_on=["entry_price", "exit_price", "entry_atr"],
            normalizer="atr_multiple", is_learnable=True, is_configurable=False,
            comparison_range=5.0,
        ),
        FeatureDefinition(
            feature_id="outcome.mfe_atr_multiple",
            name="MFE ATR Multiple",
            description="Maximum Favorable Excursion normalized by ATR",
            category="OUTCOME", data_type="float", unit="atr_units", nullable=True,
            produced_by="ExperienceNormalizer",
            depends_on=["highest_unrealized_profit", "entry_atr"],
            normalizer="atr_multiple", is_learnable=True, is_configurable=False,
            comparison_range=5.0,
        ),
        FeatureDefinition(
            feature_id="outcome.mae_atr_multiple",
            name="MAE ATR Multiple",
            description="Maximum Adverse Excursion normalized by ATR",
            category="OUTCOME", data_type="float", unit="atr_units", nullable=True,
            produced_by="ExperienceNormalizer",
            depends_on=["maximum_drawdown", "entry_atr"],
            normalizer="atr_multiple", is_learnable=True, is_configurable=False,
            comparison_range=5.0,
        ),

        # ── TEMPORAL ──
        FeatureDefinition(
            feature_id="temporal.holding_duration_minutes",
            name="Holding Duration",
            description="Total time from entry to exit in minutes",
            category="TEMPORAL", data_type="float", unit="minutes", nullable=True,
            produced_by="ExperienceNormalizer",
            depends_on=["entry_timestamp", "exit_timestamp"],
            normalizer="none", is_learnable=True, is_configurable=False,
            comparison_range=10080.0,
        ),
        FeatureDefinition(
            feature_id="temporal.bars_held",
            name="Bars Held",
            description="Number of timeframe bars from entry to exit",
            category="TEMPORAL", data_type="float", unit="bars", nullable=True,
            produced_by="ExperienceNormalizer",
            depends_on=["holding_duration_minutes", "timeframe"],
            normalizer="none", is_learnable=True, is_configurable=False,
            comparison_range=1000.0,
        ),

        # ── CALIBRATION ──
        FeatureDefinition(
            feature_id="calibration.entry_error_bps",
            name="Calibration Entry Error",
            description="Difference between shadow and live entry price in bps",
            category="CALIBRATION", data_type="float", unit="bps", nullable=True,
            produced_by="PositionManager",
            depends_on=["shadow_entry", "live_entry"],
            is_learnable=True, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="calibration.exit_error_bps",
            name="Calibration Exit Error",
            description="Difference between shadow and live exit price in bps",
            category="CALIBRATION", data_type="float", unit="bps", nullable=True,
            produced_by="PositionManager",
            depends_on=["shadow_exit", "live_exit"],
            is_learnable=True, is_configurable=False,
        ),
        FeatureDefinition(
            feature_id="calibration.fee_error_usdt",
            name="Calibration Fee Error",
            description="Difference between shadow and live fees in USDT",
            category="CALIBRATION", data_type="float", unit="USDT", nullable=True,
            produced_by="PositionManager",
            depends_on=["shadow_fees", "live_fees"],
            is_learnable=True, is_configurable=False,
        ),

        # ── CONTEXT ──
        FeatureDefinition(
            feature_id="context.symbol",
            name="Trading Symbol",
            description="The traded instrument symbol",
            category="CONTEXT", data_type="string", unit="", nullable=False,
            produced_by="Scanner", is_learnable=False, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),
        FeatureDefinition(
            feature_id="context.timeframe",
            name="Timeframe",
            description="The candle timeframe for this trade",
            category="CONTEXT", data_type="string", unit="", nullable=False,
            produced_by="Scanner", is_learnable=False, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),
        FeatureDefinition(
            feature_id="context.execution_mode",
            name="Execution Mode",
            description="Whether the trade was LIVE or SHADOW",
            category="CONTEXT", data_type="string", unit="", nullable=False,
            produced_by="TradeCoordinator", is_learnable=False, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),
        FeatureDefinition(
            feature_id="context.origin",
            name="Origin",
            description="How the trade originated (NORMAL, CONSTRAINT, MIRROR)",
            category="CONTEXT", data_type="string", unit="", nullable=False,
            produced_by="TradeCoordinator", is_learnable=False, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),
        FeatureDefinition(
            feature_id="context.anchor_symbol",
            name="Anchor Symbol",
            description="The anchor symbol used for correlation analysis",
            category="CONTEXT", data_type="string", unit="", nullable=False,
            produced_by="CorrelationEngine", is_learnable=False, is_configurable=False,
            comparison_type="categorical", distance_function="exact_match",
        ),
    ]

    return FeatureCatalog({f.feature_id: f for f in features})
