from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.models.learning.trade_experience import ConfigurationEntry, ConfigurationSnapshot


class ConfigurationItem(BaseModel, frozen=True):
    """A tunable parameter in the trading system.
    Every parameter the agent may adapt must be registered here.
    The agent cannot modify parameters not in the catalog."""
    parameter_id: str
    component: str
    description: str
    current_default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    data_type: str
    learnable: bool = False
    shadow_validation_required: bool = False
    live_validation_required: bool = False
    introduced_version: str = "1.0"


class ConfigurationCatalog:
    """Whitelist of every tunable parameter.
    Pure metadata — never references runtime objects."""
    def __init__(self, items: dict[str, ConfigurationItem]):
        self._items = items

    @property
    def version(self) -> str:
        return "1.0"

    def get_item(self, parameter_id: str) -> Optional[ConfigurationItem]:
        return self._items.get(parameter_id)

    def has_item(self, parameter_id: str) -> bool:
        return parameter_id in self._items

    def get_tunable_parameters(self) -> list[ConfigurationItem]:
        return [item for item in self._items.values() if item.learnable]

    def get_all_items(self) -> list[ConfigurationItem]:
        return list(self._items.values())

    @staticmethod
    def build_runtime_snapshot(
        values: dict[str, Any],
        catalog: ConfigurationCatalog,
        source: str = "hardcoded",
        config_version: str = "1.0",
    ) -> ConfigurationSnapshot:
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        entries: list[ConfigurationEntry] = []
        for param_id, value in values.items():
            entry_source = (
                source if catalog.has_item(param_id) else "unknown"
            )
            entries.append(ConfigurationEntry(
                parameter_id=param_id,
                value=value,
                source=entry_source,
                loaded_at=now,
                config_version=config_version,
            ))
        return ConfigurationSnapshot(items=entries)

    def __len__(self) -> int:
        return len(self._items)


def _build_default_catalog() -> ConfigurationCatalog:
    items: list[ConfigurationItem] = [
        # ── Risk ──
        ConfigurationItem(
            parameter_id="risk.max_positions",
            component="RiskManager",
            description="Maximum number of concurrently open positions",
            current_default=6, minimum=1, maximum=10,
            data_type="int", learnable=True,
            shadow_validation_required=True, live_validation_required=True,
        ),
        ConfigurationItem(
            parameter_id="risk.min_llm_confidence",
            component="RiskManager",
            description="Minimum LLM confidence to approve a candidate",
            current_default=0.3, minimum=0.0, maximum=1.0,
            data_type="float", learnable=True,
            shadow_validation_required=True, live_validation_required=True,
        ),
        ConfigurationItem(
            parameter_id="risk.max_live_exposure_usdt",
            component="RiskManager",
            description="Maximum total live exposure in USDT",
            current_default=25000.0, minimum=100.0, maximum=1000000.0,
            data_type="float", learnable=True,
            shadow_validation_required=False, live_validation_required=True,
        ),

        # ── Execution ──
        ConfigurationItem(
            parameter_id="execution.leverage",
            component="ExecutionService",
            description="Futures leverage for LIVE positions",
            current_default=5, minimum=1, maximum=125,
            data_type="int", learnable=True,
            shadow_validation_required=False, live_validation_required=True,
        ),
        ConfigurationItem(
            parameter_id="execution.sizing_mode",
            component="ExecutionService",
            description="Position sizing method (risk_pct, fixed_usdt)",
            current_default="fixed_usdt",
            data_type="string", learnable=True,
            shadow_validation_required=True, live_validation_required=True,
        ),
        ConfigurationItem(
            parameter_id="execution.sizing_value",
            component="ExecutionService",
            description="Value used by the sizing mode",
            current_default=5.0, minimum=0.1, maximum=100.0,
            data_type="float", learnable=True,
            shadow_validation_required=True, live_validation_required=True,
        ),
        ConfigurationItem(
            parameter_id="execution.stop_loss_pct",
            component="ExecutionService",
            description="Stop loss as percentage of entry price",
            current_default=0.98, minimum=0.90, maximum=0.99,
            data_type="float", learnable=True,
            shadow_validation_required=True, live_validation_required=True,
        ),
        ConfigurationItem(
            parameter_id="execution.take_profit_pct",
            component="ExecutionService",
            description="Take profit as percentage of entry price",
            current_default=1.04, minimum=1.01, maximum=2.0,
            data_type="float", learnable=True,
            shadow_validation_required=True, live_validation_required=True,
        ),
        ConfigurationItem(
            parameter_id="execution.spread_bps",
            component="ExecutionService",
            description="Estimated spread in basis points for SHADOW execution",
            current_default=2.0, minimum=0.0, maximum=10.0,
            data_type="float", learnable=True,
            shadow_validation_required=False, live_validation_required=False,
        ),
        ConfigurationItem(
            parameter_id="execution.fee_bps",
            component="ExecutionService",
            description="Estimated fee rate in basis points for SHADOW execution",
            current_default=4.0, minimum=0.0, maximum=10.0,
            data_type="float", learnable=True,
            shadow_validation_required=False, live_validation_required=False,
        ),
        ConfigurationItem(
            parameter_id="execution.slippage_bps",
            component="ExecutionService",
            description="Estimated slippage in basis points for SHADOW entry",
            current_default=3.0, minimum=0.0, maximum=10.0,
            data_type="float", learnable=True,
            shadow_validation_required=False, live_validation_required=False,
        ),

        # ── Position Manager ──
        ConfigurationItem(
            parameter_id="execution.trailing_stop_atr_mult",
            component="PositionManager",
            description="ATR multiplier for trailing stop updates",
            current_default=2.0, minimum=0.5, maximum=5.0,
            data_type="float", learnable=True,
            shadow_validation_required=True, live_validation_required=True,
        ),

        # ── Scanner ──
        ConfigurationItem(
            parameter_id="scanner.min_correlation",
            component="MarketScanner",
            description="Minimum absolute correlation coefficient for deterministic gate",
            current_default=0.6, minimum=0.0, maximum=1.0,
            data_type="float", learnable=True,
            shadow_validation_required=True, live_validation_required=True,
        ),
        ConfigurationItem(
            parameter_id="scanner.max_p_value",
            component="MarketScanner",
            description="Maximum p-value for correlation significance",
            current_default=0.05, minimum=0.0, maximum=1.0,
            data_type="float", learnable=True,
            shadow_validation_required=True, live_validation_required=True,
        ),
    ]

    return ConfigurationCatalog({item.parameter_id: item for item in items})
