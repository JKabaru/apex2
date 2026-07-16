import keyring
import os
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..learning.config_catalog import _build_default_catalog
from ..llm.registry import LLMRegistry, PROVIDER_MAP, KNOWN_MODELS
from ..main import save_config, _parse_value, KEYS_FILE
from .setup_wizard import encrypt_keys, decrypt_keys

console = Console()

_LLM_PROVIDER_CHOICES = {
    "llm.provider": list(PROVIDER_MAP.keys()),
    "llm.fallback_provider": [""] + list(PROVIDER_MAP.keys()),
}


def _get_nested(d: dict, key_path: str, default=None):
    parts = key_path.split(".")
    target = d
    for part in parts:
        if isinstance(target, dict) and part in target:
            target = target[part]
        else:
            return default
    return target


def _set_nested(d: dict, key_path: str, value):
    parts = key_path.split(".")
    target = d
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def _format_value(value):
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None or value == "":
        return "(not set)"
    return str(value)


def _resolve_llm_api_key(key_name: str) -> str | None:
    try:
        value = keyring.get_password("apex", key_name)
        if value:
            return value
    except Exception:
        pass

    if not os.path.exists(KEYS_FILE):
        return None

    try:
        passphrase = questionary.password(
            f"Enter encryption passphrase to unlock {key_name}:"
        ).ask()
        if not passphrase:
            return None
        with open(KEYS_FILE, "rb") as f:
            encrypted = f.read()
        keys = decrypt_keys(encrypted, passphrase)
        return keys.get(key_name)
    except Exception:
        return None


async def _fetch_models_for_param(config: dict, pid: str) -> list[str]:
    if pid == "llm.model":
        provider = _get_nested(config, "llm.provider", "")
        key_name = "llm_key"
        custom_base_url = _get_nested(config, "llm.custom_base_url", "")
    elif pid == "llm.fallback_model":
        provider = _get_nested(config, "llm.fallback_provider", "")
        key_name = "fallback_llm_key"
        custom_base_url = ""
    else:
        return []

    if not provider:
        return []

    api_key = _resolve_llm_api_key(key_name)

    if not api_key:
        console.print(f"[yellow]No API key stored for this provider. Please enter it now.[/]")
        api_key = await questionary.password(
            f"Enter API key for {provider}:"
        ).ask_async()
        if api_key:
            try:
                keyring.set_password("apex", key_name, api_key)
                console.print(f"[green]Key saved to OS keychain.[/]")
            except Exception:
                console.print("[yellow]Could not save key to OS keychain.[/]")

    if api_key:
        try:
            registry = LLMRegistry(
                provider=provider,
                api_key=api_key,
                custom_base_url=custom_base_url or None,
            )
            with console.status("[yellow]Fetching available models..."):
                models = await registry.fetch_available_models()
            if models:
                return models
        except Exception as e:
            console.print(f"[yellow]Could not fetch models: {e}[/]")

    known = KNOWN_MODELS.get(provider, [])
    if known:
        console.print("[yellow]Using known model list (API fetch unavailable).[/]")
    return known


_ENUM_CHOICES = {
    "execution.mode": ["testnet", "live"],
    "execution.sizing_mode": ["risk_pct", "fixed_usdt"],
    "output.mode": ["focus", "verbose"],
    "logging.level": ["DEBUG", "INFO", "WARNING", "ERROR"],
    "universe.quote_filter": ["USDT", "USDC", "all"],
    "binance.mode": ["testnet", "live"],
    "adaptive_feedback.mode": ["deterministic", "llm", "both"],
    **_LLM_PROVIDER_CHOICES,
}

_MULTI_SELECT_CHOICES = {
    "logging.focus": ["scanner", "execution", "learning", "reconciliation", "data", "system"],
}

_SECTION_ORDER = [
    "Risk",
    "Execution",
    "Scanner",
    "Binance",
    "Universe",
    "Feature Flags",
    "Data",
    "Correlation Engine",
    "Adaptive Feedback",
    "Evidence",
    "Output",
    "LLM Configuration",
]

_SECTION_COMPONENT_MAP = {
    "RiskManager": "Risk",
    "ExecutionService": "Execution",
    "PositionManager": "Execution",
    "MarketScanner": "Scanner",
    "BinanceMode": "Binance",
    "UniverseConfig": "Universe",
    "FeatureFlags": "Feature Flags",
    "Output": "Output",
    "LLMRegistry": "LLM Configuration",
    "AdaptiveFeedback": "Adaptive Feedback",
    "EvidencePolicy": "Evidence",
}

_OPERATIONAL_PARAMS = [
    ("execution.mode", "Mode", "ExecutionService", "LIVE or testnet execution", "string", None, None, "testnet"),
    ("shadow.enabled", "Shadow Trading", "FeatureFlags", "Enable shadow trading (paper trading alongside live)", "bool", None, None, True),
    ("shadow.mirror_enabled", "Mirror Trades", "FeatureFlags", "Mirror live trades in shadow executor", "bool", None, None, True),
    ("calibration.enabled", "Calibration", "FeatureFlags", "Enable position calibration", "bool", None, None, True),
    ("analytics.enabled", "Analytics", "FeatureFlags", "Enable analytics service", "bool", None, None, True),
    ("data.retention_days", "Data Retention (days)", "Data", "Number of days to retain OHLCV data", "int", 1, 365, 90),
    ("data.max_timeframe_m", "Max Timeframe (min)", "Data", "Maximum aggregation timeframe in minutes", "int", 1, 10080, 1440),
    ("correlation.rolling_window_candles", "Rolling Window (candles)", "Correlation Engine", "Number of candles in the rolling correlation window", "int", 100, 5000, 500),
    ("correlation.max_lag", "Max Lag", "Correlation Engine", "Maximum lag to evaluate in the correlation engine", "int", 1, 100, 15),
    ("correlation.base_half_life", "Base Half-Life (min)", "Correlation Engine", "Base exponential half-life in minutes", "int", 10, 500, 60),
    ("correlation.min_half_life", "Min Half-Life (min)", "Correlation Engine", "Minimum exponential half-life in minutes", "int", 5, 100, 15),
    ("correlation.max_half_life", "Max Half-Life (min)", "Correlation Engine", "Maximum exponential half-life in minutes", "int", 30, 1000, 180),
    ("correlation.acf_truncation_lag", "ACF Truncation Lag", "Correlation Engine", "Autocorrelation function truncation lag", "int", 1, 50, 10),
    ("correlation.alpha_crit", "Alpha Critical", "Correlation Engine", "Critical alpha for significance testing", "float", 0.001, 0.1, 0.01),
    ("correlation.update_buffer_candles", "Update Buffer (candles)", "Correlation Engine", "Buffer size for batched correlation updates", "int", 1, 100, 10),
    ("output.mode", "Output Mode", "Output", "Verbosity level (focus suppresses TICK/CANDLE/matrix)", "string", None, None, "focus"),
    ("logging.level", "Log Level", "Output", "Global minimum log level", "string", None, None, "INFO"),
    ("logging.focus", "Log Focus Categories", "Output", "Categories for full detail", "multi_select", None, None, "learning,execution"),
    ("logging.summary_interval_seconds", "Log Summary Interval", "Output", "Seconds between summarized log bursts for non-focus categories", "int", 15, 300, 60),
    ("llm.provider", "Primary Provider", "LLMRegistry", "Primary LLM provider", "string", None, None, "opencode"),
    ("llm.model", "Primary Model", "LLMRegistry", "Primary LLM model ID", "string", None, None, ""),
    ("llm.fallback_provider", "Fallback Provider", "LLMRegistry", "Fallback LLM provider for rate-limit failover", "string", None, None, ""),
    ("llm.fallback_model", "Fallback Model", "LLMRegistry", "Fallback LLM model ID", "string", None, None, ""),
    ("protection.max_retry_attempts", "Max Protection Retries", "ExecutionService", "Maximum retry attempts for protection placement before emergency close", "int", 1, 10, 3),
    ("protection.retry_interval_seconds", "Protection Retry Interval", "ExecutionService", "Seconds between protection retry attempts", "float", 1.0, 60.0, 5.0),
    ("protection.audit_interval_seconds", "Protection Audit Interval", "PositionManager", "Seconds between protection verification checks per position", "float", 10.0, 300.0, 60.0),
    ("binance.mode", "Mode", "BinanceMode", "LIVE or testnet Binance connection", "string", None, None, "testnet"),
    ("execution.max_risk_pct", "Max Risk %", "ExecutionService", "Max risk per trade as fraction of sizing value", "float", 0.001, 0.1, 0.04),
    ("risk.max_positions", "Max Positions", "RiskManager", "Maximum concurrent open positions", "int", 1, 20, 6),
    ("evidence.exact.max_confidence", "Exact Evidence Ceiling", "EvidencePolicy", "Max confidence for exact evidence matches", "float", 0.0, 1.0, 1.0),
    ("evidence.anchor.max_confidence", "Anchor Evidence Ceiling", "EvidencePolicy", "Max confidence for anchor proxy evidence", "float", 0.0, 1.0, 0.7),
    ("evidence.regime.max_confidence", "Regime Evidence Ceiling", "EvidencePolicy", "Max confidence for broad regime evidence", "float", 0.0, 1.0, 0.5),
    ("evidence.cold_start.max_confidence", "Cold Start Ceiling", "EvidencePolicy", "Max confidence when no evidence exists (cold start)", "float", 0.0, 1.0, 0.7),
    ("universe.anchors", "Anchors", "UniverseConfig", "Comma-separated anchor symbols", "string", None, None, "BTCUSDT,ETHUSDT,BNBUSDT"),
    ("universe.alternates", "Alternates", "UniverseConfig", "Comma-separated alternate symbols", "string", None, None, ""),
    ("universe.quote_filter", "Quote Filter", "MarketScanner", "Filter pairs by quote asset: USDT, USDC, or all", "string", None, None, "USDT"),
    ("adaptive_feedback.mode", "Feedback Mode", "AdaptiveFeedback", "How adaptive thresholds are adjusted: deterministic rules, LLM meta-cognition, or both", "string", None, None, "both"),
    ("adaptive_feedback.llm_interval_cycles", "LLM Interval (cycles)", "AdaptiveFeedback", "Run LLM meta-cognition every N reflection cycles", "int", 1, 100, 20),
    ("adaptive_feedback.reflection_interval_seconds", "Reflection Interval (s)", "AdaptiveFeedback", "Seconds between reflect→evolve→adapt→feedback cycles", "int", 30, 3600, 180),
    ("research.min_sample_size", "Min Sample Size", "Research", "Minimum evaluations before research report is COMPLETE", "int", 1, 1000, 30),
    ("research.min_intervention_evals", "Min Intervention Evals", "Research", "Minimum evaluations to discover interventions", "int", 1, 100, 5),
    ("research.min_simulation_evals", "Min Simulation Evals", "Research", "Minimum evaluations to simulate a recommendation", "int", 1, 100, 3),
    ("research.min_effect_size", "Min Effect Size", "Research", "Minimum Cohen's d effect size for intervention", "float", 0.001, 1.0, 0.2),
    ("research.cycle_interval", "Research Cycle Interval", "Research", "Run research every N reflection cycles", "int", 1, 100, 10),
    ("research.min_improvement", "Min Metric Improvement", "Research", "Minimum absolute metric improvement to justify a parameter change", "float", 0.0, 1.0, 0.01),
    ("research.metric_min_subgroup", "Metric Min Subgroup", "Research", "Minimum evaluations in a subgroup for metric computation", "int", 1, 100, 3),
    ("research.metric_min_losses", "Metric Min Losses", "Research", "Minimum losing trades for avg_loss metric computation", "int", 1, 50, 2),
    ("research.pattern_min_evals", "Pattern Min Evals", "Research", "Minimum valid evaluations for ANY pattern detection", "int", 1, 1000, 10),
    ("research.pattern_high_conf_threshold", "High Conf Threshold", "Research", "Confidence boundary between high/low confidence (0-1)", "float", 0.0, 1.0, 0.5),
    ("research.pattern_min_high_conf_evals", "Min High Conf Evals", "Research", "Minimum high-confidence evals for overconfidence detection", "int", 1, 1000, 10),
    ("research.pattern_overconfidence_high_gap", "Overconf High Gap", "Research", "Gap threshold for HIGH severity overconfidence finding", "float", 0.0, 1.0, 0.30),
    ("research.pattern_overconfidence_medium_gap", "Overconf Med Gap", "Research", "Gap threshold for MEDIUM severity overconfidence finding", "float", 0.0, 1.0, 0.15),
    ("research.pattern_min_low_conf_evals", "Min Low Conf Evals", "Research", "Minimum low-confidence evals for underconfidence detection", "int", 1, 1000, 10),
    ("research.pattern_underconfidence_wr", "Underconf WR", "Research", "Win rate threshold for underconfidence finding", "float", 0.0, 1.0, 0.60),
    ("research.pattern_min_side_evals", "Min Side Evals", "Research", "Minimum evals per side (BUY/SELL) for bias detection", "int", 1, 100, 5),
    ("research.pattern_side_imbalance", "Side Imbalance", "Research", "Win rate difference threshold for LONG_BIAS finding", "float", 0.0, 1.0, 0.20),
    ("research.pattern_min_tier_evals", "Min Tier Evals", "Research", "Minimum evals per evidence tier (EXACT/COLD_START)", "int", 1, 1000, 5),
    ("research.pattern_stop_loss_rate", "Stop Loss Rate", "Research", "Stop-loss exit rate threshold for finding", "float", 0.0, 1.0, 0.40),
    ("research.pattern_min_duration_evals", "Min Duration Evals", "Research", "Minimum short-hold evals for holding time mismatch", "int", 1, 100, 5),
    ("research.pattern_short_hold_wr", "Short Hold WR", "Research", "Win rate threshold for short-hold finding (< this triggers)", "float", 0.0, 1.0, 0.40),
    ("research.observation_calibration_drift_threshold", "Cal Drift Obs", "Research", "Calibration error threshold for CALIBRATION_DRIFT observation", "float", 0.0, 1.0, 0.15),
    ("research.observation_calibration_severe_threshold", "Cal Severe Obs", "Research", "Cal error above this marks CALIBRATION_DRIFT as HIGH severity", "float", 0.0, 1.0, 0.25),
    ("research.observation_calibration_min_sample", "Cal Min Sample", "Research", "Minimum calibration bucket size for CALIBRATION_DRIFT finding", "int", 1, 1000, 10),
    ("research.observation_regime_min_sample", "Regime Min Sample", "Research", "Minimum regime sample size for REGIME_INEFFECTIVENESS finding", "int", 1, 1000, 10),
    ("research.observation_low_win_rate", "Low WR Obs", "Research", "Win rate below this triggers LOW_WIN_RATE observation", "float", 0.0, 1.0, 0.4),
    ("research.observation_high_win_rate", "High WR Obs", "Research", "Win rate above this triggers HIGH_WIN_RATE observation", "float", 0.0, 1.0, 0.7),
    ("research.observation_small_sample_threshold", "Small Sample Obs", "Research", "Sample below this triggers SMALL_SAMPLE observation", "int", 1, 10000, 100),
]


async def _edit_api_keys() -> None:
    console.print(
        Panel.fit(
            Text.assemble(
                ("API Keys", "bold yellow"),
                "\n",
                ("Update your exchange and LLM API keys below. ", "dim"),
                ("Current keys are not shown for security.", "dim"),
            ),
            border_style="yellow",
            padding=(1, 2),
        )
    )

    keys_to_update = ["binance_key", "binance_secret", "llm_key", "fallback_llm_key"]
    updated = {}

    for key_name in keys_to_update:
        label = key_name.replace("_", " ").title()
        new_val = await questionary.password(
            f"New {label} (leave blank to keep current):"
        ).ask_async()
        if new_val:
            updated[key_name] = new_val

    if not updated:
        console.print("[yellow]No keys updated.[/]")
        return

    try:
        for key_name, value in updated.items():
            keyring.set_password("apex", key_name, value)
        console.print(f"[green]\u2713 {len(updated)} key(s) saved to OS keychain.[/]")
    except Exception:
        console.print("[yellow]OS keychain unavailable. Falling back to encrypted file...[/]")
        passphrase = await questionary.password(
            "Set an encryption passphrase for local key storage:",
            validate=lambda val: len(val) >= 8 or "Passphrase must be at least 8 characters.",
        ).ask_async()
        confirm = await questionary.password("Confirm encryption passphrase:").ask_async()
        if passphrase != confirm:
            console.print("[red]Passphrases do not match. Keys not saved.[/]")
            return

        try:
            payload = {}
            existing_keys = {}
            for k in keys_to_update:
                try:
                    existing_keys[k] = keyring.get_password("apex", k) or ""
                except Exception:
                    existing_keys[k] = ""
            payload.update(existing_keys)
            payload.update(updated)
            encrypted = encrypt_keys(payload, passphrase)
            with open("keys.enc", "wb") as f:
                f.write(encrypted)
            console.print("[green]\u2713 Keys encrypted and saved to keys.enc[/]")
        except Exception as e:
            console.print(f"[red]Failed to save keys: {e}[/]")


def _build_param_list(config: dict) -> dict[str, list]:
    catalog = _build_default_catalog()
    sections: dict[str, list] = {}

    for item in catalog.get_all_items():
        current = _get_nested(config, item.parameter_id, item.current_default)
        section = _SECTION_COMPONENT_MAP.get(item.component, item.component)
        if section not in sections:
            sections[section] = []
        sections[section].append((
            item.parameter_id,
            item.parameter_id.split(".")[-1].replace("_", " ").title(),
            item.description,
            item.data_type,
            item.minimum,
            item.maximum,
            current,
        ))

    for pid, label, component, desc, dtype, mini, maxi, default in _OPERATIONAL_PARAMS:
        current = _get_nested(config, pid, default)
        section = _SECTION_COMPONENT_MAP.get(component, component)
        if section not in sections:
            sections[section] = []
        sections[section].append((pid, label, desc, dtype, mini, maxi, current))

    return sections


async def run_config_editor(config: dict, config_path: str) -> None:
    sections = _build_param_list(config)

    sorted_sections = sorted(
        sections.keys(),
        key=lambda s: _SECTION_ORDER.index(s) if s in _SECTION_ORDER else 999,
    )

    console.print(
        Panel.fit(
            Text.assemble(
                ("Configuration Editor", "bold cyan"),
                "\n",
                ("Modify settings below. Changes are saved on exit.", "dim"),
            ),
            border_style="cyan",
            padding=(1, 2),
        )
    )

    while True:
        section_choices = [
            questionary.Choice(title=f"  {sname} ({len(sections[sname])} settings)", value=sname)
            for sname in sorted_sections
        ]
        section_choices.append(questionary.Choice(title="  API Keys", value="__api_keys__"))
        section_choices.append(questionary.Choice(title="  [Save & Exit]", value="__exit__"))

        section = await questionary.select(
            "Select a section to configure:",
            choices=section_choices,
        ).ask_async()

        if section == "__exit__":
            break

        if section == "__api_keys__":
            await _edit_api_keys()
            continue

        params = sections[section]
        while True:
            param_choices = [
                questionary.Choice(
                    title=f"  {label} [{_format_value(current)}]",
                    value=(label, current),
                )
                for _, label, _, _, _, _, current in params
            ]
            param_choices.append(questionary.Choice(title="  [Back to sections]", value="__back__"))

            selected = await questionary.select(
                f"{section} \u2014 Select a parameter to edit:",
                choices=param_choices,
            ).ask_async()

            if selected == "__back__":
                break

            selected_label, _ = selected
            idx = next(i for i, p in enumerate(params) if p[1] == selected_label)
            pid, label, desc, dtype, mini, maxi, current = params[idx]

            new_value = None

            if dtype == "bool":
                new_value = await questionary.confirm(
                    f"{label}: {desc}",
                    default=bool(current),
                ).ask_async()
            elif pid in _ENUM_CHOICES:
                choices = _ENUM_CHOICES[pid]
                default_choice = str(current) if str(current) in choices else choices[0]
                new_value = await questionary.select(
                    f"{label}: {desc}",
                    choices=choices,
                    default=default_choice,
                ).ask_async()
            elif pid in _MULTI_SELECT_CHOICES:
                all_options = _MULTI_SELECT_CHOICES[pid]
                pre_selected = current if isinstance(current, list) else [c.strip() for c in current.split(",") if c.strip()]
                pre_selected = [c for c in pre_selected if c in all_options]
                selected = await questionary.checkbox(
                    f"{label}: {desc}",
                    choices=[questionary.Choice(opt, value=opt, checked=opt in pre_selected) for opt in all_options],
                ).ask_async()
                if selected is None:
                    console.print("[yellow]Value unchanged.[/]")
                    continue
                new_value = selected
            elif pid in ("llm.model", "llm.fallback_model"):
                models = await _fetch_models_for_param(config, pid)
                if models:
                    model_choices = [questionary.Choice(m, value=m) for m in models[:100]]
                    default_choice = str(current) if str(current) in models else models[0]
                    new_value = await questionary.select(
                        f"{label}: {desc}",
                        choices=model_choices,
                        default=default_choice,
                    ).ask_async()
                    if not new_value:
                        console.print("[yellow]Value unchanged.[/]")
                        continue
                else:
                    console.print(
                        f"[yellow]No models available for {pid.split('.')[-1].replace('_', ' ').title()}. "
                        f"Ensure the provider and API key are configured correctly.[/]"
                    )
                    continue
            else:
                prompt = f"{label}: {desc}"
                if mini is not None and maxi is not None:
                    prompt += f" (range: {mini} \u2013 {maxi})"
                display_default = ", ".join(str(v) for v in current) if isinstance(current, list) else _format_value(current)
                raw = await questionary.text(
                    prompt,
                    default=display_default,
                ).ask_async()

                if raw is None or raw.strip() == "":
                    console.print("[yellow]Value unchanged.[/]")
                    continue

                if isinstance(current, list):
                    new_value = [item.strip() for item in raw.split(",") if item.strip()]
                else:
                    try:
                        new_value = _parse_value(raw.strip())
                    except Exception:
                        console.print(f"[red]Invalid value: {raw}[/]")
                        continue

            if new_value is not None:
                if mini is not None and maxi is not None and isinstance(new_value, (int, float)):
                    if new_value < mini or new_value > maxi:
                        console.print(
                            f"[red]Value must be between {mini} and {maxi} (got {new_value})[/]"
                        )
                        continue

                _set_nested(config, pid, new_value)
                params[idx] = (pid, label, desc, dtype, mini, maxi, new_value)
                console.print(f"[green]\u2713 {label} updated to {_format_value(new_value)}[/]")

    save_config(config_path, config)
    console.print("[green]Configuration saved to config.toml[/]")
