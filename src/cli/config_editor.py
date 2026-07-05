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
    **_LLM_PROVIDER_CHOICES,
}

_SECTION_ORDER = [
    "Risk",
    "Execution",
    "Scanner",
    "Feature Flags",
    "Data",
    "Correlation Engine",
    "Output",
    "LLM Configuration",
]

_SECTION_COMPONENT_MAP = {
    "RiskManager": "Risk",
    "ExecutionService": "Execution",
    "PositionManager": "Execution",
    "MarketScanner": "Scanner",
    "FeatureFlags": "Feature Flags",
    "Output": "Output",
    "LLMRegistry": "LLM Configuration",
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
    ("llm.provider", "Primary Provider", "LLMRegistry", "Primary LLM provider", "string", None, None, "opencode"),
    ("llm.model", "Primary Model", "LLMRegistry", "Primary LLM model ID", "string", None, None, ""),
    ("llm.fallback_provider", "Fallback Provider", "LLMRegistry", "Fallback LLM provider for rate-limit failover", "string", None, None, ""),
    ("llm.fallback_model", "Fallback Model", "LLMRegistry", "Fallback LLM model ID", "string", None, None, ""),
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
                raw = await questionary.text(
                    prompt,
                    default=_format_value(current),
                ).ask_async()

                if raw is None or raw.strip() == "":
                    console.print("[yellow]Value unchanged.[/]")
                    continue

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
