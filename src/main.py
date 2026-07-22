import argparse
import asyncio
import json
import os
import sys
import traceback
import duckdb

import keyring
import questionary
import rich
from rich.panel import Panel
from rich.text import Text

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from .api.binance_client import BinanceClient, BinanceClientError
from .cli.setup_wizard import run_setup_wizard, decrypt_keys
from .data.aggregator import Aggregator
from .data.ingestor import Ingestor
from .data.ws_ingestor import WebSocketIngestor
from .llm.registry import LLMRegistry
from .utils.logger import init_logging, get_logger
from .correlation.engine import CorrelationEngine
from .correlation.matrix_store import CorrelationMatrixStore
from .correlation.dashboard import render_live_matrix

from .core.events import EventBus
from .db.portfolio_store import PortfolioStore, SchemaMismatchError
from .services.portfolio_manager import PortfolioManager
from .services.llm_scheduler import LLMScheduler
from .services.market_context import MarketContextService
from .services.risk_manager import RiskManager
from .db.tim_store import TimStore
from .db.write_coordinator import DatabaseWriteCoordinator
from .models.tim.enums import TIMMode
from .services.execution import ExecutionService, LiveExecutor, VirtualExecutor
from .services.position_manager import PositionManager
from .services.scanner import MarketScanner
from .services.trade_coordinator import TradeCoordinator
from .services.reasoning_coordinator import ReasoningCoordinator
from .services.reflection_engine import ReflectionEngine
from .services.belief_evolution import BeliefEvolutionEngine
from .services.profile_adapter import ProfileAdapter
from .services.adaptive_memory_tuner import AdaptiveMemoryTuner
from .services.adaptive_feedback import AdaptiveFeedbackEngine
from .services.evidence_resolver import EvidenceResolver
from .services.evidence_policy import configure_from_config
from .retrieval.pipeline import RetrievalPipeline
from .retrieval.weights import SimilarityWeights
from .intelligence.pipeline import ExperienceIntelligencePipeline
from .engine.output_mode import set_mode as set_verbose_mode, get_mode as get_verbose_mode
from .services.analytics_service import AnalyticsService
from .services.metrics_service import MetricsService
from .services.system_recovery import SystemRecoveryService
from .models.learning.trade_experience import PositionSnapshot
from .models.learning.corpus_metadata import CorpusMetadata
from .learning.extractor import ExperienceExtractor
from .learning.validator import ExperienceValidator
from .learning.normalizer import ExperienceNormalizer
from .learning.pipeline import LearningPipeline, MetadataResolver
from .learning.feature_catalog import _build_default_catalog as build_feature_catalog
from .learning.config_catalog import _build_default_catalog as build_config_catalog
from .learning.provenance import _build_default_registry as build_provenance_registry
from .learning.importance_scorer import ImportanceScorer
from .learning.timeline_manager import TimelineManager
from .learning.pattern_detector import PatternDetector
from .learning.hypothesis_extractor import HypothesisExtractor
from .learning.knowledge_promoter import KnowledgePromoter
from .models.learning.hypothesis import HypothesisStatus
from .learning.observation_compressor import ObservationCompressor
from .learning.prediction_lifecycle import PredictionLifecycle
from .learning.observation_ingestor import ObservationIngestor
from .storage.learning.learning_corpus import CandidateRejectionError, LearningCorpus, VerificationError
from .evaluation.store import DecisionCaptureStore
from .evaluation.engine import DecisionEvaluationEngine
from .evaluation.storage import EvaluationCorpus
from .recommendations.store import ConfigurationStore
from .recommendations.lifecycle import merge_adaptive_config, process_adaptive_decisions
from .recommendations.engine import RecommendationEngine
from .research.storage import ResearchCorpusStore
from .research.pipeline import ResearchPipeline
from .recommendations.models import LearningPolicy
from .models.session import TradingSession
from .services.session_manager import SessionManager
from .operator.cli import StartupCLI, write_checkpoint, read_checkpoint

CONFIG_PATH = "config.toml"
KEYS_FILE = "keys.enc"
AGENT_PARAMS_PATH = "data/agent_params.json"

AGENT_PARAMS_DEFAULTS = {
    "rolling_window_candles": 500,
    "max_lag": 15,
    "base_half_life": 60,
    "min_half_life": 15,
    "max_half_life": 180,
    "acf_truncation_lag": 10,
    "alpha_crit": 0.01,
    "update_buffer_candles": 10,
}


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _parse_value(raw: str):
    """Infer int, float, bool, or keep as string for CLI --config-set values."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _dict_to_toml(d: dict, parent_key: str = "") -> list[str]:
    """Recursively serialize a nested dict to TOML lines."""
    lines: list[str] = []
    for key, value in d.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            lines.append(f"[{full_key}]")
            lines.extend(_dict_to_toml(value, full_key))
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        elif isinstance(value, list):
            items = ", ".join(
                f'"{v}"' if isinstance(v, str) else str(v).lower() if isinstance(v, bool) else str(v)
                for v in value
            )
            lines.append(f"{key} = [{items}]")
        elif value is not None:
            lines.append(f"{key} = {value}")
    return lines


def save_config(path: str, config: dict) -> None:
    """Write config dict back to config.toml."""
    lines = _dict_to_toml(config)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def get_api_keys(passphrase_cache: list) -> tuple:
    binance_key = binance_secret = llm_key = fallback_llm_key = None

    try:
        binance_key = keyring.get_password("apex", "binance_key")
        binance_secret = keyring.get_password("apex", "binance_secret")
        llm_key = keyring.get_password("apex", "llm_key")
        fallback_llm_key = keyring.get_password("apex", "fallback_llm_key")
        if binance_key and binance_secret:
            return binance_key, binance_secret, llm_key, fallback_llm_key
    except Exception:
        pass

    if not os.path.exists(KEYS_FILE):
        raise RuntimeError(
            "No API keys found in OS keychain and no keys.enc file present. "
            "Please delete config.toml and re-run the setup wizard."
        )

    if not passphrase_cache:
        try:
            pp = questionary.password("Enter encryption passphrase to unlock keys:").ask()
        except Exception as e:
            if "No Windows console found" in str(e):
                raise RuntimeError(
                    "Interactive prompt requires a real terminal. "
                    "Run from cmd.exe, Windows Terminal, or VS Code terminal."
                ) from e
            raise RuntimeError(f"Failed to prompt for passphrase: {e}") from e
        if not pp:
            raise RuntimeError("Passphrase is required to decrypt API keys.")
        passphrase_cache.append(pp)

    try:
        with open(KEYS_FILE, "rb") as f:
            encrypted = f.read()
        keys = decrypt_keys(encrypted, passphrase_cache[0])
        return (
            keys["binance_key"],
            keys["binance_secret"],
            keys.get("llm_key"),
            keys.get("fallback_llm_key"),
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to decrypt keys.enc: {e}. The file may be corrupted or the passphrase is wrong."
        ) from e


def init_agent_params():
    if not os.path.exists(AGENT_PARAMS_PATH):
        with open(AGENT_PARAMS_PATH, "w") as f:
            json.dump(AGENT_PARAMS_DEFAULTS, f, indent=2)
        log = get_logger("main")
        log.info("Agent params file created", path=AGENT_PARAMS_PATH)


def clear_keyring():
    for service in ("binance_key", "binance_secret", "llm_key", "fallback_llm_key"):
        try:
            keyring.delete_password("apex", service)
        except Exception:
            pass


async def main():
    if not os.path.exists(CONFIG_PATH):
        rich.print("[yellow]No configuration found. Launching setup wizard...[/]")
        try:
            await run_setup_wizard()
        except Exception as e:
            if "No Windows console found" in str(e):
                rich.print(
                    "[bold red]Interactive setup requires a real terminal.[/]\n"
                    "Please run this from [bold]cmd.exe[/], [bold]Windows Terminal[/], "
                    "or the [bold]VS Code integrated terminal[/].\n"
                    "PowerShell ISE and some non-interactive shells are not supported."
                )
            else:
                rich.print(f"[bold red]Setup wizard failed: {e}[/]")
            sys.exit(1)
        rich.print("[green]Setup complete. Restarting...[/]")
        return await main()

    try:
        action = await questionary.select(
            "Configuration found. What would you like to do?",
            choices=[
                questionary.Choice("Start APEX Engine", value="start"),
                questionary.Choice("Re-run Setup Wizard (Change keys/coins/mode)", value="rerun"),
                questionary.Choice("View/Edit Configuration", value="config"),
                questionary.Choice("Emergency State Reset (Close all positions)", value="reset"),
            ],
        ).ask_async()
    except Exception:
        rich.print("[yellow]Non-interactive environment detected. Auto-starting engine.[/]")
        action = "start"

    if action == "rerun":
        os.remove(CONFIG_PATH)
        clear_keyring()
        rich.print("[yellow]Cleared existing configuration. Launching setup wizard...[/]")
        await run_setup_wizard()
        rich.print("[green]Setup complete. Restarting...[/]")
        return await main()

    config = load_config(CONFIG_PATH)

    if action == "config":
        from .cli.config_editor import run_config_editor
        await run_config_editor(config, CONFIG_PATH)
        rich.print("[green]Configuration updated. Restarting...[/]")
        return await main()

    cfg_mode = config.get("output", {}).get("mode", "focus")
    set_verbose_mode(cfg_mode)

    init_agent_params()

    passphrase_cache = []
    try:
        binance_key, binance_secret, llm_key, fallback_llm_key = get_api_keys(passphrase_cache)
    except RuntimeError as e:
        rich.print(f"[red]{e}[/]")
        sys.exit(1)

    init_logging(config)
    configure_from_config(config)
    log = get_logger("main")

    # ── Emergency State Reset (before LLM connection) ──
    if action == "reset":
        from .services.reset_service import EmergencyResetService
        rich.print("[yellow]Starting Emergency State Reset...[/]")
        try:
            client = BinanceClient(
                mode=config["binance"]["mode"],
                api_key=binance_key,
                api_secret=binance_secret,
            )
            await client.sync_time()
            await EmergencyResetService.full_data_reset(client, log)
            await client.close()
            rich.print("[bold green]Emergency Reset Complete. Positions closed. State files wiped. Memory preserved.[/]")
        except Exception as e:
            log.error("Emergency reset failed", error=str(e), traceback=traceback.format_exc())
            rich.print(f"[bold red]Reset failed: {e}[/]")
        rich.print("[green]You can now start the engine with a clean state.[/]")
        return await main()

    log.info("APEX starting", mode=config["binance"]["mode"])
    exec_mode = config.get("execution", {}).get("mode", "MISSING")
    log.info("Execution mode initialized", mode=exec_mode)

    llm_provider = config["llm"]["provider"]
    model_id = config["llm"].get("model", "")
    custom_base_url = config["llm"].get("custom_base_url", "")
    fallback_provider = config["llm"].get("fallback_provider", "")
    fallback_model = config["llm"].get("fallback_model", "")

    # --- Phase 5.0: Configuration Store (profile traceability) ---
    config_store = ConfigurationStore()
    session_manager = SessionManager()

    # ── Phase 5.0: Configuration Store + Profile Bootstrap ──
    from src.recommendations.models import ConfigurationProfile as _ConfigProfile

    current_active = config_store.get_active_profile()
    profiles = config_store.list_profiles()

    if not profiles:
        default_profile = _ConfigProfile(
            name="Default",
            system_generated=True,
            description="Auto-generated from config.toml on first startup",
            resolved_configuration={},
        )
        config_store.save_profile(default_profile)
        config_store.activate_profile(default_profile.profile_id, activated_by="system")
        profiles = [default_profile]
        current_active = default_profile

    # ── Phase 5.5: Startup safety net ──
    # If no profile is active (e.g. after a crash between deactivate/activate),
    # recover from the checkpoint file before showing the dashboard.
    if current_active is None and profiles:
        recovered = read_checkpoint()
        if recovered:
            recovered_pid, recovered_wid = recovered
            if recovered_pid and any(p.profile_id == recovered_pid for p in profiles):
                config_store.activate_profile(recovered_pid, activated_by="system")
                current_active = config_store.get_active_profile()
                log.info("STARTUP_CHECKPOINT_RECOVER", profile_id=recovered_pid, _force_log=True)
                ws = config_store.ensure_profile_workspace(recovered_pid, current_active.name)
                config_store.switch_workspace(ws.workspace_id)
                log.info("STARTUP_CHECKPOINT_WS_RECOVER", workspace_id=recovered_wid, _force_log=True)
            elif recovered_wid:
                ws_list = config_store.list_workspaces()
                if any(w.workspace_id == recovered_wid for w in ws_list):
                    config_store.switch_workspace(recovered_wid)
                    log.info("STARTUP_CHECKPOINT_WS_RECOVER", workspace_id=recovered_wid, _force_log=True)

    # ── Phase 5.5: Unified Startup Dashboard ──
    workspaces = config_store.list_workspaces()
    active_ws = config_store.get_active_workspace()
    current_active = config_store.get_active_profile()

    startup_selection = await StartupCLI.run_startup_dashboard(
        profiles=profiles,
        workspaces=workspaces,
        current_active=current_active,
        active_workspace=active_ws,
        config_store=config_store,
    )

    if startup_selection is None:
        log.info("User quit from startup dashboard")
        config_store.close()
        return

    chosen_profile_id = startup_selection.profile_id
    chosen_ws_id = startup_selection.workspace_id

    # Activate chosen profile if different
    active_profile = config_store.get_profile(chosen_profile_id)
    if active_profile and (not current_active or chosen_profile_id != current_active.profile_id):
        config_store.activate_profile(chosen_profile_id, activated_by="operator")
        current_active = active_profile
        # Switch to the profile's linked workspace
        ws = config_store.ensure_profile_workspace(chosen_profile_id, active_profile.name)
        config_store.switch_workspace(ws.workspace_id)
        chosen_ws_id = ws.workspace_id
        log.info("Memory workspace switched to profile-linked workspace", workspace_id=ws.workspace_id, profile_id=chosen_profile_id, _force_log=True)

    active_profile = current_active or profiles[0]

    # Switch workspace if different (fallback for manually picked workspace)
    if chosen_ws_id and (not active_ws or chosen_ws_id != active_ws.workspace_id):
        config_store.switch_workspace(chosen_ws_id)
        log.info("Memory workspace switched", workspace_id=chosen_ws_id)
    active_ws = config_store.get_active_workspace()

    # ── Phase 5.3: Merge adaptive parameters into runtime config ──
    active_adaptations = config_store.get_active_adaptive_versions(active_profile.profile_id)
    param_defs = config_store.get_all_adaptive_parameters()
    if active_adaptations:
        config = merge_adaptive_config(config, active_adaptations, param_defs)
        log.info(
            "Adaptive parameters merged into config",
            parameter_count=len(active_adaptations),
            profile_id=active_profile.profile_id,
        )

    # ── Phase 5.3: Process pending adaptive decisions ──
    decision_results = process_adaptive_decisions(config_store, active_profile.profile_id)
    if decision_results:
        log.info("Adaptive decisions processed at startup", results=decision_results)

    config_hash = TradingSession.compute_config_hash(active_profile.resolved_configuration)

    session = session_manager.start_session(
        profile_id=active_profile.profile_id,
        config_hash=config_hash,
    )
    active_session_id = session.session_id

    log.info(
        "Trading session initialized",
        session_id=active_session_id,
        profile_id=active_profile.profile_id,
        profile_name=active_profile.name,
        adaptive_overrides=[f"{k}={v.value}" for k, v in active_adaptations.items()],
    )

    # --- LLM + Binance connection checks ---
    llm_ok = False
    registry = None
    fallback_registry = None
    if not llm_provider:
        rich.print("[yellow][SKIP] LLM not configured (skipped during setup). Configure later via wizard.[/]")
        llm_ok = True
    else:
        try:
            log.info("Verifying LLM connection", provider=llm_provider, model=model_id or "auto")
            registry = LLMRegistry(
                provider=llm_provider,
                api_key=llm_key,
                custom_base_url=custom_base_url,
                model_id=model_id or None,
            )
            await registry.verify_connection()
            if registry.is_degraded():
                rich.print("[yellow][WARN] Primary LLM starting in degraded mode (rate-limited).[/]")
            else:
                rich.print("[bold green][OK] LLM Connected[/]")
            llm_ok = True
        except Exception as e:
            log.error("LLM connection failed", error=str(e), traceback=traceback.format_exc())
            rich.print(f"[bold red][FAIL] LLM Failed: {e}[/]")
            registry._llm_degraded = True

        if fallback_provider and fallback_llm_key:
            try:
                log.info(
                    "Verifying fallback LLM connection",
                    provider=fallback_provider,
                    model=fallback_model or "auto",
                )
                fallback_registry = LLMRegistry(
                    provider=fallback_provider,
                    api_key=fallback_llm_key,
                    model_id=fallback_model or None,
                )
                await fallback_registry.verify_connection()
                if fallback_registry.is_degraded():
                    rich.print("[yellow][WARN] Fallback LLM is rate-limited.[/]")
                else:
                    rich.print("[bold green][OK] Fallback LLM Connected[/]")
                llm_ok = True
            except Exception as e:
                log.warning(
                    "Fallback LLM connection failed, continuing without fallback",
                    error=str(e),
                )
                rich.print(f"[yellow][WARN] Fallback LLM unavailable: {e}[/]")
                fallback_registry = None

    binance_ok = False
    balance_str = "N/A"
    client = None
    try:
        log.info("Verifying Binance Futures connection")
        client = BinanceClient(
            mode=config["binance"]["mode"],
            api_key=binance_key,
            api_secret=binance_secret,
        )
        await client.sync_time()
        account_info = await client.get_account_info()
        for asset in account_info.get("assets", []):
            if asset["asset"] == "USDT":
                balance_str = asset.get("walletBalance", "N/A")
                break
        rich.print(f"[bold green][OK] Binance Connected. Balance: {balance_str} USDT[/]")
        binance_ok = True
    except (BinanceClientError, Exception) as e:
        log.error("Binance connection failed", error=str(e), traceback=traceback.format_exc())
        rich.print(f"[bold red][FAIL] Binance Failed: {e}[/]")

    if not (llm_ok and binance_ok):
        log.warning("One or more connections failed. Exiting.")
        sys.exit(1)

    anchors = config["universe"].get("anchors", [])
    alternates = config["universe"].get("alternates", [])

    quote_filter = config["universe"].get("quote_filter", "USDT")
    if quote_filter != "all":
        anchors = [s for s in anchors if s.endswith(quote_filter)]
        alternates = [s for s in alternates if s.endswith(quote_filter)]

    all_symbols = anchors + alternates

    if not all_symbols:
        log.warning("No symbols configured in universe.anchors / universe.alternates. Exiting.")
        sys.exit(1)

    # --- Phase 1/2 Data Infrastructure ---
    ingestor = Ingestor(mode=config["binance"]["mode"], binance_client=client)
    max_timeframe_m = config.get("data", {}).get("max_timeframe_m", 1440)
    aggregator = Aggregator(max_timeframe_m=max_timeframe_m)

    correlation_cfg = config.get("correlation", {})
    matrix_store = CorrelationMatrixStore()
    correlation_engine = CorrelationEngine(
        rolling_window_candles=correlation_cfg.get("rolling_window_candles", 500),
        max_lag=correlation_cfg.get("max_lag", 15),
        base_half_life=float(correlation_cfg.get("base_half_life", 60)),
        min_half_life=float(correlation_cfg.get("min_half_life", 15)),
        max_half_life=float(correlation_cfg.get("max_half_life", 180)),
        acf_truncation_lag=correlation_cfg.get("acf_truncation_lag", 10),
        alpha_crit=float(correlation_cfg.get("alpha_crit", 0.01)),
        update_buffer_candles=correlation_cfg.get("update_buffer_candles", 10),
        anchors=anchors,
        alternates=alternates,
        max_timeframe_m=max_timeframe_m,
    )

    log.info("Seeding correlation engine from historical data")
    try:
        import duckdb
        duck_conn = duckdb.connect("data/ohlcv.duckdb")
        for sym in all_symbols:
            rows = duck_conn.execute(
                "SELECT close FROM ohlcv_1m WHERE symbol = ? ORDER BY open_time ASC",
                [sym],
            ).fetchall()
            closes = [r[0] for r in rows]
            if closes:
                correlation_engine.seed_history(sym, closes)
        duck_conn.close()
    except Exception as e:
        log.error("Failed to seed correlation engine", error=str(e), traceback=traceback.format_exc())

    candle_queue: asyncio.Queue = asyncio.Queue()

    ws_ingestor = WebSocketIngestor(
        symbols=all_symbols,
        mode=config["binance"]["mode"],
        candle_queue=candle_queue,
    )

    # --- Phase 4.2 Core Services ---
    event_bus = EventBus()
    try:
        portfolio_store = PortfolioStore()
        portfolio_store.create_schema()
        portfolio_mgr = PortfolioManager(portfolio_store, event_bus)
    except KeyboardInterrupt:
        log.error("Startup interrupted")
        sys.exit(1)
    except SchemaMismatchError as e:
        log.error("Schema mismatch detected", actual=e.actual, expected=e.expected)
        answer = input(
            f"Database schema mismatch (expected {e.expected} columns, "
            f"found {e.actual}). Delete and rebuild? (y/n): "
        ).strip().lower()
        if answer == "y":
            portfolio_store = PortfolioStore.rebuild()
            portfolio_store.create_schema()
            portfolio_mgr = PortfolioManager(portfolio_store, event_bus)
        else:
            log.error("Aborting due to schema mismatch")
            sys.exit(1)
    except KeyboardInterrupt:
        log.error("Startup interrupted")
        sys.exit(1)
    except Exception as e:
        log.error("Database error during initialization", error=str(e))
        answer = input(
            f"Database error: {e}. Delete and rebuild? (y/n): "
        ).strip().lower()
        if answer == "y":
            portfolio_store = PortfolioStore.rebuild()
            portfolio_store.create_schema()
            portfolio_mgr = PortfolioManager(portfolio_store, event_bus)
        else:
            log.error("Aborting due to database error")
            sys.exit(1)

    # --- TIM Store (separate connection to same database) ---
    try:
        tim_conn = duckdb.connect("data/apex_portfolio.duckdb")
        tim_store = TimStore(connection=tim_conn)
        tim_store.create_schema()
        tim_coordinator = DatabaseWriteCoordinator(tim_conn)
        tim_config = tim_store.load_tim_config()
        tim_bootstrap_enabled = tim_store.get_feature_flag("tim_bootstrap_enabled")
    except Exception:
        log.warning("TIM_SCHEMA_UNAVAILABLE — continuing without TIM", exc_info=True)
        tim_store = None
        tim_coordinator = None
        tim_config = None
        tim_bootstrap_enabled = False

    tim_mode = tim_config.tim_mode if tim_config is not None else TIMMode.OFF
    log.info("TIM_MODE_LOADED", tim_mode=tim_mode.value, bootstrap_enabled=tim_bootstrap_enabled)
    if tim_mode == TIMMode.OFF:
        log.info("TIM_MEMORY_DISABLED", tim_mode="OFF")
    if tim_bootstrap_enabled:
        log.info("TIM_BOOTSTRAP_ENABLED", enabled=True)

    # --- Position Mode Verification ---
    try:
        position_mode = await client.get_position_mode()
        log.info("Position mode verified", mode=position_mode)
        if position_mode != "ONE_WAY":
            log.warning("Hedge mode detected; execution assumes ONE_WAY (positionSide=BOTH)", mode=position_mode)
    except Exception as e:
        log.warning("Failed to verify position mode, assuming ONE_WAY", error=str(e))

    # --- Risk config (used by reconciliation and RiskManager init) ---
    risk_cfg = config.get("risk", {})
    exec_cfg = config.get("execution", {})

    # --- Startup Reconciliation (blocks scanner start) ---
    # Identity-preserving: persisted positions are matched by (symbol, side) and
    # overlaid with exchange-authoritative fields. No purge — exchange and local
    # are merged, not replaced.
    try:
        recon_result = await portfolio_mgr.reconcile(
            client,
            max_positions=risk_cfg.get("max_positions", 3),
            take_profit_pct=float(exec_cfg.get("take_profit_pct", 1.04)),
        )
        orphaned = recon_result.get("orphaned_closed", 0)
        adopted = recon_result.get("adopted", 0)
        mismatches = recon_result.get("qty_mismatches", 0)
        if any([orphaned, adopted, mismatches]):
            log.warning(
                "Reconciliation found discrepancies",
                orphaned_closed=orphaned,
                adopted=adopted,
                qty_mismatches=mismatches,
            )
        else:
            log.info("Reconciliation clean — no discrepancies found")
    except Exception as e:
        log.error("Reconciliation failed, continuing startup", error=str(e), traceback=traceback.format_exc())

    # --- TIM Startup Recovery (after reconciliation, before services) ---
    if tim_store is not None and tim_mode != TIMMode.OFF:
        from .tim.recovery import TimMemoryRecoveryService
        tim_recovery = TimMemoryRecoveryService(
            tim_store=tim_store,
            coordinator=tim_coordinator,
            tim_mode=tim_mode,
            bootstrap_enabled=tim_bootstrap_enabled,
        )
        open_positions = portfolio_mgr.get_open_positions()
        if open_positions:
            await tim_recovery.recover_open_positions(open_positions)

    # --- Feature flags ---
    shadow_cfg = config.get("shadow", {})
    shadow_enabled = shadow_cfg.get("enabled", True)
    mirror_enabled = shadow_cfg.get("mirror_enabled", True)
    calibration_enabled = config.get("calibration", {}).get("enabled", True)
    analytics_enabled = config.get("analytics", {}).get("enabled", True)

    risk_mgr = RiskManager(
        client=client,
        event_bus=event_bus,
        max_positions=risk_cfg.get("max_positions", 3),
        min_llm_confidence=risk_cfg.get("min_llm_confidence", 0.3),
        max_live_exposure_usdt=risk_cfg.get("max_live_exposure_usdt", 10000.0),
        take_profit_pct=float(exec_cfg.get("take_profit_pct", 1.04)),
    )

    llm_scheduler = LLMScheduler(
        registry=registry,
        model=model_id,
        audit_logger=portfolio_store,
        fallback_registry=fallback_registry,
        fallback_model=fallback_model,
    )
    context = MarketContextService()
    # --- Phase 4.6.1: Evidence Resolver (needs corpus + catalog) ---
    try:
        active_workspace = config_store.ensure_default_workspace()
    except duckdb.CatalogException:
        log.info("Recreating config store schema (missing table)")
        config_store._create_schema()
        active_workspace = config_store.ensure_default_workspace()

    # Ensure active profile has a linked workspace; switch to it
    active_profile = config_store.get_active_profile()
    if active_profile:
        profile_ws = config_store.ensure_profile_workspace(active_profile.profile_id, active_profile.name)
        if profile_ws.workspace_id != active_workspace.workspace_id:
            config_store.switch_workspace(profile_ws.workspace_id)
            active_workspace = profile_ws
            log.info("Switched to profile-linked workspace", workspace_id=active_workspace.workspace_id, profile_id=active_profile.profile_id, _force_log=True)

    learning_corpus = LearningCorpus(db_path=active_workspace.db_path)
    learning_corpus.create_schema()
    log.info(
        "LearningCorpus bound to workspace",
        workspace=active_workspace.name,
        db_path=active_workspace.db_path,
    )

    # Sync workspace trade_count from actual experience count
    try:
        config_store.sync_workspace_trade_count(active_workspace.workspace_id)
    except Exception as e:
        log.warning("Failed to sync workspace trade_count at startup", error=str(e))

    reasoning_coordinator = ReasoningCoordinator(
        llm_scheduler=llm_scheduler,
        event_bus=event_bus,
        corpus=learning_corpus,
    )

    # --- Phase 4.2 Execution Infrastructure ---
    live_executor = LiveExecutor(client, config)
    virtual_executor = VirtualExecutor(context, config.get("execution", {}))
    execution_svc = ExecutionService(
        live_executor=live_executor,
        virtual_executor=virtual_executor,
        market_context=context,
        portfolio_mgr=portfolio_mgr,
        event_bus=event_bus,
        config=config,
        mirror_enabled=mirror_enabled,
    )
    from .tim.bridge import TimMemoryBridge
    tim_bridge = TimMemoryBridge(
        tim_store=tim_store,
        coordinator=tim_coordinator,
        tim_mode=tim_mode,
    )
    feature_catalog = build_feature_catalog()
    retrieval_pipeline = RetrievalPipeline(
        corpus=learning_corpus,
        feature_catalog=feature_catalog,
    )
    intel_pipeline = ExperienceIntelligencePipeline()
    evidence_resolver = EvidenceResolver(
        retrieval_pipeline=retrieval_pipeline,
        intel_pipeline=intel_pipeline,
    )

    # --- Phase 4.2 Orchestration ---
    trade_coordinator = TradeCoordinator(
        risk_manager=risk_mgr,
        execution_svc=execution_svc,
        portfolio_mgr=portfolio_mgr,
        event_bus=event_bus,
        config=config,
        reasoning_coordinator=reasoning_coordinator,
        market_context_svc=context,
        evidence_resolver=evidence_resolver,
        config_store=config_store,
        session_id=active_session_id,
    )

    # --- Phase 4.2 Analytics ---
    if analytics_enabled:
        analytics_svc = AnalyticsService(event_bus, portfolio_store)
    else:
        analytics_svc = None

    # --- Phase 5.2: Metrics Pipeline ---
    metrics_svc = MetricsService(event_bus, portfolio_store)
    log.info("MetricsService initialized")

    # --- Phase 4.7: Decision Evaluation ---
    evaluation_corpus = EvaluationCorpus()
    decision_capture_store = DecisionCaptureStore(corpus=evaluation_corpus)
    event_bus.subscribe("CANDIDATE_EVALUATED", decision_capture_store._on_candidate_evaluated)
    evaluation_engine = DecisionEvaluationEngine()

    # --- Phase 4.8: Research & Recommendation Pipeline ---
    research_store = ResearchCorpusStore()
    research_pipeline = ResearchPipeline(evaluation_corpus, research_store)
    log.info("Research pipeline initialized")

    position_mgr = PositionManager(
        portfolio_mgr, execution_svc, llm_scheduler, event_bus, context,
        calibration_enabled=calibration_enabled,
        tim_bridge=tim_bridge,
    )
    scanner = MarketScanner(event_bus)

    # --- Phase 4.3: Learning Pipeline ---
    learning_extractor = ExperienceExtractor()
    learning_validator = ExperienceValidator()
    learning_normalizer = ExperienceNormalizer()
    config_catalog = build_config_catalog()
    rcfg = config.get("research", {})
    metric_config = {
        "min_metric_subgroup": rcfg.get("metric_min_subgroup", 3),
        "min_metric_losses": rcfg.get("metric_min_losses", 2),
        "min_improvement": rcfg.get("min_improvement", 0.01),
    }
    recommendation_engine = RecommendationEngine(catalog=config_catalog, store=config_store, metric_config=metric_config)
    log.info("Recommendation engine initialized")
    provenance_registry = build_provenance_registry()
    metadata_resolver = MetadataResolver(feature_catalog, config_catalog, provenance_registry)

    git_commit = os.environ.get("GIT_COMMIT", "")
    build_id = os.environ.get("BUILD_ID", "")
    application_version = "1.0.0"

    corpus_metadata = CorpusMetadata(
        pipeline_version="1.0",
        feature_catalog_hash=feature_catalog.catalog_hash,
        config_catalog_version=config_catalog.version,
        provenance_version=provenance_registry.version,
        application_version=application_version,
        git_commit=git_commit,
    )
    learning_corpus.save_corpus_metadata(corpus_metadata)

    learning_pipeline = LearningPipeline(
        extractor=learning_extractor,
        validator=learning_validator,
        normalizer=learning_normalizer,
        corpus=learning_corpus,
        metadata_resolver=metadata_resolver,
        git_commit=git_commit,
        build_id=build_id,
        application_version=application_version,
    )

    # ── Phase B: Observation → Timeline → Pattern → Hypothesis → Knowledge ──
    importance_scorer = ImportanceScorer(learning_corpus)
    timeline_manager = TimelineManager(learning_corpus)
    pattern_detector = PatternDetector(learning_corpus)
    hypothesis_extractor = HypothesisExtractor(learning_corpus)
    knowledge_promoter = KnowledgePromoter(learning_corpus)
    observation_compressor = ObservationCompressor(learning_corpus, window_minutes=15, batch_size=100)
    prediction_lifecycle = PredictionLifecycle(learning_corpus)
    observation_ingestor = ObservationIngestor(
        learning_corpus, importance_scorer, timeline_manager,
    )
    event_bus.subscribe("OBSERVATION_EMITTED", observation_ingestor.on_observation_emitted)
    log.info("Phase B observation pipeline initialized")

    # ── Phase 5C-5G: Self-Critique + Reflection + Belief + Profile + Memory ──
    reflection_engine = ReflectionEngine(learning_corpus, min_confidence_threshold=0.3)
    log.info("ReflectionEngine initialized")
    belief_evolution = BeliefEvolutionEngine(learning_corpus)
    log.info("BeliefEvolutionEngine initialized")
    profile_adapter = ProfileAdapter(learning_corpus, config_store)
    log.info("ProfileAdapter initialized")
    memory_tuner = AdaptiveMemoryTuner(retrieval_pipeline, learning_corpus)
    log.info("AdaptiveMemoryTuner initialized")

    feedback_mode = config.get("adaptive_feedback", {}).get("mode", "both")
    feedback_llm_interval = config.get("adaptive_feedback", {}).get("llm_interval_cycles", 20)
    feedback_engine = AdaptiveFeedbackEngine(
        config_store=config_store,
        llm_registry=registry if feedback_mode in ("llm", "both") else None,
        llm_model=model_id,
        evaluation_corpus=evaluation_corpus,
        research_store=research_store,
    )
    # Restore last-saved feedback state on startup
    saved_state = feedback_engine.load_state()
    if saved_state:
        log.info("ADAPTIVE_FEEDBACK_STATE_RESTORED", state=saved_state, _force_log=True)
        for key, value in saved_state.items():
            if key == "auto_merge":
                config.setdefault("adaptive", {})["auto_merge"] = value
            elif key == "show_confidence_in_prompt":
                config.setdefault("adaptive", {})["show_confidence_in_prompt"] = value
            elif key == "evidence_min_count":
                config.setdefault("learning", {})["evidence_min_count"] = int(value)
            elif key == "min_llm_confidence":
                config.setdefault("risk", {})["min_llm_confidence"] = float(value)
        # Propagate restored values into cached services
        if trade_coordinator is not None:
            trade_coordinator._config = config
        if execution_svc is not None:
            execution_svc._config = config
        if risk_mgr is not None:
            risk_mgr._min_llm_confidence = config.get("risk", {}).get("min_llm_confidence", 0.3)
    log.info("AdaptiveFeedbackEngine initialized", mode=feedback_mode, llm_interval=feedback_llm_interval)

    # ── Phase 5H: Full Feedback Loop Diagnostic ──
    loop_status = {
        "5B_decision_capture": "REASONING_EPISODE_CAPTURED/SAVED/RECORDED_PUBLISHED",
        "5C_self_critique": "SELF_CRITIQUE_STARTED/VERDICT/DECISION_FINALIZED",
        "5D_reflection": "REFLECTION_CYCLE_COMPLETE",
        "5E_belief_evolution": "BELIEF_EVOLUTION_CYCLE",
        "5F_profile_adaptation": "PROFILE_ADAPT_CYCLE",
        "5G_adaptive_memory": "MEMORY_TUNE_APPLIED",
        "exploration_protocol": "REMOVED — LLM uses natural confidence",
        "symbol_bias_handling": "FINDING created in ConfigurationStore",
    }
    log.info(
        "FEEDBACK_LOOP_READY",
        stages=loop_status,
        _force_log=True,
    )

    ws_task = None
    agg_task = None
    pipeline_task = None
    dispatcher_task = None
    llm_task = None
    monitor_task = None
    scan_task = None
    metrics_task = None
    analysis_task = None
    compressor_task = None
    reflection_task = None

    try:
        ws_task = asyncio.create_task(ws_ingestor.start_stream())
        agg_task = asyncio.create_task(_aggregation_catchup_loop(aggregator, log))
        pipeline_task = asyncio.create_task(
            _background_pipeline_worker(candle_queue, ingestor, aggregator, correlation_engine, matrix_store, log)
        )

        dispatcher_task = asyncio.create_task(event_bus.start_dispatcher())
        llm_task = asyncio.create_task(llm_scheduler.start())
        monitor_task = asyncio.create_task(position_mgr.monitor_positions())
        scan_task = asyncio.create_task(
            scanner_loop(scanner, alternates, context)
        )
        metrics_task = asyncio.create_task(
            _slow_metrics_loop(metrics_svc, log)
        )
        maintenance_task = asyncio.create_task(
            _memory_maintenance_loop(learning_corpus, config_store, log, knowledge_promoter)
        )
        analysis_task = asyncio.create_task(
            _timeline_analysis_loop(learning_corpus, pattern_detector, hypothesis_extractor, knowledge_promoter, log)
        )
        compressor_task = asyncio.create_task(
            _observation_compression_loop(observation_compressor, log)
        )
        reflection_interval = config.get("adaptive_feedback", {}).get("reflection_interval_seconds", 180)
        reflection_task = asyncio.create_task(
            _reflection_loop(reflection_engine, belief_evolution, profile_adapter, memory_tuner, feedback_engine, log, config, config_store, trade_coordinator, risk_mgr, execution_svc, llm_scheduler, evaluation_corpus=evaluation_corpus, research_pipeline=research_pipeline, recommendation_engine=recommendation_engine, research_store=research_store, interval=reflection_interval)
        )
        checkpoint_task = asyncio.create_task(
            _checkpoint_writer(config_store, interval=reflection_interval)
        )

        from src.services.metrics_service import compute_realized_pnl as _compute_trade_pnl

        async def _on_position_closed_learning(event):
            position_id = event.payload.get("position_id", "")
            new_state = event.payload.get("new_state", "")
            log.debug("Position event", position_id=position_id, new_state=new_state)

            # Phase 5.4: Track active versions at position OPEN
            if new_state == "OPEN":
                try:
                    active_versions = config_store.get_active_adaptive_versions(active_profile.profile_id)
                    if active_versions:
                        config_store.record_version_snapshot(
                            position_id, active_profile.profile_id, active_versions,
                        )
                except Exception as e:
                    log.warning("Version tracking failed at OPEN", position_id=position_id, error=str(e))
                return

            if new_state != "CLOSED":
                log.debug("Skipping non-closed state", new_state=new_state)
                return

            pos = portfolio_mgr.get_position_by_id(position_id)
            if pos is None:
                log.warning("Position not found for learning", position_id=position_id)
                return

            # Record version effectiveness on position close
            try:
                snapshot_data = config_store.get_version_snapshot(position_id)
                if snapshot_data:
                    for param_id, ver_id in snapshot_data.items():
                        version_obj = config_store.get_active_adaptive_versions(active_profile.profile_id).get(param_id)
                        conf = version_obj.effective_confidence if version_obj else 0.0
                        pnl = _compute_trade_pnl(pos)
                        if pnl is not None:
                            config_store.record_version_outcome(
                                ver_id, position_id, param_id, pnl, conf,
                            )
            except Exception as e:
                log.warning("Version effectiveness tracking failed", position_id=position_id, error=str(e))

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    log.info(
                        "Building snapshot",
                        position_id=pos.position_id,
                        symbol=pos.symbol,
                        _force_log=True,
                    )
                    snapshot = PositionSnapshot.from_position(pos)

                    exec_cfg = config.get("execution", {})
                    risk_cfg = config.get("risk", {})
                    runtime_config_values = {
                        "execution.leverage": exec_cfg.get("leverage", 5),
                        "execution.sizing_mode": exec_cfg.get("sizing_mode", "fixed_usdt"),
                        "execution.sizing_value": exec_cfg.get("sizing_value", 5.0),
                        "risk.max_concurrent_positions": risk_cfg.get("max_positions", 6),
                        "risk.max_live_exposure_usdt": risk_cfg.get("max_live_exposure_usdt", 25000),
                        "risk.min_llm_confidence": 0.3,
                        "execution.stop_loss_pct": 0.98,
                        "execution.take_profit_pct": 1.04,
                        "execution.trailing_stop_atr_mult": 2.0,
                        "execution.spread_bps": 2.0,
                        "execution.fee_bps": 4.0,
                        "execution.slippage_bps": 3.0,
                        "scanner.min_correlation": 0.6,
                        "scanner.max_p_value": 0.05,
                    }

                    # Phase 5 Completion: Use process_candidate instead of process
                    active_policy = config_store.get_active_learning_policy() or LearningPolicy(
                        name="Default", tier="balanced",
                    )

                    opp_id = ""
                    if hasattr(snapshot, "opportunity_id"):
                        opp_id = snapshot.opportunity_id
                    capture = decision_capture_store.get(opp_id) if opp_id else None

                    evidence_override = config.get("learning", {}).get("evidence_min_count")
                    result = await learning_pipeline.process_candidate(
                        snapshot=snapshot,
                        policy=active_policy,
                        runtime_config_values=runtime_config_values,
                        decision_capture=capture,
                        evaluation_engine=evaluation_engine,
                        evaluation_corpus=evaluation_corpus,
                        config_store=config_store,
                        evidence_min_count_override=evidence_override,
                    )

                    # If candidate was stored as pending (below threshold), save to corpus
                    if result.get("status") == "pending":
                        pending_manifest = result.get("manifest")
                        manifest_json = (
                            pending_manifest.model_dump(mode="json")
                            if pending_manifest is not None
                            else {}
                        )
                        candidate_data = {
                            "candidate_id": result.get("position_id", position_id),
                            "position_id": position_id,
                            "manifest_json": manifest_json,
                            "status": "pending",
                            "evidence_count": result.get("evidence_count", 1),
                            "validation_report": result.get("validation"),
                            "noise_assessment": {"noise_score": result.get("noise_score")},
                            "confidence_score": {"score": result.get("confidence")},
                            "policy_id": active_policy.policy_id,
                        }
                        learning_corpus.save_candidate(candidate_data)
                        log.info(
                            "Candidate saved as pending",
                            position_id=position_id,
                            evidence_count=result.get("evidence_count"),
                            threshold=active_policy.evidence_min_count,
                        )

                    # If rejected as noise/validation, store as rejected
                    if result.get("status") in ("rejected", "noise_rejected", "low_confidence"):
                        learning_corpus.record_rejection(
                            candidate={"candidate_id": position_id, "position_id": position_id, "manifest_json": {}},
                            reason=result.get("status", "rejected"),
                            stage="pipeline",
                            details={"validation": result.get("validation"), "noise_score": result.get("noise_score")},
                        )

                    if result.get("stored"):
                        try:
                            active_ws = config_store.get_active_workspace()
                            if active_ws:
                                config_store.increment_workspace_trade_count(active_ws.workspace_id)
                        except Exception as e:
                            log.warning("Failed to increment workspace trade_count", error=str(e))

                    eval_data = result.get("evaluation")
                    if eval_data:
                        log.info(
                            "LEARNING_FEEDBACK",
                            phase="close",
                            position_id=pos.position_id,
                            symbol=pos.symbol,
                            status=result.get("status"),
                            evidence_count=result.get("evidence_count", 0),
                            was_profitable=eval_data.get("was_profitable"),
                            pnl=eval_data.get("actual_pnl"),
                            confidence_vs_outcome=eval_data.get("confidence_vs_outcome"),
                            exit_reason=eval_data.get("exit_reason"),
                            _force_log=True,
                        )
                    else:
                        log.info(
                            "Learning candidate processed",
                            position_id=pos.position_id,
                            status=result.get("status"),
                            _force_log=True,
                        )
                    return
                except CandidateRejectionError as e:
                    learning_corpus.record_rejection(
                        candidate={"candidate_id": position_id, "position_id": position_id, "manifest_json": {}},
                        reason=str(e),
                        stage="pipeline",
                        details={"status": "lifecycle_rejected"},
                    )
                    log.info("Candidate rejected by lifecycle", position_id=position_id, reason=str(e))
                    return
                except Exception as e:
                    log.warning(
                        "Learning pipeline attempt failed",
                        position_id=position_id, attempt=attempt + 1, error=str(e),
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        log.error(
                            "Learning pipeline failed after all retries",
                            position_id=position_id, error=str(e),
                        )

        event_bus.subscribe("POSITION_UPDATED", _on_position_closed_learning)

        async def _on_interim_learning(event):
            position_id = event.payload.get("position_id", "")
            trigger = event.payload.get("trigger", "unknown")
            log.info(
                "Interim learning triggered",
                position_id=position_id, trigger=trigger,
            )

            pos = portfolio_mgr.get_position_by_id(position_id)
            if pos is None:
                log.debug("Position not found for interim learning", position_id=position_id)
                return

            if pos.lifecycle_state.value in ("CLOSED", "ARCHIVED"):
                log.debug("Skipping interim learning — position already closed", position_id=position_id)
                return

            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    snapshot = PositionSnapshot.from_position_interim(pos)

                    active_policy = config_store.get_active_learning_policy() or LearningPolicy(
                        name="Default", tier="balanced",
                    )

                    result = await learning_pipeline.process_candidate(
                        snapshot=snapshot,
                        policy=active_policy,
                        runtime_config_values=runtime_config_values,
                        decision_capture=None,
                    )

                    status = result.get("status")
                    evidence_count = result.get("evidence_count", 0)

                    # Save pending interim candidates so evidence accumulates for future matches
                    if status == "pending":
                        pending_manifest = result.get("manifest")
                        manifest_json = (
                            pending_manifest.model_dump(mode="json")
                            if pending_manifest is not None
                            else {}
                        )
                        candidate_data = {
                            "candidate_id": result.get("position_id", position_id),
                            "position_id": position_id,
                            "manifest_json": manifest_json,
                            "status": "pending",
                            "evidence_count": evidence_count,
                            "validation_report": result.get("validation"),
                            "noise_assessment": {"noise_score": result.get("noise_score")},
                            "confidence_score": {"score": result.get("confidence")},
                            "policy_id": active_policy.policy_id,
                        }
                        learning_corpus.save_candidate(candidate_data)

                    if result.get("stored"):
                        try:
                            active_ws = config_store.get_active_workspace()
                            if active_ws:
                                config_store.increment_workspace_trade_count(active_ws.workspace_id)
                        except Exception as e:
                            log.warning("Failed to increment workspace trade_count during interim learning", error=str(e))

                    log.info(
                        "Interim learning result",
                        position_id=position_id,
                        trigger=trigger,
                        status=status,
                        evidence_count=evidence_count,
                        threshold=active_policy.evidence_min_count,
                        _force_log=True,
                    )
                    return
                except CandidateRejectionError as e:
                    log.info("Interim candidate rejected", position_id=position_id, reason=str(e))
                    return
                except Exception as e:
                    log.warning(
                        "Interim learning attempt failed",
                        position_id=position_id, attempt=attempt + 1, error=str(e),
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(3)

        event_bus.subscribe("LEARNING_INTERIM", _on_interim_learning)

        # ── Phase 6: Startup Learning Recovery ──
        try:
            recovery_svc = SystemRecoveryService(
                portfolio_mgr=portfolio_mgr,
                learning_pipeline=learning_pipeline,
                learning_corpus=learning_corpus,
                evaluation_corpus=evaluation_corpus,
                decision_capture_store=decision_capture_store,
                evaluation_engine=evaluation_engine,
            )
            exec_cfg = config.get("execution", {})
            risk_cfg = config.get("risk", {})
            runtime_config_values = {
                "execution.leverage": exec_cfg.get("leverage", 5),
                "execution.sizing_mode": exec_cfg.get("sizing_mode", "fixed_usdt"),
                "execution.sizing_value": exec_cfg.get("sizing_value", 5.0),
                "risk.max_concurrent_positions": risk_cfg.get("max_positions", 6),
                "risk.max_live_exposure_usdt": risk_cfg.get("max_live_exposure_usdt", 25000),
                "risk.min_llm_confidence": 0.3,
                "execution.stop_loss_pct": 0.98,
                "execution.take_profit_pct": 1.04,
                "execution.trailing_stop_atr_mult": 2.0,
                "execution.spread_bps": 2.0,
                "execution.fee_bps": 4.0,
                "execution.slippage_bps": 3.0,
                "scanner.min_correlation": 0.6,
                "scanner.max_p_value": 0.05,
            }
            recovered_orphaned = await recovery_svc.recover_orphaned_positions(runtime_config_values, config_store=config_store)
            if recovered_orphaned > 0:
                log.info("Startup learning recovery complete", recovered=recovered_orphaned)
        except Exception as e:
            log.warning("Startup learning recovery error", error=str(e))

        # Phase 5 Extension: Full system recovery
        active_policy = config_store.get_active_learning_policy() or LearningPolicy(
            name="Default", tier="balanced",
        )
        try:
            recovery_result = await recovery_svc.full_system_recovery(
                corpus=learning_corpus,
                config_store=config_store,
                policy=active_policy,
                runtime_config_values=runtime_config_values,
            )
        except Exception as e:
            log.warning("Full system recovery error", error=str(e))
            recovery_result = {"remaining_issues": [str(e)]}

        # Phase 5 Extension: Log MEMORY_STARTUP
        try:
            health = learning_corpus.get_memory_health()
            log.info(
                "MEMORY_STARTUP",
                workspace=config_store.get_active_workspace().name if config_store.get_active_workspace() else "default",
                learning_policy=active_policy.tier,
                experience_count=health.experience_count,
                pending_candidates=health.pending_candidates,
                rejected_candidates=health.rejected_count,
                duplicate_count=health.duplicate_count,
                last_maintenance=health.last_maintenance,
                integrity_state=health.integrity_state,
                result="PASS" if not recovery_result.get("remaining_issues") else "PARTIAL",
            )
        except Exception as e:
            log.warning("MEMORY_STARTUP logging failed", error=str(e))

        # Initial metrics baseline
        try:
            metrics_svc.record_slow_metrics()
        except Exception as e:
            log.warning("Initial metrics snapshot failed", error=str(e))

        feature_summary = ", ".join(
            f"{k}={v}" for k, v in [
                ("shadow", shadow_enabled),
                ("mirror", mirror_enabled),
                ("calibration", calibration_enabled),
                ("analytics", analytics_enabled),
            ]
        )
        rich.print(
            Panel.fit(
                Text.assemble(
                    ("Ingestion pipeline active", "bold green"),
                    "\n",
                    (f"Tracking {len(all_symbols)} symbols via WebSocket", "dim"),
                    "\n",
                    ("Event-driven scanner active", "bold cyan"),
                    "\n",
                    (f"Phase 4.2: {feature_summary}", "yellow"),
                    "\n",
                    (f"Output mode: {get_verbose_mode().upper()}", "cyan"),
                    "\n",
                    (f"Log focus: {', '.join(config.get('logging', {}).get('focus', ['all']))}", "cyan"),
                    "\n",
                    ("Press Ctrl+C to stop", "dim"),
                ),
                border_style="green",
            )
        )

        tasks_to_gather = [
            ws_task, agg_task, pipeline_task,
            dispatcher_task, llm_task, monitor_task, scan_task, metrics_task,
            maintenance_task, analysis_task, compressor_task, reflection_task,
        ]
        print(">>> GATHER: starting", flush=True)
        log.info("ALL_TASKS_READY", task_count=len(tasks_to_gather), _force_log=True)
        await asyncio.gather(*tasks_to_gather)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    except Exception:
        log.error("Fatal error in main loop", exc_info=True)
        import sys as _sys
        _sys.stderr.flush()
        graceful = True
        return
    finally:
        graceful = False
        try:
            rich.print("\n[yellow]Shutting down gracefully...[/]")
            ws_ingestor.stop()
            loop_tasks = [ws_task, agg_task, pipeline_task, dispatcher_task, llm_task, monitor_task, scan_task, metrics_task, maintenance_task, analysis_task, compressor_task, reflection_task]

            active = [t for t in loop_tasks if t and not t.done()]
            for t in active:
                t.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)

            await event_bus.stop()
            await llm_scheduler.stop()
            await asyncio.sleep(0.1)

            if client:
                try:
                    await client.close()
                except Exception as e:
                    log.error("Failed to close Binance client session", error=str(e))
            ingestor.close()
            aggregator.close()
            matrix_store.close()
            correlation_engine.close()
            portfolio_store.close()
            context.close()
            learning_normalizer.close()
            learning_corpus.close()
            session_manager.end_session(active_session_id)
            config_store.close()
            session_manager.close()
            from src.utils.log_categories import stop_category_logging
            stop_category_logging()
            log.info("APEX shutdown complete.")
            rich.print("[green]Goodbye.[/]")
            graceful = True
        except (asyncio.CancelledError, KeyboardInterrupt):
            rich.print("\n[yellow]Shutdown interrupted by signal. Forcing exit.[/]")
        except Exception:
            log.error("Unexpected error during shutdown", exc_info=True)
        finally:
            if not graceful:
                import sys as _sys
                _sys.stderr.flush()
                os._exit(0)


async def _slow_metrics_loop(metrics_svc: MetricsService, log, interval: int = 3600):
    while True:
        try:
            await asyncio.sleep(interval)
            log.info("Running slow metrics computation")
            snapshot_id = metrics_svc.record_slow_metrics()
            log.info("Slow metrics snapshot recorded", snapshot_id=snapshot_id)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("Slow metrics loop failed", error=str(e), traceback=traceback.format_exc())


def _create_profiles_from_recommendations(config_store, config, log, research_report=None) -> list[str]:
    """Create ConfigurationProfiles from HIGH-confidence recommendations or COMPLETE research reports.
    Returns list of new profile_ids created."""
    from src.recommendations.models import ConfigurationProfile as _ConfigProfile
    from copy import deepcopy
    from datetime import datetime

    consumed_rec_ids: set[str] = set()
    for p in config_store.list_profiles(limit=100):
        consumed_rec_ids.update(p.derived_from_recommendations)

    recs = config_store.list_recommendations(status_filter="SIMULATED", limit=100)
    high_recs = [r for r in recs if r.confidence_tier == "HIGH" and r.recommendation_id not in consumed_rec_ids]

    base_config = deepcopy(config)
    param_defs = config_store.get_all_adaptive_parameters()

    if high_recs:
        profile_rec_ids: list[str] = []
        description_parts: list[str] = []
        for rec in high_recs:
            intervention = config_store.get_intervention(rec.intervention_id)
            if intervention is None:
                continue
            param_def = param_defs.get(intervention.parameter_id)
            if param_def is None:
                continue
            keys = param_def.config_path.split(".")
            target = base_config
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = intervention.recommended_value
            profile_rec_ids.append(rec.recommendation_id)
            description_parts.append(f"{intervention.parameter_id}={intervention.recommended_value}")

        if profile_rec_ids:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            param_summary = ", ".join(description_parts[:3])
            if len(description_parts) > 3:
                param_summary += f" +{len(description_parts) - 3} more"
            profile_name = f"Research: {param_summary}"
            profile = _ConfigProfile(
                name=profile_name,
                description=f"Auto-generated from {len(profile_rec_ids)} HIGH recommendations on {timestamp}",
                system_generated=True,
                resolved_configuration=base_config,
                derived_from_recommendations=profile_rec_ids,
            )
            config_store.save_profile(profile)
            ws = config_store.ensure_profile_workspace(profile.profile_id, profile_name)
            config_store.set_workspace_trade_count(ws.workspace_id, len(profile_rec_ids))
            log.info("[PROFILE] Created from recommendations", profile_id=profile.profile_id, name=profile_name, rec_count=len(profile_rec_ids), _force_log=True)
            return [profile.profile_id]

    # Fallback: create a profile from a COMPLETE research report even without HIGH recommendations
    if research_report and research_report.status == "COMPLETE":
        existing_names = {p.name for p in config_store.list_profiles(limit=100)}
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        profile_name = f"Research: {research_report.sample_size} evaluations"
        if profile_name in existing_names:
            profile_name = f"Research: {research_report.sample_size} evals ({timestamp})"
        bias_types = [b.bias_type for b in research_report.bias_findings]
        description = (
            f"Auto-generated from COMPLETE research report on {timestamp}. "
            f"Sample: {research_report.sample_size} evaluations. "
            f"Findings: {len(research_report.bias_findings)} bias, {len(research_report.observations)} observations."
        )
        profile = _ConfigProfile(
            name=profile_name,
            description=description,
            system_generated=True,
            resolved_configuration=base_config,
            derived_from_recommendations=[],
        )
        config_store.save_profile(profile)
        ws = config_store.ensure_profile_workspace(profile.profile_id, profile_name)
        config_store.set_workspace_trade_count(ws.workspace_id, research_report.sample_size)
        log.info("[PROFILE] Created from research report", profile_id=profile.profile_id, name=profile_name, _force_log=True)
        return [profile.profile_id]

    return []


async def _reflection_loop(reflection_engine, belief_evolution, profile_adapter, memory_tuner, feedback_engine, log, config: dict, config_store, trade_coordinator=None, risk_mgr=None, execution_svc=None, llm_scheduler=None, evaluation_corpus=None, research_pipeline=None, recommendation_engine=None, research_store=None, interval: int = 180):
    """Periodically reflect, evolve beliefs, adapt profile, tune memory, adjust thresholds."""
    research_cycle = 0
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            log.info("[REFLECTION] Shutting down reflection loop")
            break
        try:
            observations = reflection_engine.reflect(lookback_minutes=60)
            beliefs_generated = 0
            adaptations_count = 0
            memory_tuned = False
            beliefs = []
            if observations:
                log.info("[REFLECTION] Generated observations", count=len(observations))
                beliefs = belief_evolution.evolve(observations)
                if beliefs:
                    beliefs_generated = len(beliefs)
                    log.info("[BELIEF] Evolved beliefs", count=beliefs_generated)
            adaptations = profile_adapter.adapt()
            if adaptations:
                adaptations_count = len(adaptations)
                log.info("[PROFILE] Profile adapted", actions=adaptations_count)

            # Emit additional observations for feedback engine
            if adaptations_count > 0:
                observations.append({
                    "category": "reflection_profile_adaptations",
                    "importance": 0.5,
                    "data": {"count": adaptations_count},
                })
            try:
                from src.storage.learning.learning_corpus import LearningCorpus
                pending = learning_corpus.count_pending_candidates()
                if pending > 0:
                    observations.append({
                        "category": "reflection_evidence_pending",
                        "importance": 0.4,
                        "data": {"count": pending},
                    })
            except Exception:
                pass

            # Re-merge adaptive parameters into runtime config
            if config.get("adaptive", {}).get("auto_merge", True):
                active_profile_ = config_store.get_active_profile()
                if active_profile_:
                    try:
                        active_adaptations = config_store.get_active_adaptive_versions(active_profile_.profile_id)
                        param_defs_ = config_store.get_all_adaptive_parameters()
                        if active_adaptations:
                            from src.recommendations.lifecycle import merge_adaptive_config
                            merged = merge_adaptive_config(config, active_adaptations, param_defs_)
                            config.clear()
                            config.update(merged)
                            # Propagate merged values into services that cache them
                            if trade_coordinator is not None:
                                trade_coordinator._config = config
                            if execution_svc is not None:
                                execution_svc._config = config
                            if risk_mgr is not None:
                                risk_mgr._min_llm_confidence = config.get("risk", {}).get("min_llm_confidence", 0.3)
                            log.info("[AUTO_MERGE] Adaptive params re-merged", count=len(active_adaptations))
                    except Exception as e:
                        log.warning("Auto-merge failed", error=str(e))

            # Adaptive feedback — adjust thresholds based on reflection metrics
            feedback_mode = config.get("adaptive_feedback", {}).get("mode", "both")
            try:
                adjustments = await feedback_engine.run(
                    mode=feedback_mode,
                    config=config,
                    observations=observations,
                    beliefs=beliefs,
                )
                if adjustments:
                    log.info("ADAPTIVE_FEEDBACK_ADJUSTMENTS", adjustments=adjustments, _force_log=True)
                    for key, value in adjustments.items():
                        if key == "auto_merge":
                            config.setdefault("adaptive", {})["auto_merge"] = value
                        elif key == "show_confidence_in_prompt":
                            config.setdefault("adaptive", {})["show_confidence_in_prompt"] = value
                        elif key == "evidence_min_count":
                            config.setdefault("learning", {})["evidence_min_count"] = int(value)
                        elif key == "min_llm_confidence":
                            config.setdefault("risk", {})["min_llm_confidence"] = float(value)
                    # Propagate to cached services
                    if trade_coordinator is not None:
                        trade_coordinator._config = config
                    if execution_svc is not None:
                        execution_svc._config = config
                    if risk_mgr is not None:
                        risk_mgr._min_llm_confidence = config.get("risk", {}).get("min_llm_confidence", 0.3)
            except Exception as e:
                log.warning("Adaptive feedback cycle failed", error=str(e))

            tuned = memory_tuner.tune()
            if tuned:
                memory_tuned = True
                log.info("[MEMORY] Retrieval weights tuned")
            log.info(
                "FEEDBACK_LOOP_CYCLE",
                observations=len(observations),
                beliefs_generated=beliefs_generated,
                adaptations=adaptations_count,
                memory_tuned=memory_tuned,
                _force_log=True,
            )

            # ── Research & Recommendation Stage (every 10 cycles) ──
            research_cycle += 1
            research_interval = config.get("research", {}).get("cycle_interval", 10)
            if research_pipeline and evaluation_corpus and recommendation_engine and research_cycle % research_interval == 0:
                try:
                    log.info("[RESEARCH] Starting research cycle", cycle=research_cycle, _force_log=True)
                    rcfg = config.get("research", {})
                    pattern_config = {k.removeprefix("pattern_"): v for k, v in rcfg.items() if k.startswith("pattern_")}
                    observation_config = {k.removeprefix("observation_"): v for k, v in rcfg.items() if k.startswith("observation_")}
                    report_id = research_pipeline.generate_research_report(
                        min_sample_size=rcfg.get("min_sample_size", 30),
                        pattern_config=pattern_config,
                        observation_config=observation_config,
                    )
                    reports = research_store.list(limit=1)
                    if reports:
                        report = reports[0]
                        log.info(
                            "[RESEARCH] Report generated",
                            report_id=report.report_id,
                            status=report.status,
                            sample_size=report.sample_size,
                            _force_log=True,
                        )
                        research_min_sample = config.get("research", {}).get("min_sample_size", 30)
                        if report.status == "COMPLETE" and report.sample_size >= research_min_sample:
                            evaluations = evaluation_corpus.list(limit=0)
                            log.info(
                                "[RESEARCH] Loaded evaluations for recommendation",
                                count=len(evaluations),
                                _force_log=True,
                            )
                            findings, interventions, recommendations = recommendation_engine.generate(
                                report=report,
                                evaluations=evaluations,
                                min_sample_size=config.get("research", {}).get("min_sample_size", 30),
                                min_intervention_evals=config.get("research", {}).get("min_intervention_evals", 5),
                                min_simulation_evals=config.get("research", {}).get("min_simulation_evals", 3),
                                min_effect_size=config.get("research", {}).get("min_effect_size", 0.2),
                                pattern_config=pattern_config,
                                observation_config=observation_config,
                            )
                            high_count = sum(1 for r in recommendations if r.confidence_tier == "HIGH")
                            log.info(
                                "[RECOMMEND] Pipeline complete",
                                findings=len(findings),
                                interventions=len(interventions),
                                recommendations=len(recommendations),
                                high_confidence=high_count,
                                _force_log=True,
                            )
                            new_profile_ids = _create_profiles_from_recommendations(config_store, config, log, research_report=report)
                            if new_profile_ids:
                                log.info(
                                    "[PROFILE] Research profiles created",
                                    count=len(new_profile_ids),
                                    ids=new_profile_ids,
                                    _force_log=True,
                                )
                    else:
                        log.info("[RESEARCH] No reports available yet (insufficient data)")
                except Exception as e:
                    log.warning("[RESEARCH] Research/recommendation cycle failed", error=str(e))
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("Reflection cycle failed", error=str(e))
            await asyncio.sleep(interval)


async def _memory_maintenance_loop(corpus, config_store, log, knowledge_promoter=None):
    while True:
        try:
            policy = config_store.get_active_learning_policy()
            if policy is None:
                from src.recommendations.models import LearningPolicy as _LP
                policy = _LP(name="Default", tier="balanced")
            await asyncio.sleep(policy.maintenance_interval_hours * 3600)
            await corpus.run_maintenance(policy)
            if knowledge_promoter is not None:
                promoted = knowledge_promoter.promote_all()
                if promoted:
                    log.info("[KNOWLEDGE] Promoted during maintenance", count=len(promoted))
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("Memory maintenance cycle failed", error=str(e))
            await asyncio.sleep(3600)


async def _timeline_analysis_loop(corpus, pattern_detector, hypothesis_extractor, knowledge_promoter, log, interval: int = 120):
    """Periodically process closed timelines: detect patterns, extract hypotheses, promote knowledge."""
    while True:
        try:
            await asyncio.sleep(interval)
            from datetime import datetime, timedelta
            since = datetime.utcnow() - timedelta(hours=24)
            closed = corpus.get_closed_timelines_since(since)
            ready = [t for t in closed if t.status.value == "ready_for_analysis"]
            if not ready:
                continue
            log.info("[MEMORY] Timeline analysis cycle", ready_count=len(ready))
            for tl in ready:
                try:
                    patterns = pattern_detector.detect_all(tl.timeline_id)
                    if patterns:
                        hypotheses = hypothesis_extractor.extract_all(tl.timeline_id)
                        if hypotheses:
                            for hyp in hypotheses:
                                corpus.update_hypothesis_status(hyp.hypothesis_id, HypothesisStatus.MATURE)
                            log.info("[MEMORY] Timeline analyzed",
                                      timeline_id=tl.timeline_id,
                                      patterns=len(patterns),
                                      hypotheses=len(hypotheses))
                    corpus.update_timeline_status(tl.timeline_id, type(tl.status)("analyzed"))
                except Exception as e:
                    log.warning("[MEMORY] Timeline analysis failed",
                                 timeline_id=tl.timeline_id, error=str(e))
            promoted = knowledge_promoter.promote_all()
            if promoted:
                log.info("[MEMORY] Knowledge promoted", count=len(promoted))
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("Timeline analysis cycle failed", error=str(e))
            await asyncio.sleep(interval)


async def _observation_compression_loop(compressor, log, interval: int = 900):
    """Periodically compress eligible observations into aggregates."""
    while True:
        try:
            await asyncio.sleep(interval)
            count = compressor.compress_recent_all()
            if count:
                log.info("[MEMORY] Observations compressed", aggregates_created=count)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("Observation compression cycle failed", error=str(e))


async def _checkpoint_writer(config_store, interval: int = 30):
    """Periodically persist active profile/workspace to a crash-safe checkpoint file."""
    while True:
        try:
            await asyncio.sleep(interval)
            ap = config_store.get_active_profile()
            aw = config_store.get_active_workspace()
            if ap:
                write_checkpoint(ap.profile_id, aw.workspace_id if aw else None)
        except asyncio.CancelledError:
            break
        except Exception:
            pass


async def _aggregation_catchup_loop(aggregator: Aggregator, log, interval: int = 60):
    while True:
        try:
            await asyncio.sleep(interval)
            log.info("Running aggregation catch-up cycle")
            await asyncio.to_thread(aggregator.aggregate_timeframes)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("Aggregation catch-up failed", error=str(e), traceback=traceback.format_exc())


async def _background_pipeline_worker(
    queue: asyncio.Queue,
    ingestor: Ingestor,
    aggregator: Aggregator,
    engine: CorrelationEngine,
    matrix_store: CorrelationMatrixStore,
    log,
):
    while True:
        payload = await queue.get()
        try:
            candle_dict = {
                "symbol": payload.symbol,
                "open_time": payload.open_time,
                "close_time": payload.close_time,
                "open": payload.open,
                "high": payload.high,
                "low": payload.low,
                "close": payload.close,
                "volume": payload.volume,
            }

            await ingestor.append_candle(payload.symbol, candle_dict)

            minute_idx = payload.open_time // 60_000
            for tf in range(2, aggregator.max_timeframe_m + 1):
                if (minute_idx + 1) % tf == 0:
                    await asyncio.to_thread(aggregator.update_timeframe, candle_dict, tf)

            lr = engine.update_price(payload.symbol, payload.close)
            if lr is not None:
                engine.append_log_return(payload.symbol, lr)
                if engine.ready_to_compute():
                    results = await asyncio.to_thread(engine.compute_all_pairs)
                    await asyncio.to_thread(matrix_store.insert_snapshot, results)
                    try:
                        render_live_matrix(matrix_store)
                    except Exception:
                        log.warning("Failed to render live correlation matrix", exc_info=True)
        except Exception as e:
            log.error(
                "Pipeline worker error",
                symbol=payload.symbol,
                error=str(e),
                traceback=traceback.format_exc(),
            )


async def scanner_loop(
    scanner: MarketScanner,
    alternates: list[str],
    context: MarketContextService,
    interval: int = 30,
) -> None:
    log = get_logger("scanner_loop")
    log.info("Scanner loop started", interval_seconds=interval, alternates=alternates)
    while True:
        try:
            await asyncio.sleep(interval)
            await scanner.run_scan_cycle(alternates, context)
        except asyncio.CancelledError:
            log.info("Scanner loop cancelled")
            break
        except Exception as e:
            log.error("Scanner loop error", error=str(e), traceback=traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX Trading Engine")
    parser.add_argument(
        "--config-set",
        nargs=2,
        action="append",
        metavar=("KEY", "VALUE"),
        default=[],
        help="Update a config value (dot-notation key) and exit. Can be repeated.",
    )
    args, _ = parser.parse_known_args()

    if args.config_set:
        config = load_config(CONFIG_PATH)
        for key_path, raw_value in args.config_set:
            value = _parse_value(raw_value)
            parts = key_path.split(".")
            target = config
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        save_config(CONFIG_PATH, config)
        rich.print(f"[green]Updated {len(args.config_set)} config value(s) in {CONFIG_PATH}[/]")
        sys.exit(0)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rich.print("\n[yellow]Interrupted.[/]")
    except SystemExit as _se:
        import sys as _sys2
        print(f"[SYSTEM_EXIT] code={_se.code}", file=_sys2.stderr, flush=True)
    os._exit(0)
