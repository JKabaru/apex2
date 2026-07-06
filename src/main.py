import argparse
import asyncio
import json
import os
import sys
import traceback

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
from .services.execution import ExecutionService, LiveExecutor, VirtualExecutor
from .services.position_manager import PositionManager
from .services.scanner import MarketScanner
from .services.trade_coordinator import TradeCoordinator
from .services.reasoning_coordinator import ReasoningCoordinator
from .services.evidence_resolver import EvidenceResolver
from .retrieval.pipeline import RetrievalPipeline
from .retrieval.weights import SimilarityWeights
from .intelligence.pipeline import ExperienceIntelligencePipeline
from .engine.output_mode import set_mode as set_verbose_mode, get_mode as get_verbose_mode
from .services.analytics_service import AnalyticsService
from .models.learning.trade_experience import PositionSnapshot
from .models.learning.corpus_metadata import CorpusMetadata
from .learning.extractor import ExperienceExtractor
from .learning.validator import ExperienceValidator
from .learning.normalizer import ExperienceNormalizer
from .learning.pipeline import LearningPipeline, MetadataResolver
from .learning.feature_catalog import _build_default_catalog as build_feature_catalog
from .learning.config_catalog import _build_default_catalog as build_config_catalog
from .learning.provenance import _build_default_registry as build_provenance_registry
from .storage.learning.learning_corpus import LearningCorpus
from .evaluation.store import DecisionCaptureStore
from .evaluation.engine import DecisionEvaluationEngine
from .evaluation.storage import EvaluationCorpus
from .recommendations.store import ConfigurationStore
from .models.session import TradingSession
from .services.session_manager import SessionManager
from .operator.cli import StartupCLI

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

    init_logging()
    log = get_logger("main")
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

    # --- Phase 5.0: Profile Review (blocking, no network) ---
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

    recommendations_for_review = config_store.list_recommendations(status_filter="SIMULATED")

    chosen_profile_id = await StartupCLI.run_startup_review(
        profiles=profiles,
        recommendations=recommendations_for_review,
        current_active=current_active,
    )

    if chosen_profile_id is None:
        chosen_profile_id = current_active.profile_id if current_active else profiles[0].profile_id

    active_profile = config_store.get_profile(chosen_profile_id)
    if active_profile and (not current_active or chosen_profile_id != current_active.profile_id):
        config_store.activate_profile(chosen_profile_id, activated_by="operator")
        current_active = active_profile

    active_profile = current_active or profiles[0]
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

    # --- Emergency State Reset ---
    if action == "reset":
        from .services.reset_service import EmergencyResetService
        rich.print("[yellow]Starting Emergency State Reset...[/]")
        await EmergencyResetService.execute_hard_reset(client, portfolio_store, portfolio_mgr, log)
        rich.print("[bold green]Emergency Reset Complete. Local state cleared. Exchange account flat.[/]")
        log.info("Emergency reset completed. Returning to main menu.")
        portfolio_store.close()
        await client.close()
        rich.print("[green]You can now start the engine with a clean state.[/]")
        return await main()

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

    # --- Startup Reconciliation (blocks scanner start) ---
    try:
        recon_result = await portfolio_mgr.reconcile(
            client,
            max_positions=risk_cfg.get("max_positions", 3),
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
    )

    llm_scheduler = LLMScheduler(
        registry=registry,
        model=model_id,
        audit_logger=portfolio_store,
        fallback_registry=fallback_registry,
        fallback_model=fallback_model,
    )
    context = MarketContextService()
    reasoning_coordinator = ReasoningCoordinator(
        llm_scheduler=llm_scheduler,
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

    # --- Phase 4.6.1: Evidence Resolver (needs corpus + catalog) ---
    learning_corpus = LearningCorpus()
    learning_corpus.create_schema()
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

    # --- Phase 4.7: Decision Evaluation ---
    decision_capture_store = DecisionCaptureStore()
    event_bus.subscribe("CANDIDATE_EVALUATED", decision_capture_store._on_candidate_evaluated)
    evaluation_engine = DecisionEvaluationEngine()
    evaluation_corpus = EvaluationCorpus()

    position_mgr = PositionManager(
        portfolio_mgr, execution_svc, llm_scheduler, event_bus, context,
        calibration_enabled=calibration_enabled,
    )
    scanner = MarketScanner(event_bus)

    # --- Phase 4.3: Learning Pipeline ---
    learning_extractor = ExperienceExtractor()
    learning_validator = ExperienceValidator()
    learning_normalizer = ExperienceNormalizer()
    config_catalog = build_config_catalog()
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

    ws_task = None
    agg_task = None
    pipeline_task = None
    dispatcher_task = None
    llm_task = None
    monitor_task = None
    scan_task = None

    try:
        ws_task = asyncio.create_task(ws_ingestor.start_stream())
        agg_task = asyncio.create_task(_aggregation_catchup_loop(aggregator, log))
        pipeline_task = asyncio.create_task(
            _background_pipeline_worker(candle_queue, ingestor, aggregator, correlation_engine, matrix_store, log)
        )

        dispatcher_task = asyncio.create_task(event_bus.start_dispatcher())
        llm_task = asyncio.create_task(llm_scheduler.process_queue())
        monitor_task = asyncio.create_task(position_mgr.monitor_positions())
        scan_task = asyncio.create_task(
            scanner_loop(scanner, alternates, context)
        )

        async def _on_position_closed_learning(event):
            log.info("Learning adapter fired", event_type=event.event_type, payload_keys=list(event.payload.keys()))
            if event.payload.get("new_state") != "CLOSED":
                log.debug("Skipping non-closed state", new_state=event.payload.get("new_state"))
                return

            position_id = event.payload["position_id"]
            pos = portfolio_mgr.get_position_by_id(position_id)
            if pos is None:
                log.warning("Position not found for learning", position_id=position_id)
                return

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    log.info("Building snapshot", position_id=pos.position_id, symbol=pos.symbol)
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
                    manifest = await learning_pipeline.process(snapshot, runtime_config_values=runtime_config_values)
                    log.info("Learning pipeline completed", position_id=pos.position_id, symbol=pos.symbol)

                    opp_id = manifest.opportunity_identity.opportunity_id if manifest.opportunity_identity else ""
                    capture = decision_capture_store.get(opp_id) if opp_id else None
                    if capture:
                        evaluation = evaluation_engine.evaluate(
                            manifest=manifest,
                            capture=capture,
                            actual_side=pos.side,
                            actual_quantity=pos.quantity,
                            actual_exit_reason=pos.exit_reason,
                        )
                        if evaluation:
                            evaluation_corpus.save(evaluation)
                    else:
                        log.info("Decision evaluation skipped",
                                 opportunity_id=opp_id or "N/A",
                                 position_id=pos.position_id)
                    return
                except Exception as e:
                    pos_id = event.payload.get("position_id", "?")
                    log.warning(
                        "Learning pipeline attempt failed",
                        position_id=pos_id, attempt=attempt + 1, error=str(e),
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        log.error(
                            "Learning pipeline failed after all retries",
                            position_id=pos_id, error=str(e),
                        )

        event_bus.subscribe("POSITION_UPDATED", _on_position_closed_learning)

        # ── Phase 4.9: Startup Learning Recovery ──
        # Positions closed during reconciliation (or prior restarts) may not have
        # LearningManifests. Recover them idempotently — only processing positions
        # that have zero artifacts.
        try:
            closed_states = {"CLOSED", "ARCHIVED"}
            missing_positions = portfolio_mgr.get_terminal_positions()
            recovered_count = 0
            for pos in missing_positions:
                existing_manifest = learning_corpus.find_by_position_id(pos.position_id)
                if existing_manifest:
                    continue
                evaluation = evaluation_corpus.get_by_position_id(pos.position_id)
                if evaluation is not None:
                    continue
                try:
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
                    manifest = await learning_pipeline.process(snapshot, runtime_config_values=runtime_config_values)
                    opp_id = manifest.opportunity_identity.opportunity_id if manifest.opportunity_identity else ""
                    capture = decision_capture_store.get(opp_id) if opp_id else None
                    if capture:
                        evaluation = evaluation_engine.evaluate(
                            manifest=manifest,
                            capture=capture,
                            actual_side=pos.side,
                            actual_quantity=pos.quantity,
                            actual_exit_reason=pos.exit_reason or "ORPHANED_RECONCILIATION",
                        )
                        if evaluation:
                            evaluation_corpus.save(evaluation)
                    recovered_count += 1
                except Exception as e:
                    log.warning(
                        "Learning recovery failed for position",
                        position_id=pos.position_id, symbol=pos.symbol, error=str(e),
                    )
            if recovered_count > 0:
                log.info("Startup learning recovery complete", recovered=recovered_count)
        except Exception as e:
            log.warning("Startup learning recovery error", error=str(e))

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
                    ("Press Ctrl+C to stop", "dim"),
                ),
                border_style="green",
            )
        )

        tasks_to_gather = [
            ws_task, agg_task, pipeline_task,
            dispatcher_task, llm_task, monitor_task, scan_task,
        ]
        await asyncio.gather(*tasks_to_gather)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    except Exception:
        log.error("Fatal error in main loop", exc_info=True)
        raise
    finally:
        graceful = False
        try:
            rich.print("\n[yellow]Shutting down gracefully...[/]")
            ws_ingestor.stop()
            loop_tasks = [ws_task, agg_task, pipeline_task, dispatcher_task, llm_task, monitor_task, scan_task]

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
            log.info("APEX shutdown complete.")
            rich.print("[green]Goodbye.[/]")
            graceful = True
        except (asyncio.CancelledError, KeyboardInterrupt):
            rich.print("\n[yellow]Shutdown interrupted by signal. Forcing exit.[/]")
        except Exception:
            log.error("Unexpected error during shutdown", exc_info=True)
        finally:
            if not graceful:
                os._exit(0)


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
    except SystemExit:
        pass
    os._exit(0)
