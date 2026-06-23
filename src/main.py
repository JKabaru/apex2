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
from .db.portfolio_store import PortfolioStore
from .services.portfolio_manager import PortfolioManager
from .services.llm_scheduler import LLMScheduler
from .services.market_context import MarketContextService
from .services.risk_manager import RiskManager
from .services.execution import ExecutionService
from .services.position_manager import PositionManager
from .services.scanner import MarketScanner

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


def get_api_keys(passphrase_cache: list) -> tuple:
    binance_key = binance_secret = llm_key = None

    try:
        binance_key = keyring.get_password("apex", "binance_key")
        binance_secret = keyring.get_password("apex", "binance_secret")
        llm_key = keyring.get_password("apex", "llm_key")
        if all([binance_key, binance_secret, llm_key]):
            return binance_key, binance_secret, llm_key
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
        return keys["binance_key"], keys["binance_secret"], keys["llm_key"]
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
    for service in ("binance_key", "binance_secret", "llm_key"):
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

    init_agent_params()

    passphrase_cache = []
    try:
        binance_key, binance_secret, llm_key = get_api_keys(passphrase_cache)
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

    llm_ok = False
    registry = None
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
            rich.print("[bold green][OK] LLM Connected[/]")
            llm_ok = True
        except Exception as e:
            log.error("LLM connection failed", error=str(e), traceback=traceback.format_exc())
            rich.print(f"[bold red][FAIL] LLM Failed: {e}[/]")

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

    # --- Phase 4.1 Core Services ---
    event_bus = EventBus()
    portfolio_store = PortfolioStore()
    portfolio_store.create_schema()
    portfolio_mgr = PortfolioManager(portfolio_store, event_bus)

    # --- Position Mode Verification ---
    try:
        position_mode = await client.get_position_mode()
        log.info("Position mode verified", mode=position_mode)
        if position_mode != "ONE_WAY":
            log.warning("Hedge mode detected; execution assumes ONE_WAY (positionSide=BOTH)", mode=position_mode)
    except Exception as e:
        log.warning("Failed to verify position mode, assuming ONE_WAY", error=str(e))

    # --- Startup Reconciliation (blocks scanner start) ---
    try:
        recon_result = await portfolio_mgr.reconcile(client)
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

    llm_scheduler = LLMScheduler(registry=registry, model=model_id, audit_logger=portfolio_store)
    context = MarketContextService()
    risk_mgr = RiskManager()
    execution_svc = ExecutionService(client, event_bus, config)
    position_mgr = PositionManager(portfolio_mgr, execution_svc, llm_scheduler, event_bus)
    scanner = MarketScanner(event_bus)

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
        monitor_task = asyncio.create_task(position_mgr.monitor_positions(context))
        scan_task = asyncio.create_task(
            scanner_loop(scanner, alternates, context, risk_mgr, llm_scheduler, portfolio_mgr)
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
            log.info("APEX shutdown complete.")
            rich.print("[green]Goodbye.[/]")
        except (asyncio.CancelledError, KeyboardInterrupt):
            rich.print("\n[yellow]Shutdown interrupted by signal. Forcing exit.[/]")


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
                    render_live_matrix(matrix_store)
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
    risk: RiskManager,
    llm: LLMScheduler,
    portfolio: PortfolioManager,
    interval: int = 30,
) -> None:
    log = get_logger("scanner_loop")
    log.info("Scanner loop started", interval_seconds=interval, alternates=alternates)
    while True:
        try:
            await asyncio.sleep(interval)
            await scanner.run_scan_cycle(alternates, context, risk, llm, portfolio)
        except asyncio.CancelledError:
            log.info("Scanner loop cancelled")
            break
        except Exception as e:
            log.error("Scanner loop error", error=str(e), traceback=traceback.format_exc())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rich.print("\n[yellow]Interrupted.[/]")
    except SystemExit:
        pass
