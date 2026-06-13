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

from rich.table import Table

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
from .agent.state_builder import build_state
from .agent.memory_router import retrieve_memories, store_memory
from .agent.prompt_compiler import compile_prompt
from .agent.llm_executor import execute_decision
from .agent.decision_logger import log_decision
from .execution.trade_manager import TradeManager
from .execution.router import execute_trade_decision, round_step_size
from .execution.monitor import monitor_open_trades

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


async def _get_usdt_balance(client) -> float:
    try:
        info = await client.get_account_info()
        for asset in info.get("assets", []):
            if asset["asset"] == "USDT":
                return float(asset.get("walletBalance", 0))
        return 0.0
    except Exception:
        return 0.0


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

    trade_manager = TradeManager()
    await trade_manager.create_trades_table()
    latest_prices: dict[str, float] = {}

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

    ws_task = None
    agg_task = None
    pipeline_task = None
    agent_task = None

    try:
        ws_task = asyncio.create_task(ws_ingestor.start_stream())
        agg_task = asyncio.create_task(_aggregation_catchup_loop(aggregator, log))
        pipeline_task = asyncio.create_task(
            _background_pipeline_worker(candle_queue, ingestor, aggregator, correlation_engine, matrix_store, log, latest_prices)
        )
        if llm_provider:
            agent_task = asyncio.create_task(
                agent_evaluation_loop(
                    anchors=anchors,
                    llm_provider=llm_provider,
                    llm_key=llm_key,
                    model_id=model_id,
                    custom_base_url=custom_base_url,
                    log=log,
                    config=config,
                    binance_client=client,
                    trade_manager=trade_manager,
                )
            )

        monitor_task = asyncio.create_task(
            monitor_open_trades(client, trade_manager, latest_prices)
        )

        rich.print(
            Panel.fit(
                Text.assemble(
                    ("Ingestion pipeline active", "bold green"),
                    "\n",
                    (f"Tracking {len(all_symbols)} symbols via WebSocket", "dim"),
                    "\n",
                    ("Agent evaluation loop active", "bold cyan") if llm_provider else ("Agent loop disabled (no LLM)", "dim yellow"),
                    ("\nPress Ctrl+C to stop", "dim"),
                ),
                border_style="green",
            )
        )

        tasks_to_gather = [ws_task, agg_task, pipeline_task, monitor_task]
        if agent_task:
            tasks_to_gather.append(agent_task)
        await asyncio.gather(*tasks_to_gather)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        rich.print("\n[yellow]Shutting down gracefully...[/]")
        ws_ingestor.stop()
        for t in (ws_task, agg_task, pipeline_task, agent_task, monitor_task):
            if t and not t.done():
                t.cancel()
        ingestor.close()
        aggregator.close()
        matrix_store.close()
        correlation_engine.close()
        trade_manager.close()
        log.info("APEX shutdown complete.")
        rich.print("[green]Goodbye.[/]")


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
    latest_prices: dict = None,
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

            if latest_prices is not None:
                latest_prices[payload.symbol] = payload.close

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


async def agent_evaluation_loop(
    anchors: list[str],
    llm_provider: str,
    llm_key: str,
    model_id: str,
    custom_base_url: str,
    log,
    config: dict = None,
    binance_client=None,
    trade_manager=None,
    interval_seconds: int = 30,
):
    log.info("Agent evaluation loop started", interval_seconds=interval_seconds, anchors=anchors)
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            log.info("Agent evaluation cycle starting", anchors=anchors)

            for symbol in anchors:
                try:
                    state = await build_state(symbol, "5m")

                    memories = await retrieve_memories(state, limit=3)

                    prompt = compile_prompt(state, memories)

                    decision = await execute_decision(
                        prompt=prompt,
                        provider=llm_provider,
                        api_key=llm_key,
                        model=model_id,
                        custom_base_url=custom_base_url,
                    )

                    decision_id = await log_decision(decision, state)

                    table = Table(title=f"Agent Decision — {symbol}", title_style="bold cyan")
                    table.add_column("Field", style="dim", width=20)
                    table.add_column("Value", style="white")
                    table.add_row("Decision ID", str(decision_id))
                    table.add_row("Action", f"[bold {'green' if decision.action == 'BUY' else 'red' if decision.action == 'SELL' else 'yellow'}]{decision.action}[/]")
                    table.add_row("Confidence", f"{decision.confidence:.2%}")
                    table.add_row("Rationale", decision.rationale)
                    table.add_row("Timeframe", decision.suggested_timeframe)
                    table.add_row("Price", str(state.get("current_price", "N/A")))
                    rich.print(table)

                    log.info("DIAGNOSTIC: Agent decision received", symbol=symbol, action=decision.action, confidence=decision.confidence)

                    if decision.action != "HOLD" and config and binance_client and trade_manager:
                        try:
                            log.info("DIAGNOSTIC: Action is not HOLD, entering execution block", symbol=symbol, action=decision.action)

                            balance = await _get_usdt_balance(binance_client)
                            result = await execute_trade_decision(
                                decision=decision,
                                symbol=symbol,
                                current_price=state.get("current_price", 0.0),
                                account_balance=balance,
                                config=config,
                                binance_client=binance_client,
                                trade_manager=trade_manager,
                            )

                            if result["status"] == "EXECUTED":
                                log.info("Trade executed successfully", symbol=symbol, trade_id=result["trade_id"], fill_price=result["fill_price"])
                                rich.print(f"\n[bold green]✅ TRADE EXECUTED: {symbol}[/]")
                                trade_table = Table(title="Execution Details")
                                trade_table.add_column("Field", style="cyan")
                                trade_table.add_column("Value", style="magenta")
                                trade_table.add_row("Trade ID", result["trade_id"])
                                side_label = "LONG" if decision.action == "BUY" else "SHORT"
                                trade_table.add_row("Side", side_label)
                                trade_table.add_row("Fill Price", f"${result['fill_price']:,.2f}")
                                trade_table.add_row("Position Size", f"{result['position_size']}")
                                trade_table.add_row("Stop Loss", f"${result['sl']:,.2f}")
                                trade_table.add_row("Take Profit", f"${result['tp']:,.2f}")
                                rich.print(trade_table)
                            elif result["status"] == "FAILED":
                                log.error("Execution router returned FAILED status", symbol=symbol, error=result.get("error"))
                                rich.print(f"\n[bold red]❌ TRADE FAILED: {symbol} - {result.get('error', 'Unknown error')}[/]")
                        except Exception as e:
                            log.error("FATAL: Execution block crashed", symbol=symbol, error=str(e), traceback=True)
                            rich.print(f"\n[bold red]💥 EXECUTION ERROR: {symbol} - {e}[/]")
                    else:
                        log.info("DIAGNOSTIC: Action is HOLD or no config/client/trade_manager, skipping execution", symbol=symbol, action=decision.action)

                except Exception as e:
                    log.error(
                        "Agent evaluation failed for symbol",
                        symbol=symbol,
                        error=str(e),
                        traceback=traceback.format_exc(),
                    )
                    rich.print(f"[bold red][FAIL] Agent evaluation failed for {symbol}: {e}[/]")

        except asyncio.CancelledError:
            log.info("Agent evaluation loop cancelled")
            break
        except Exception as e:
            log.error("Agent evaluation loop error", error=str(e), traceback=traceback.format_exc())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rich.print("\n[yellow]Interrupted.[/]")
    except SystemExit:
        pass
    finally:
        os._exit(0)
