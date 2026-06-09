import asyncio
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

CONFIG_PATH = "config.toml"
KEYS_FILE = "keys.enc"


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
        raise RuntimeError(f"Failed to decrypt keys.enc: {e}. The file may be corrupted or the passphrase is wrong.") from e


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

    config = load_config(CONFIG_PATH)

    passphrase_cache = []
    try:
        binance_key, binance_secret, llm_key = get_api_keys(passphrase_cache)
    except RuntimeError as e:
        rich.print(f"[red]{e}[/]")
        sys.exit(1)

    init_logging()
    log = get_logger("main")
    log.info("APEX starting", mode=config["binance"]["mode"])

    llm_provider = config["llm"]["provider"]
    custom_base_url = config["llm"].get("custom_base_url", "")

    llm_ok = False
    try:
        log.info("Verifying LLM connection", provider=llm_provider)
        registry = LLMRegistry(
            provider=llm_provider,
            api_key=llm_key,
            custom_base_url=custom_base_url,
        )
        await registry.verify_connection()
        rich.print("[bold green]✅ LLM Connected[/]")
        llm_ok = True
    except Exception as e:
        log.error("LLM connection failed", error=str(e), traceback=traceback.format_exc())
        rich.print(f"[bold red]❌ LLM Failed: {e}[/]")

    binance_ok = False
    balance_str = "N/A"
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
        rich.print(f"[bold green]✅ Binance Connected. Balance: {balance_str} USDT[/]")
        binance_ok = True
    except (BinanceClientError, Exception) as e:
        log.error("Binance connection failed", error=str(e), traceback=traceback.format_exc())
        rich.print(f"[bold red]❌ Binance Failed: {e}[/]")
        client = None

    if not (llm_ok and binance_ok):
        log.warning("One or more connections failed. Exiting.")
        sys.exit(1)

    anchors = config["universe"].get("anchors", [])
    alternates = config["universe"].get("alternates", [])
    all_symbols = anchors + alternates

    if not all_symbols:
        log.warning("No symbols configured in universe.anchors / universe.alternates. Exiting.")
        sys.exit(1)

    ingestor = Ingestor(mode=config["binance"]["mode"], binance_client=client)
    aggregator = Aggregator()
    ws_ingestor = WebSocketIngestor(
        symbols=all_symbols,
        mode=config["binance"]["mode"],
        ingestor=ingestor,
    )

    ws_task = None
    agg_task = None

    try:
        ws_task = asyncio.create_task(ws_ingestor.start_stream())
        agg_task = asyncio.create_task(_aggregation_loop(aggregator))

        rich.print(
            Panel.fit(
                Text.assemble(
                    ("🚀 Ingestion pipeline active", "bold green"),
                    "\n",
                    (f"Tracking {len(all_symbols)} symbols via WebSocket", "dim"),
                    "\n",
                    ("Press Ctrl+C to stop", "dim"),
                ),
                border_style="green",
            )
        )

        await ws_task
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        rich.print("\n[yellow]Shutting down gracefully...[/]")
        ws_ingestor.stop()
        if ws_task and not ws_task.done():
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
        if agg_task and not agg_task.done():
            agg_task.cancel()
        ingestor.close()
        log.info("APEX shutdown complete.")
        rich.print("[green]Goodbye.[/]")
        sys.exit(0)


async def _aggregation_loop(aggregator: Aggregator, interval: int = 60):
    log = get_logger("aggregator_loop")
    while True:
        try:
            await asyncio.sleep(interval)
            log.info("Running aggregation cycle")
            await aggregator.aggregate_timeframes()
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("Aggregation cycle failed", error=str(e), traceback=traceback.format_exc())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rich.print("\n[yellow]Interrupted.[/]")
        sys.exit(0)
