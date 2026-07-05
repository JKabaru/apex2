import asyncio
import json
import traceback
from dataclasses import dataclass

import structlog
import websockets
from rich import print as rprint

from ..engine.output_mode import is_verbose


@dataclass
class CandlePayload:
    symbol: str
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


LIVE_WS = "wss://fstream.binance.com/ws"
TESTNET_WS = "wss://stream.binancefuture.com/ws"

RECONNECT_DELAY = 3


class WebSocketIngestor:
    def __init__(self, symbols: list, mode: str, candle_queue: asyncio.Queue):
        if mode == "testnet":
            self.ws_url = TESTNET_WS
        else:
            self.ws_url = LIVE_WS
        self.symbols = symbols
        self._candle_queue = candle_queue
        self.log = structlog.get_logger("ws_ingestor")
        self._stop = False

    def stop(self):
        self._stop = True

    async def start_stream(self):
        streams = [f"{symbol.lower()}@kline_1m" for symbol in self.symbols]
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1,
        }

        self.log.info("Starting WebSocket ingestor", symbols=self.symbols, url=self.ws_url)

        backoff_time = 3
        connection_established_time = None

        while not self._stop:
            try:
                self.log.info("Attempting WebSocket connection...")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=180,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    self.log.info("Connection established. Subscribed to streams.", stream_count=len(streams))
                    connection_established_time = asyncio.get_event_loop().time()

                    while not self._stop:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            
                            if connection_established_time is not None:
                                current_time = asyncio.get_event_loop().time()
                                if current_time - connection_established_time >= 60.0:
                                    backoff_time = 3
                                    connection_established_time = None
                                    self.log.info("WebSocket connection maintained for 1m, resetting backoff time.")

                            await self._handle_message(message)
                        except asyncio.TimeoutError:
                            continue

            except (websockets.ConnectionClosed, websockets.WebSocketException, asyncio.TimeoutError, OSError) as e:
                self.log.warning("WebSocket connection error, backing off...", error=str(e), backoff_time=backoff_time)
            except asyncio.CancelledError:
                self.log.info("WebSocket task cancelled")
                break
            except Exception as e:
                self.log.error("Unexpected WebSocket error", error=str(e), traceback=traceback.format_exc())

            if not self._stop:
                self.log.info(f"Reconnecting in {backoff_time}s...")
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 60)

        self.log.info("WebSocket ingestor stopped.")

    async def _handle_message(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            self.log.warning("Invalid JSON from WebSocket", error=str(e))
            return

        if "e" not in data or data.get("e") != "kline":
            return

        k = data.get("k", {})
        if not k:
            return

        symbol = data.get("s", "UNKNOWN")
        open_price = k.get("o", "0")
        close_price = k.get("c", "0")
        is_closed = k.get("x", False)
        open_time = k.get("t", 0)
        close_time = k.get("T", 0)

        if not is_closed:
            if is_verbose():
                rprint(f"[dim][TICK] {symbol}: {close_price}[/]")
            return

        high_price = k.get("h", "0")
        low_price = k.get("l", "0")
        volume = k.get("v", "0")

        if is_verbose():
            rprint(
                f"[green][CLOSED CANDLE] {symbol} | "
                f"Time: {open_time} | "
                f"O: {open_price} | "
                f"H: {high_price} | "
                f"L: {low_price} | "
                f"C: {close_price} | "
                f"V: {volume}[/]"
            )

        payload = CandlePayload(
            symbol=symbol,
            open_time=int(open_time),
            close_time=int(close_time),
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=float(volume),
        )

        await self._candle_queue.put(payload)
