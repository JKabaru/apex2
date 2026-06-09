import hashlib
import hmac
import time
import traceback
from datetime import datetime

import aiohttp
import structlog

LIVE_REST = "https://fapi.binance.com"
TESTNET_REST = "https://testnet.binancefuture.com"


class BinanceClientError(Exception):
    pass


class BinanceClient:
    def __init__(self, mode: str, api_key: str, api_secret: str):
        if mode not in ("testnet", "live"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'testnet' or 'live'.")

        self.base_url = TESTNET_REST if mode == "testnet" else LIVE_REST
        self.api_key = api_key
        self.api_secret = api_secret
        self.log = structlog.get_logger("binance_client")
        self.log.info("BinanceClient initialized", mode=mode, base_url=self.base_url)

    def _sign_request(self, params: dict) -> str:
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{query}&signature={signature}"

    async def _signed_get(self, path: str, params: dict = None) -> dict:
        if params is None:
            params = {}
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000

        signature = self._sign_request(params)
        url = f"{self.base_url}{path}?{signature}"
        headers = {"X-MBX-APIKEY": self.api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                body = await resp.text()
                if resp.status != 200:
                    self.log.error(
                        "Binance API error",
                        path=path,
                        status=resp.status,
                        body=body[:500],
                    )
                    raise BinanceClientError(
                        f"HTTP {resp.status} on {path}: {body[:200]}"
                    )
                return await resp.json()

    async def _signed_get_raw(self, path: str, params: dict = None) -> tuple:
        if params is None:
            params = {}
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000

        signature = self._sign_request(params)
        url = f"{self.base_url}{path}?{signature}"
        headers = {"X-MBX-APIKEY": self.api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                body = await resp.text()
                return resp.status, body

    async def get_account_info(self) -> dict:
        self.log.info("Fetching Futures account info...")
        try:
            data = await self._signed_get("/fapi/v2/account")
            for asset in data.get("assets", []):
                if asset["asset"] == "USDT":
                    balance = float(asset.get("walletBalance", 0))
                    self.log.info(
                        "Account info retrieved",
                        usdt_balance=balance,
                        total_assets=len(data.get("assets", [])),
                        positions_count=len(data.get("positions", [])),
                    )
                    return data

            self.log.info("Account info retrieved", assets_count=len(data.get("assets", [])))
            return data

        except BinanceClientError:
            raise
        except aiohttp.ClientError as e:
            self.log.error("HTTP request failed", error=str(e))
            raise BinanceClientError(f"HTTP request failed: {e}") from e
        except Exception as e:
            self.log.error("Unexpected error fetching account", error=str(e), traceback=traceback.format_exc())
            raise BinanceClientError(f"Unexpected error: {e}") from e

    async def get_historical_klines(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        interval: str = "1m",
    ) -> list:
        self.log.info("Fetching historical klines", symbol=symbol, start=start_time, end=end_time)
        all_candles = []
        current_start = start_time

        try:
            while current_start < end_time:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": end_time,
                    "limit": 1000,
                }

                params["timestamp"] = int(time.time() * 1000)
                params["recvWindow"] = 5000
                signature = self._sign_request(params)
                url = f"{self.base_url}/fapi/v1/klines?{signature}"
                headers = {"X-MBX-APIKEY": self.api_key}

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        body = await resp.text()
                        if resp.status != 200:
                            self.log.error(
                                "Klines API error",
                                symbol=symbol,
                                status=resp.status,
                                body=body[:300],
                            )
                            raise BinanceClientError(
                                f"HTTP {resp.status} on klines for {symbol}: {body[:200]}"
                            )
                        raw = await resp.json()

                if not raw:
                    break

                for candle in raw:
                    parsed = {
                        "open_time": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                        "close_time": int(candle[6]),
                    }
                    all_candles.append(parsed)

                last_open = raw[-1][0]
                if last_open <= current_start:
                    break
                current_start = last_open + 60000

            self.log.info("Historical klines fetched", symbol=symbol, count=len(all_candles))
            return all_candles

        except BinanceClientError:
            raise
        except aiohttp.ClientError as e:
            self.log.error("HTTP request failed fetching klines", error=str(e))
            raise BinanceClientError(f"HTTP request failed for klines: {e}") from e
        except Exception as e:
            self.log.error("Unexpected error fetching klines", error=str(e), traceback=traceback.format_exc())
            raise BinanceClientError(f"Unexpected error fetching klines: {e}") from e
