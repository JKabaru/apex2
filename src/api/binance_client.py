import asyncio
import hashlib
import hmac
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

import aiohttp
import structlog


def _round_step_size(quantity: float, step_size: float = 0.001) -> str:
    qty = Decimal(str(quantity))
    step = Decimal(str(step_size))
    precision = abs(step.as_tuple().exponent)
    valid_qty = (qty // step) * step
    return str(valid_qty.quantize(Decimal(10) ** -precision, rounding=ROUND_DOWN))

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
        self._session = None
        self.time_offset = 0
        self._step_size_cache = {}
        self._tick_size_cache = {}
        self._algo_locks: dict[str, asyncio.Lock] = {}

    async def sync_time(self):
        self.log.info("Synchronizing time with Binance...")
        try:
            url = f"{self.base_url}/fapi/v1/time"
            session = await self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    server_time = data.get("serverTime")
                    if server_time:
                        local_time = int(time.time() * 1000)
                        self.time_offset = server_time - local_time
                        self.log.info("Time synchronized", offset_ms=self.time_offset)
                else:
                    self.log.warning("Time sync response status not 200", status=resp.status)
        except Exception as e:
            self.log.warning("Failed to synchronize time with Binance, using local time", error=str(e))

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            self.log.info("Closing Binance client session...")
            await self._session.close()

    def _sign_request(self, params: dict) -> str:
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{query}&signature={signature}"

    async def _public_get(self, path: str, params: dict = None) -> dict:
        if params is None:
            params = {}
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"

        session = await self._get_session()
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            body = await resp.text()
            if resp.status != 200:
                self.log.error(
                    "Binance Public API error",
                    path=path,
                    status=resp.status,
                    body=body[:500],
                )
                raise BinanceClientError(
                    f"HTTP {resp.status} on {path}: {body[:200]}"
                )
            return await resp.json()

    async def _signed_get(self, path: str, params: dict = None) -> dict:
        if params is None:
            params = {}
        params["timestamp"] = int(time.time() * 1000) + self.time_offset
        params["recvWindow"] = 5000

        signature = self._sign_request(params)
        url = f"{self.base_url}{path}?{signature}"
        headers = {"X-MBX-APIKEY": self.api_key}

        session = await self._get_session()
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
        params["timestamp"] = int(time.time() * 1000) + self.time_offset
        params["recvWindow"] = 5000

        signature = self._sign_request(params)
        url = f"{self.base_url}{path}?{signature}"
        headers = {"X-MBX-APIKEY": self.api_key}

        session = await self._get_session()
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            body = await resp.text()
            return resp.status, body

    async def _signed_post(self, path: str, params: dict = None) -> dict:
        if params is None:
            params = {}
        params["timestamp"] = int(time.time() * 1000) + self.time_offset
        params["recvWindow"] = 5000

        signature = self._sign_request(params)
        url = f"{self.base_url}{path}?{signature}"
        headers = {"X-MBX-APIKEY": self.api_key}

        session = await self._get_session()
        async with session.post(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            body = await resp.text()
            if resp.status not in (200, 201):
                self.log.error(
                    "Binance POST API error",
                    path=path,
                    status=resp.status,
                    body=body[:500],
                )
                raise BinanceClientError(
                    f"HTTP {resp.status} on POST {path}: {body[:200]}"
                )
            return await resp.json()

    async def _signed_delete(self, path: str, params: dict = None) -> dict:
        if params is None:
            params = {}
        params["timestamp"] = int(time.time() * 1000) + self.time_offset
        params["recvWindow"] = 5000

        signature = self._sign_request(params)
        url = f"{self.base_url}{path}?{signature}"
        headers = {"X-MBX-APIKEY": self.api_key}

        session = await self._get_session()
        async with session.delete(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            body = await resp.text()
            if resp.status not in (200, 201, 204):
                self.log.error(
                    "Binance DELETE API error",
                    path=path,
                    status=resp.status,
                    body=body[:500],
                )
                raise BinanceClientError(
                    f"HTTP {resp.status} on DELETE {path}: {body[:200]}"
                )
            if body:
                return await resp.json()
            return {}

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        self.log.info("Setting leverage", symbol=symbol, leverage=leverage)
        try:
            params = {
                "symbol": symbol,
                "leverage": leverage,
            }
            data = await self._signed_post("/fapi/v1/leverage", params)
            self.log.info(
                "Leverage set",
                symbol=symbol,
                leverage=data.get("leverage"),
            )
            return data
        except BinanceClientError:
            raise
        except aiohttp.ClientError as e:
            self.log.error("HTTP request failed setting leverage", error=str(e))
            raise BinanceClientError(f"HTTP request failed setting leverage: {e}") from e
        except Exception as e:
            self.log.error(
                "Unexpected error setting leverage",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            raise BinanceClientError(f"Unexpected error setting leverage: {e}") from e

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        position_side: str,
        new_client_order_id: str = None,
    ) -> dict:
        self.log.info(
            "Placing market order",
            symbol=symbol,
            side=side,
            position_side=position_side,
            quantity=quantity,
        )
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
                "positionSide": position_side,
                "newOrderRespType": "RESULT",
            }
            if new_client_order_id:
                params["newClientOrderId"] = new_client_order_id
            data = await self._signed_post("/fapi/v1/order", params)
            if data.get("status") != "FILLED":
                raise BinanceClientError(f"Order not filled. Status: {data.get('status')}")

            avg_price = float(data.get("avgPrice", 0.0))
            executed_qty = float(data.get("executedQty", 0.0))
            total_commission = sum(float(fill.get("commission", 0.0)) for fill in data.get("fills", []))

            result = {
                "orderId": data.get("orderId"),
                "clientOrderId": data.get("clientOrderId", ""),
                "avgPrice": avg_price,
                "executedQty": executed_qty,
                "cumQuote": float(data.get("cumQuote", 0.0)),
                "commission": total_commission,
                "status": data.get("status", ""),
                "fills": data.get("fills", []),
            }
            self.log.info(
                "Market order placed",
                symbol=symbol,
                order_id=result["orderId"],
                client_order_id=result["clientOrderId"],
                avg_price=result["avgPrice"],
                executed_qty=result["executedQty"],
                commission=total_commission,
                status=result["status"],
            )
            return result

        except BinanceClientError:
            raise
        except aiohttp.ClientError as e:
            self.log.error("HTTP request failed for market order", error=str(e))
            raise BinanceClientError(f"HTTP request failed for market order: {e}") from e
        except Exception as e:
            self.log.error(
                "Unexpected error placing market order",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            raise BinanceClientError(f"Unexpected error placing market order: {e}") from e

    @staticmethod
    def _short_id(position_id: str) -> str:
        return position_id.replace("-", "")[:16]

    @staticmethod
    def _round_price(price: float, tick_size: float) -> str:
        prec = abs(Decimal(str(tick_size)).as_tuple().exponent)
        qty_dec = Decimal(str(price))
        step = Decimal(str(tick_size))
        rounded = (qty_dec // step) * step
        return str(rounded.quantize(Decimal(10) ** -prec, rounding=ROUND_DOWN))

    @asynccontextmanager
    async def algo_lock(self, symbol: str):
        """Per-symbol lock that serializes all algo order operations for a symbol.
        Prevents -4130 races when multiple coroutines concurrently place/cancel
        conditional orders for the same symbol."""
        lock = self._algo_locks.setdefault(symbol, asyncio.Lock())
        async with lock:
            yield

    async def place_algo_stop(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        position_id: str,
        client_algo_id: str = None,
        estimated_qty: float = None,
        current_price: float = None,
    ) -> dict:
        if client_algo_id is None:
            client_algo_id = f"SL_{self._short_id(position_id)}"
        tick_size = await self.get_symbol_price_filter(symbol)
        price_str = self._round_price(stop_price, tick_size)
        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "triggerPrice": price_str,
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "clientAlgoId": client_algo_id,
        }
        if estimated_qty is not None and current_price is not None and (estimated_qty * current_price) >= 100.0:
            params["priceProtect"] = "true"
        self.log.info(
            "Placing algo STOP_MARKET",
            symbol=symbol, side=side, trigger_price=price_str,
            client_algo_id=client_algo_id,
        )
        try:
            data = await self._signed_post("/fapi/v1/algoOrder", params)
            self.log.info(
                "Algo STOP_MARKET placed",
                symbol=symbol, algo_id=data.get("algoId"),
                client_algo_id=client_algo_id,
            )
            return data
        except BinanceClientError:
            raise
        except Exception as e:
            self.log.error("Unexpected error placing algo STOP_MARKET", error=str(e))
            raise BinanceClientError(f"Algo STOP_MARKET failed: {e}") from e

    async def place_algo_tp(
        self,
        symbol: str,
        side: str,
        tp_price: float,
        position_id: str,
        client_algo_id: str = None,
        estimated_qty: float = None,
        current_price: float = None,
    ) -> dict:
        if client_algo_id is None:
            client_algo_id = f"TP_{self._short_id(position_id)}"
        tick_size = await self.get_symbol_price_filter(symbol)
        price_str = self._round_price(tp_price, tick_size)
        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "triggerPrice": price_str,
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "clientAlgoId": client_algo_id,
        }
        if estimated_qty is not None and current_price is not None and (estimated_qty * current_price) >= 100.0:
            params["priceProtect"] = "true"
        self.log.info(
            "Placing algo TAKE_PROFIT_MARKET",
            symbol=symbol, side=side, trigger_price=price_str,
            client_algo_id=client_algo_id,
        )
        try:
            data = await self._signed_post("/fapi/v1/algoOrder", params)
            self.log.info(
                "Algo TAKE_PROFIT_MARKET placed",
                symbol=symbol, algo_id=data.get("algoId"),
                client_algo_id=client_algo_id,
            )
            return data
        except BinanceClientError:
            raise
        except Exception as e:
            self.log.error("Unexpected error placing algo TAKE_PROFIT_MARKET", error=str(e))
            raise BinanceClientError(f"Algo TAKE_PROFIT_MARKET failed: {e}") from e

    async def cancel_algo_by_client_id(self, symbol: str, client_algo_id: str) -> dict:
        self.log.info(
            "Cancelling algo order by clientAlgoId",
            symbol=symbol, client_algo_id=client_algo_id,
        )
        try:
            params = {
                "symbol": symbol,
                "origClientAlgoId": client_algo_id,
            }
            data = await self._signed_delete("/fapi/v1/algoOrder", params)
            self.log.info(
                "Algo order cancelled by clientAlgoId",
                symbol=symbol, client_algo_id=client_algo_id,
            )
            return data
        except BinanceClientError:
            raise
        except Exception as e:
            self.log.error("Unexpected error cancelling algo order by clientAlgoId", error=str(e))
            raise BinanceClientError(f"Cancel algo by clientAlgoId failed: {e}") from e

    async def cancel_algo_by_algo_id(self, symbol: str, algo_id: int | str) -> dict:
        self.log.info(
            "Cancelling algo order by algoId",
            symbol=symbol, algo_id=algo_id,
        )
        try:
            params = {
                "symbol": symbol,
                "algoId": algo_id,
            }
            data = await self._signed_delete("/fapi/v1/algoOrder", params)
            self.log.info(
                "Algo order cancelled by algoId",
                symbol=symbol, algo_id=algo_id,
            )
            return data
        except BinanceClientError:
            raise
        except Exception as e:
            self.log.error("Unexpected error cancelling algo order by algoId", error=str(e))
            raise BinanceClientError(f"Cancel algo by algoId failed: {e}") from e

    async def cancel_algo_order(
        self,
        symbol: str,
        client_algo_id: str = None,
        algo_id: int | str = None,
    ) -> dict:
        """Cancel an algo order by clientAlgoId (preferred) or algoId."""
        if client_algo_id:
            return await self.cancel_algo_by_client_id(symbol, client_algo_id)
        if algo_id is not None:
            return await self.cancel_algo_by_algo_id(symbol, algo_id)
        raise BinanceClientError("Either client_algo_id or algo_id is required to cancel algo order")

    async def cancel_all_open_orders(self, symbol: str) -> dict:
        self.log.info("Cancelling all open orders", symbol=symbol)
        try:
            params = {"symbol": symbol}
            data = await self._signed_delete("/fapi/v1/allOpenOrders", params)
            self.log.info("All open orders cancelled", symbol=symbol)
            return data
        except BinanceClientError:
            raise
        except aiohttp.ClientError as e:
            self.log.error("HTTP request failed cancelling open orders", error=str(e))
            raise BinanceClientError(f"HTTP request failed cancelling open orders: {e}") from e
        except Exception as e:
            self.log.error(
                "Unexpected error cancelling open orders",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            raise BinanceClientError(f"Unexpected error cancelling open orders: {e}") from e

    async def get_open_positions(self) -> list[dict]:
        self.log.info("Fetching open positions from Binance...")
        try:
            data = await self._signed_get("/fapi/v2/positionRisk")
            open_positions = []
            for pos in data:
                amt = float(pos.get("positionAmt", 0.0))
                if amt != 0.0:
                    open_positions.append({
                        "symbol": pos.get("symbol"),
                        "position_amt": amt,
                        "entry_price": float(pos.get("entryPrice", 0.0)),
                        "mark_price": float(pos.get("markPrice", 0.0)),
                        "unrealized_profit": float(pos.get("unRealizedProfit", 0.0)),
                        "liquidation_price": float(pos.get("liquidationPrice", 0.0)),
                        "leverage": int(pos.get("leverage", 0)),
                        "margin_type": pos.get("marginType", ""),
                        "position_side": pos.get("positionSide", ""),
                        "notional": float(pos.get("notional", 0.0)),
                        "max_notional_value": float(pos.get("maxNotionalValue", 0.0)),
                        "isolated": bool(pos.get("isolated", False)),
                        "isolated_margin": float(pos.get("isolatedMargin", 0.0)),
                        "break_even_price": float(pos.get("breakEvenPrice", 0.0)),
                        "update_time": int(pos.get("updateTime", 0)),
                    })
            self.log.info("Open positions retrieved", count=len(open_positions))
            return open_positions
        except Exception as e:
            self.log.error("Failed to get open positions", error=str(e))
            raise BinanceClientError(f"Failed to get open positions: {e}") from e

    async def get_order_status(
        self,
        symbol: str,
        orig_client_order_id: str = None,
        order_id: int = None,
    ) -> dict:
        params = {"symbol": symbol}
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        elif order_id:
            params["orderId"] = order_id
        else:
            raise BinanceClientError("Either orig_client_order_id or order_id is required")

        self.log.info("Fetching order status", symbol=symbol, params=params)
        try:
            return await self._signed_get("/fapi/v1/order", params)
        except BinanceClientError:
            raise
        except aiohttp.ClientError as e:
            self.log.error("HTTP request failed for order status", error=str(e))
            raise BinanceClientError(f"HTTP request failed for order status: {e}") from e
        except Exception as e:
            self.log.error("Unexpected error fetching order status", error=str(e), traceback=traceback.format_exc())
            raise BinanceClientError(f"Unexpected error fetching order status: {e}") from e

    async def get_open_orders(self, symbol: str = None) -> list[dict]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        self.log.info("Fetching open orders", symbol=symbol or "all")
        try:
            return await self._signed_get("/fapi/v1/openOrders", params)
        except BinanceClientError:
            raise
        except aiohttp.ClientError as e:
            self.log.error("HTTP request failed for open orders", error=str(e))
            raise BinanceClientError(f"HTTP request failed for open orders: {e}") from e
        except Exception as e:
            self.log.error("Unexpected error fetching open orders", error=str(e), traceback=traceback.format_exc())
            raise BinanceClientError(f"Unexpected error fetching open orders: {e}") from e

    async def get_open_algo_orders(self, symbol: str = None) -> list[dict]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        self.log.info("Fetching open algo orders", symbol=symbol or "all")
        try:
            return await self._signed_get("/fapi/v1/openAlgoOrders", params)
        except BinanceClientError:
            raise
        except aiohttp.ClientError as e:
            self.log.error("HTTP request failed for open algo orders", error=str(e))
            raise BinanceClientError(f"HTTP request failed for open algo orders: {e}") from e
        except Exception as e:
            self.log.error("Unexpected error fetching open algo orders", error=str(e), traceback=traceback.format_exc())
            raise BinanceClientError(f"Unexpected error fetching open algo orders: {e}") from e

    async def get_position_mode(self) -> str:
        self.log.info("Fetching position mode...")
        try:
            data = await self._signed_get("/fapi/v1/positionSide/dual")
            is_dual = data.get("dualSidePosition", False)
            mode = "HEDGE" if is_dual else "ONE_WAY"
            self.log.info("Position mode retrieved", mode=mode)
            return mode
        except BinanceClientError:
            raise
        except aiohttp.ClientError as e:
            self.log.error("HTTP request failed for position mode", error=str(e))
            raise BinanceClientError(f"HTTP request failed for position mode: {e}") from e
        except Exception as e:
            self.log.error("Unexpected error fetching position mode", error=str(e), traceback=traceback.format_exc())
            raise BinanceClientError(f"Unexpected error fetching position mode: {e}") from e

    async def get_symbol_step_size(self, symbol: str) -> float:
        if symbol in self._step_size_cache:
            return self._step_size_cache[symbol]

        self.log.info("Fetching exchangeInfo to populate step size cache...", symbol=symbol)
        try:
            data = await self._public_get("/fapi/v1/exchangeInfo")
            symbols_info = data.get("symbols", [])
            for sym_info in symbols_info:
                sym = sym_info.get("symbol")
                for filt in sym_info.get("filters", []):
                    if filt.get("filterType") == "LOT_SIZE":
                        self._step_size_cache[sym] = float(filt["stepSize"])
                    elif filt.get("filterType") == "PRICE_FILTER":
                        self._tick_size_cache[sym] = float(filt["tickSize"])
                        
            if symbol in self._step_size_cache:
                step_size = self._step_size_cache[symbol]
                self.log.info("Step size retrieved and cached", symbol=symbol, step_size=step_size)
                return step_size
                
            raise BinanceClientError(f"Symbol {symbol} not found in exchangeInfo after fetching")
        except Exception as e:
            self.log.error("Failed to fetch step size from exchangeInfo", symbol=symbol, error=str(e))
            raise BinanceClientError(f"Failed to fetch step size: {e}") from e

    async def get_symbol_price_filter(self, symbol: str) -> float:
        if symbol in self._tick_size_cache:
            return self._tick_size_cache[symbol]

        self.log.info("Fetching exchangeInfo to populate tick size cache...", symbol=symbol)
        try:
            data = await self._public_get("/fapi/v1/exchangeInfo")
            symbols_info = data.get("symbols", [])
            for sym_info in symbols_info:
                sym = sym_info.get("symbol")
                for filt in sym_info.get("filters", []):
                    if filt.get("filterType") == "PRICE_FILTER":
                        self._tick_size_cache[sym] = float(filt["tickSize"])
                    elif filt.get("filterType") == "LOT_SIZE":
                        self._step_size_cache[sym] = float(filt["stepSize"])

            if symbol in self._tick_size_cache:
                tick_size = self._tick_size_cache[symbol]
                self.log.info("Tick size retrieved and cached", symbol=symbol, tick_size=tick_size)
                return tick_size

            raise BinanceClientError(f"Symbol {symbol} not found in exchangeInfo after fetching")
        except Exception as e:
            self.log.error("Failed to fetch tick size from exchangeInfo", symbol=symbol, error=str(e))
            raise BinanceClientError(f"Failed to fetch tick size: {e}") from e

    async def force_close_position(self, symbol: str, position_amt: float):
        self.log.info("Force closing position", symbol=symbol, amt=position_amt)
        try:
            side = "SELL" if position_amt > 0 else "BUY"
            qty_val = abs(position_amt)
            step_size = await self.get_symbol_step_size(symbol)
            qty_str = _round_step_size(qty_val, step_size)

            await self.place_market_order(
                symbol=symbol,
                side=side,
                quantity=qty_str,
                position_side="BOTH",
            )
            self.log.info("Force closed position successfully", symbol=symbol)
        except Exception as e:
            self.log.error("Failed to force close position", symbol=symbol, error=str(e))
            raise BinanceClientError(f"Failed to force close position: {e}") from e

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

    async def get_commission_rate(self, symbol: str) -> dict:
        self.log.info("Fetching commission rate", symbol=symbol)
        try:
            params = {"symbol": symbol}
            return await self._signed_get("/fapi/v1/commissionRate", params)
        except BinanceClientError:
            raise
        except aiohttp.ClientError as e:
            self.log.error("HTTP request failed for commission rate", error=str(e))
            raise BinanceClientError(f"HTTP request failed for commission rate: {e}") from e
        except Exception as e:
            self.log.error("Unexpected error fetching commission rate", error=str(e), traceback=traceback.format_exc())
            raise BinanceClientError(f"Unexpected error fetching commission rate: {e}") from e

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

                raw = await self._public_get("/fapi/v1/klines", params)

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
