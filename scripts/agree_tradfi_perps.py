"""One-time script to sign Binance Testnet TradFi Perpetuals agreement."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import keyring
import structlog
from src.api.binance_client import BinanceClient

logger = structlog.get_logger("agree_tradfi")


async def main():
    api_key = keyring.get_password("apex", "binance_key")
    api_secret = keyring.get_password("apex", "binance_secret")
    if not api_key or not api_secret:
        print("ERROR: Binance API keys not found in OS keychain.")
        print("Run the setup wizard first or store keys with:")
        print('  keyring.set_password("apex", "binance_key", "<your_key>")')
        print('  keyring.set_password("apex", "binance_secret", "<your_secret>")')
        sys.exit(1)

    client = BinanceClient(mode="testnet", api_key=api_key, api_secret=api_secret)
    try:
        result = await client.agree_tradfi_perps()
        print(f"SUCCESS: TradFi Perpetuals agreement signed.")
        print(f"Response: {result}")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
