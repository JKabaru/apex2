import sys
import traceback

from openai import AsyncOpenAI, APIError, APIConnectionError
from rich.console import Console

console = Console()


class LLMRegistry:
    PROVIDERS = {
        "opencode": {"base_url": "https://opencode.ai/zen/v1"},
        "openai": {"base_url": "https://api.openai.com/v1"},
        "anthropic": {"base_url": "https://api.anthropic.com/v1"},
        "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1"},
        "ollama": {"base_url": "http://localhost:11434/v1"},
        "custom": {"base_url": None},
    }

    FREE_MODEL_KEYWORDS = ["free", "mini", "tiny", "nano", "pickle", "flash"]

    def __init__(self, provider: str, api_key: str, custom_base_url: str = None, model_id: str = None):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Supported: {', '.join(self.PROVIDERS)}")

        self.provider = provider
        self.api_key = api_key
        self.model_id = model_id

        if provider == "custom":
            self.base_url = custom_base_url
        else:
            self.base_url = self.PROVIDERS[provider]["base_url"]

        if not self.base_url:
            raise ValueError(f"No base_url for provider '{provider}'. Provide custom_base_url.")

    def _pick_model(self, model_ids: list) -> str:
        if self.model_id and self.model_id in model_ids:
            return self.model_id
        for kw in self.FREE_MODEL_KEYWORDS:
            for mid in model_ids:
                if kw in mid.lower():
                    return mid
        return model_ids[0] if model_ids else None

    async def fetch_available_models(self) -> list:
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        try:
            with console.status("[yellow]Fetching available models..."):
                models = await client.models.list()
        except APIConnectionError as e:
            raise ConnectionError(f"Could not connect to {self.base_url}: {e}") from e
        except APIError as e:
            raise ConnectionError(f"API error from {self.provider}: HTTP {e.status_code} - {e.message}") from e
        except Exception as e:
            raise ConnectionError(f"Unexpected error listing models: {e}\n{traceback.format_exc()}") from e

        return [m.id for m in models.data]

    async def verify_connection(self):
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        try:
            with console.status("[yellow]Fetching available models..."):
                models = await client.models.list()
        except APIConnectionError as e:
            raise ConnectionError(f"Could not connect to {self.base_url}: {e}") from e
        except APIError as e:
            raise ConnectionError(f"API error from {self.provider}: HTTP {e.status_code} - {e.message}") from e
        except Exception as e:
            raise ConnectionError(f"Unexpected error listing models: {e}\n{traceback.format_exc()}") from e

        model_ids = [m.id for m in models.data]

        if not model_ids:
            console.print("[yellow]No models returned by provider.[/]")
            return

        display_ids = model_ids[:5]
        suffix = "..." if len(model_ids) > 5 else ""
        console.print(f"[dim]Available models ({len(model_ids)}): {', '.join(display_ids)}{suffix}[/]")

        target_model = self._pick_model(model_ids)
        if not target_model:
            target_model = model_ids[0]

        console.print(f"[dim]Testing with model: {target_model}[/]")

        try:
            stream = await client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

            console.print("[bold cyan]LLM Response:[/] ", end="")
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    sys.stdout.write(content)
                    sys.stdout.flush()
            console.print()
            console.print("[green]✅ LLM connection verified successfully.[/]")
            return True

        except APIError as e:
            console.print(f"\n[yellow]Chat completion failed (model may not support chat): {e.status_code} - {e.message}[/]")
            raise ConnectionError(f"Chat completion failed: {e}") from e
        except Exception as e:
            raise ConnectionError(f"Chat completion error: {e}\n{traceback.format_exc()}") from e
