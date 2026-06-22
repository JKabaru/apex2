import sys
import traceback
import asyncio
import structlog

from rich.console import Console

from .providers import OpenAICompatibleProvider, AnthropicProvider

console = Console()

PROVIDER_MAP: dict[str, tuple[str, str | None]] = {
    "opencode": ("https://opencode.ai/zen/v1", "OpenCode Zen"),
    "openai": ("https://api.openai.com/v1", "OpenAI"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA"),
    "z_ai": ("https://api.z.ai/v1", "Z.ai"),
    "ollama": ("http://localhost:11434/v1", "Ollama"),
    "custom": (None, "Custom OpenAI-compatible"),
    "anthropic": (None, "Anthropic"),
}

FREE_MODEL_KEYWORDS = ["free", "mini", "tiny", "nano", "pickle", "flash"]


def _build_provider(provider_name: str) -> OpenAICompatibleProvider | AnthropicProvider:
    if provider_name == "anthropic":
        return AnthropicProvider()

    base_url, _ = PROVIDER_MAP.get(provider_name, (None, ""))
    display = PROVIDER_MAP.get(provider_name, (None, provider_name))[1]

    if base_url is not None:
        return OpenAICompatibleProvider(provider_name, base_url)
    else:
        return OpenAICompatibleProvider(provider_name, "")


class LLMRegistry:
    def __init__(
        self,
        provider: str,
        api_key: str,
        custom_base_url: str | None = None,
        model_id: str | None = None,
    ):
        if provider not in PROVIDER_MAP:
            allowed = ", ".join(PROVIDER_MAP)
            raise ValueError(f"Unknown provider '{provider}'. Supported: {allowed}")

        self.provider = provider
        self.model_id = model_id

        self._llm_provider = _build_provider(provider)
        self._client = self._llm_provider.build_client(api_key, custom_base_url)

    def _ensure_model(self, model: str) -> str:
        return model or self.model_id or ""

    async def fetch_available_models(self) -> list[str]:
        return await self._llm_provider.fetch_models(self._client)

    async def chat_completion(
        self, model: str, messages: list, **kwargs
    ) -> str:
        logger = structlog.get_logger("llm_registry")
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._llm_provider.chat_completion(
                    self._client, self._ensure_model(model), messages, **kwargs
                )
            except Exception as e:
                if "429" in str(e) and attempt < max_attempts:
                    wait_time = 5 * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM Rate limited (429). Backing off...",
                        wait_time=wait_time,
                        attempt=attempt,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise

    async def verify_connection(self, model: str | None = None) -> bool:
        target = model or self.model_id
        if not target:
            try:
                models = await self.fetch_available_models()
            except Exception as e:
                raise ConnectionError(
                    f"No model specified and cannot fetch model list: {e}"
                ) from e

            if not models:
                raise ConnectionError(
                    "No model specified and provider returned no models."
                )

            for kw in FREE_MODEL_KEYWORDS:
                for mid in models:
                    if kw in mid.lower():
                        target = mid
                        break
                if target:
                    break

            if not target:
                target = models[0]

            console.print(
                f"[dim]Auto-selected model: {target} (from {len(models)} available)[/]"
            )

        console.print(f"[dim]Testing connection with model: {target}[/]")
        try:
            result = await self._llm_provider.verify(self._client, target)
            console.print("[green][OK] LLM connection verified successfully.[/]")
            return result
        except Exception as e:
            console.print(f"\n[yellow]Connection test failed: {e}[/]")
            raise
