import sys
import traceback
import asyncio
import time
import structlog

from rich.console import Console

from .providers import OpenAICompatibleProvider, AnthropicProvider

console = Console()

PROVIDER_MAP: dict[str, tuple[str, str | None]] = {
    "opencode": ("https://opencode.ai/zen/v1", "OpenCode Zen"),
    "openai": ("https://api.openai.com/v1", "OpenAI"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA"),
    "z_ai": ("https://api.z.ai/api/coding/paas/v4", "Z.ai"),
    "ollama": ("http://localhost:11434/v1", "Ollama"),
    "custom": (None, "Custom OpenAI-compatible"),
    "anthropic": (None, "Anthropic"),
}

FREE_MODEL_KEYWORDS = ["free", "mini", "tiny", "nano", "pickle", "flash"]

KNOWN_MODELS: dict[str, list[str]] = {
    "opencode": ["opencode-zen-v4", "opencode-zen-v3"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini", "o3-mini"],
    "nvidia": ["meta/llama-3.3-70b-instruct", "meta/llama-3.1-405b-instruct", "mistralai/mistral-large-2-instruct"],
    "z_ai": ["gpt-4o", "deepseek-chat"],
    "anthropic": ["claude-sonnet-4-20250514", "claude-3.5-sonnet-20241022", "claude-3-haiku-20240307"],
    "ollama": [],
    "custom": [],
}

DEGRADED_COOLDOWN_SECONDS = 60.0


class RateLimitError(Exception):
    """Raised when an LLM provider is rate-limited (HTTP 429)."""


def _is_rate_limit_error(exc: Exception) -> bool:
    error_str = str(exc).lower()
    return (
        isinstance(exc, RateLimitError)
        or "429" in error_str
        or "too many requests" in error_str
        or "rate limit" in error_str
    )


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
        self._llm_degraded = False
        self._cooldown_until = 0.0

        self._llm_provider = _build_provider(provider)
        self._client = self._llm_provider.build_client(api_key, custom_base_url)

    def is_degraded(self) -> bool:
        if self._llm_degraded and time.time() >= self._cooldown_until:
            self._llm_degraded = False
            self._cooldown_until = 0.0
        return self._llm_degraded

    def _mark_degraded(self, reason: str) -> None:
        logger = structlog.get_logger("llm_registry")
        self._llm_degraded = True
        self._cooldown_until = time.time() + DEGRADED_COOLDOWN_SECONDS
        logger.warning(
            "LLM provider entering degraded mode",
            provider=self.provider,
            reason=reason,
            cooldown_seconds=DEGRADED_COOLDOWN_SECONDS,
        )

    def _ensure_model(self, model: str) -> str:
        return model or self.model_id or ""

    async def fetch_available_models(self) -> list[str]:
        return await self._llm_provider.fetch_models(self._client)

    async def chat_completion(
        self, model: str, messages: list, **kwargs
    ) -> str:
        logger = structlog.get_logger("llm_registry")
        if self.is_degraded():
            raise RateLimitError(
                f"LLM provider '{self.provider}' is in degraded mode (rate-limited)"
            )

        max_attempts = 4
        timeout = kwargs.pop("timeout", 90.0)
        for attempt in range(1, max_attempts + 1):
            try:
                result = await asyncio.wait_for(
                    self._llm_provider.chat_completion(
                        self._client, self._ensure_model(model), messages, **kwargs
                    ),
                    timeout=timeout,
                )
                result = result.strip() if result else ""
                if not result:
                    raise ValueError("LLM returned empty response")
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    "LLM request timed out",
                    timeout=timeout,
                    attempt=attempt,
                )
                if attempt < max_attempts:
                    wait_time = 5 * (2 ** (attempt - 1))
                    await asyncio.sleep(wait_time)
                    continue
                raise TimeoutError(f"LLM request timed out after {max_attempts} attempts")
            except RateLimitError:
                self._mark_degraded("rate limit during chat_completion")
                raise
            except IndexError as e:
                logger.warning(
                    "LLM response missing choices array (likely rate-limited), backing off",
                    attempt=attempt,
                )
                if attempt < max_attempts:
                    wait_time = 5 * (2 ** (attempt - 1))
                    await asyncio.sleep(wait_time)
                    continue
                self._mark_degraded("empty choices after retries")
                raise RateLimitError("LLM returned empty choices after retries") from e
            except Exception as e:
                error_str = str(e)
                if _is_rate_limit_error(e):
                    self._mark_degraded(error_str)
                    raise RateLimitError(error_str) from e
                if ("429" in error_str or "empty" in error_str.lower()) and attempt < max_attempts:
                    wait_time = 5 * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM request failed, backing off",
                        wait_time=wait_time,
                        attempt=attempt,
                        error=error_str,
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
                if _is_rate_limit_error(e):
                    self._mark_degraded(str(e))
                    console.print(
                        "[yellow][WARN] LLM rate-limited during verification; "
                        "continuing in degraded mode.[/]"
                    )
                    return True
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
            if _is_rate_limit_error(e):
                self._mark_degraded(str(e))
                console.print(
                    "[yellow][WARN] LLM rate-limited during verification; "
                    "continuing in degraded mode.[/]"
                )
                return True
            console.print(f"\n[yellow]Connection test failed: {e}[/]")
            raise
