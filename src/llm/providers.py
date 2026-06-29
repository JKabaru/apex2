import sys
import traceback
from abc import ABC, abstractmethod
from typing import Any

import httpx
from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError
from rich.console import Console

console = Console()


class LLMProvider(ABC):
    name: str
    base_url: str

    @abstractmethod
    def build_client(self, api_key: str, base_url: str | None = None) -> Any:
        ...

    @abstractmethod
    async def fetch_models(self, client: Any) -> list[str]:
        ...

    @abstractmethod
    async def chat_completion(
        self, client: Any, model: str, messages: list, **kwargs
    ) -> str:
        ...

    @abstractmethod
    async def verify(self, client: Any, model: str) -> bool:
        ...


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url

    def build_client(self, api_key: str, base_url: str | None = None) -> AsyncOpenAI:
        url = base_url or self.base_url
        return AsyncOpenAI(
            base_url=url,
            api_key=api_key,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
            ),
        )

    async def fetch_models(self, client: AsyncOpenAI) -> list[str]:
        try:
            models = await client.models.list()
        except APIConnectionError as e:
            raise ConnectionError(f"Could not connect to {client.base_url}: {e}") from e
        except APIError as e:
            raise ConnectionError(
                f"API error: HTTP {e.status_code} - {e.message}"
            ) from e
        return [m.id for m in models.data]

    async def chat_completion(
        self, client: AsyncOpenAI, model: str, messages: list, **kwargs
    ) -> str:
        use_stream = kwargs.pop("stream", False)
        if use_stream:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            collected = []
            async for chunk in resp:
                if (
                    chunk.choices
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.content
                ):
                    collected.append(chunk.choices[0].delta.content)
            return "".join(collected)
        else:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            if not resp or not hasattr(resp, 'choices') or not resp.choices:
                raise ValueError(
                    "LLM returned empty choices (likely rate-limited or server error)"
                )
            raw_text = resp.choices[0].message.content
            if not raw_text or not raw_text.strip():
                raise ValueError("LLM returned empty text content")
            return raw_text

    async def verify(self, client: AsyncOpenAI, model: str) -> bool:
        try:
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            return True
        except APITimeoutError:
            raise ConnectionError("Connection timed out. The API did not respond within 120 seconds.")
        except APIError as e:
            code = getattr(e, "status_code", "N/A")
            msg = getattr(e, "message", str(e))
            if code == 401:
                raise ConnectionError(f"Authentication failed: {msg}")
            elif code == 403:
                raise ConnectionError(
                    f"API key does not have access to model '{model}': {msg}"
                )
            raise ConnectionError(f"Chat completion failed: HTTP {code} - {msg}")
        except Exception as e:
            raise ConnectionError(f"Connection test failed: {e}") from e


class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.name = "anthropic"
        self.base_url = "https://api.anthropic.com/v1"

    def build_client(self, api_key: str, base_url: str | None = None):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Run: pip install anthropic"
            )
        return AsyncAnthropic(api_key=api_key, max_retries=0)

    async def fetch_models(self, client) -> list[str]:
        return []

    async def chat_completion(
        self, client, model: str, messages: list, **kwargs
    ) -> str:
        system_prompt = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        resp = await client.messages.create(
            model=model,
            system=system_prompt or None,
            messages=chat_messages,
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return resp.content[0].text if resp.content else ""

    async def verify(self, client, model: str) -> bool:
        try:
            await client.messages.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            raise ConnectionError(f"Anthropic connection failed: {e}") from e
