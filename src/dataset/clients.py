import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Literal, Optional

import aiohttp
import requests
import urllib3
from langfuse import observe
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
GIGA_MODELS = Literal["GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max"]


class BaseLLM(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def _call_async(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def __call__(self, messages: list[dict[str, str]]) -> str:
        return asyncio.run(self._call_async(messages))


class GigaChat(BaseLLM):
    def __init__(
        self,
        token: str,
        model: GIGA_MODELS,
    ):
        if not isinstance(token, str) or len(token) < 16:
            raise ValueError(f"Something is wrong with {token=}")
        self.token = token
        self.model = model
        self.auth_token = ""
        self.call_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        self.tokens_count_url = (
            "https://gigachat.devices.sberbank.ru/api/v1/tokens/count"
        )
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        print("Started_auth")
        self.__auth()
        print("Finished_auth")

    def __auth(self):
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Authorization": f"Basic {self.token}",
            "RqUID": str(uuid.uuid4()),
        }
        scope = "GIGACHAT_API_CORP"
        body = {"scope": scope}
        response = requests.post(
            self.auth_url, headers=headers, data=body, verify=False
        )
        data = response.json()
        self.auth_token = data["access_token"]

    async def __auth_async(self, max_retries: int = 3):
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Authorization": f"Basic {self.token}",
            "RqUID": str(uuid.uuid4()),
        }
        scope = "GIGACHAT_API_CORP"
        body = {"scope": scope}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.auth_url, headers=headers, data=body, ssl=False
            ) as response:
                if response.status == 429:
                    await asyncio.sleep(2)
                    return await self.__auth_async(max_retries - 1)
                data = await response.json()
                self.auth_token = data["access_token"]

    def __call__(self, messages: list[dict[str, str]], max_retries: int = 1):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }

        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "update_interval": 0,
            "temperature": 0,
        }
        response = requests.post(
            self.call_url, headers=headers, json=data, verify=False
        )
        if response.status_code == 401 and max_retries > 0:
            self.__auth()
            return self(messages, max_retries - 1)

        return response.json()["choices"][0]["message"]["content"]

    async def _call_async(self, messages: list[dict[str, str]], max_retries: int = 10):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }

        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "update_interval": 0,
            "temperature": 0,
            "profanity_check": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.call_url, headers=headers, json=data, ssl=False
            ) as response:
                if response.status == 401 and max_retries > 0:
                    await self.__auth_async()
                    return await self._call_async(messages, max_retries - 1)
                if response.status == 429 and max_retries > -3:
                    await asyncio.sleep(2)
                    return await self._call_async(messages, max_retries - 1)
                response_data = await response.json()
                return response_data["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_token_count(response_data: Any) -> list[int]:
        """Normalize API response into list of token counts per input."""
        if isinstance(response_data, list):
            counts = []
            for item in response_data:
                if isinstance(item, dict):
                    value = item.get("tokens")
                    if value is None:
                        value = item.get("count")
                    if value is not None:
                        counts.append(int(value))
            if counts:
                return counts

        if "tokens" in response_data and isinstance(response_data["tokens"], list):
            return [int(v) for v in response_data["tokens"]]

        if "tokens" in response_data and isinstance(response_data["tokens"], int):
            return [int(response_data["tokens"])]

        if "count" in response_data and isinstance(response_data["count"], list):
            return [int(v) for v in response_data["count"]]

        if "count" in response_data and isinstance(response_data["count"], int):
            return [int(response_data["count"])]

        if "data" in response_data and isinstance(response_data["data"], list):
            counts = []
            for item in response_data["data"]:
                if isinstance(item, dict):
                    value = item.get("tokens")
                    if value is None:
                        value = item.get("count")
                    if value is not None:
                        counts.append(int(value))
            if counts:
                return counts

        raise RuntimeError(f"Unexpected tokens/count response format: {response_data}")

    def count_tokens(
        self,
        input_texts: str | list[str],
        model: str = "GigaChat",
        max_retries: int = 1,
    ) -> int | list[int]:
        """Return token count for one text or list of texts."""
        payload_input = [input_texts] if isinstance(input_texts, str) else input_texts
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }
        data = {
            "model": model,
            "input": payload_input,
        }

        response = requests.post(
            self.tokens_count_url, headers=headers, json=data, verify=False
        )
        if response.status_code == 401 and max_retries > 0:
            self.__auth()
            return self.count_tokens(
                input_texts, model=model, max_retries=max_retries - 1
            )

        response.raise_for_status()
        counts = self._extract_token_count(response.json())
        return counts[0] if isinstance(input_texts, str) else counts

    async def count_tokens_async(
        self,
        input_texts: str | list[str],
        model: str = "GigaChat",
        max_retries: int = 3,
    ) -> int | list[int]:
        """Async token count for one text or list of texts."""
        payload_input = [input_texts] if isinstance(input_texts, str) else input_texts
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }
        data = {
            "model": model,
            "input": payload_input,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.tokens_count_url, headers=headers, json=data, ssl=False
            ) as response:
                if response.status == 401 and max_retries > 0:
                    await self.__auth_async()
                    return await self.count_tokens_async(
                        input_texts,
                        model=model,
                        max_retries=max_retries - 1,
                    )

                response_data = await response.json()
                if response.status >= 400:
                    raise RuntimeError(
                        f"GigaChat tokens/count request failed with status {response.status}: {response_data}"
                    )

        counts = self._extract_token_count(response_data)
        return counts[0] if isinstance(input_texts, str) else counts


class OpenAIClient(BaseLLM):
    def __init__(
        self, client: AsyncOpenAI, model: str, temperature: float = 0.0, **kwargs
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.openai_kwargs = kwargs

    async def _call_async(self, messages: list[dict[str, str]]):
        return (
            (
                await self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    **self.openai_kwargs,
                )
            )
            .choices[0]
            .message.content
        )


class GeminiClient(BaseLLM):
    def __init__(
        self, *, api_key: Optional[str] = None, api_keys: Optional[list[str]] = None
    ):
        assert (api_key is None) != (
            api_keys is None
        ), "Either pass single key or list of keys"
        if api_key is not None:
            api_keys = [api_key]
        self.api_keys = deque(api_keys)
        self.model = "gemini-2.5-flash"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def _get_key(self):
        self.api_keys.append(self.api_keys.popleft())
        return self.api_keys[-1]

    @staticmethod
    def _convert_messages(messages: list[dict[str, str]]):
        return [
            {
                "role": "user" if msg["role"] in ["user", "system"] else "model",
                "parts": [{"text": msg["content"]}],
            }
            for msg in messages
        ]

    def __call__(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.base_url}?key={self._get_key()}"
        payload = {"contents": self._convert_messages(messages)}

        response = requests.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
            },
            timeout=60,
        )

        if not response.ok:
            if response.status_code == 403:
                print(f"Invalid key: {self.api_keys.pop()}")
                return self(messages)
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {response.text}"
            )

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return str(data)

    async def _call_async(self, messages: list[dict[str, str]]) -> str:
        pass


class LLM(BaseLLM):
    def __init__(self, client: BaseLLM, max_threads: int = 10, is_test: bool = True):
        self.client = client
        self.max_threads = max_threads
        if is_test:
            response = self.test_client()
            print(response)
            assert "Paris" in response or "Париж" in response

    def test_client(self):
        messages = [
            {"role": "user", "content": "What's the capital of France?"},
        ]
        return self.client(messages=messages)

    @classmethod
    def from_giga_token(cls, token: str, model: GIGA_MODELS, max_threads=10):
        giga = cls(GigaChat(token=token, model=model), max_threads=max_threads)
        giga.token = token
        giga.model = model
        return giga

    @classmethod
    def from_openai_client(
        cls,
        client: AsyncOpenAI,
        model: str,
        temperature: float = 0.0,
        max_threads=1,
        **kwargs,
    ):
        return cls(
            OpenAIClient(client, model=model, temperature=temperature, **kwargs),
            max_threads=max_threads,
        )

    @classmethod
    def from_gemini_token(
        cls, *, api_key: Optional[str] = None, api_keys: Optional[list[str]] = None
    ):
        return cls(GeminiClient(api_key=api_key, api_keys=api_keys))

    @observe
    def __call__(self, messages: list[dict[str, str]]) -> str:
        return self.client(messages=messages)

    @observe
    async def _call_async(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        start_time = time.time()
        answer = await self.client._call_async(messages=messages)
        return {
            "response": answer,
            "duration": time.time() - start_time,
        }

    async def _call_with_semaphore(
        self, messages: list[dict[str, str]], semaphore: asyncio.Semaphore
    ) -> dict[str, Any] | None:
        async with semaphore:
            try:
                return await self._call_async(messages=messages)
            except Exception as e:
                print(f"Request failed: {e}")
                await asyncio.sleep(1)
                return None

    async def call_async(
        self,
        message_list: list[dict[str, str]] | list[list[dict[str, str]]],
    ):
        if not message_list:
            return []

        if isinstance(message_list[0], dict):
            message_list = [message_list]

        max_threads = self.max_threads
        if max_threads < 1:
            max_threads = 1000
        semaphore = asyncio.Semaphore(max_threads)
        tasks = []
        for messages in message_list:
            tasks.append(
                self._call_with_semaphore(messages=messages, semaphore=semaphore)
            )
        res = await async_tqdm.gather(*tasks, total=len(tasks))
        return [None if isinstance(r, Exception) else r for r in res]

    def call_sync(
        self,
        message_list: list[dict[str, str]] | list[list[dict[str, str]]],
    ):
        """
        Synchronous wrapper for call_async method
        """
        return asyncio.run(self.call_async(message_list))
