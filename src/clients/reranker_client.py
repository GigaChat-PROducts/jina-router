import asyncio
from abc import ABC, abstractmethod

import aiohttp
from langfuse import observe


class BaseRerankerClient(ABC):
    """Контракт transport-клиента для внешнего reranker backend-а.

    Сейчас такой клиент используется в слоях, где нужно пересчитать
    относительную релевантность набора документов для одного query. Сам
    контракт не привязан к конкретному orchestration-сценарию и не знает ничего
    о retrieval phases, quota allocation или product-specific логике.
    """

    def get_relevance_scores(self, query: str, documents: list[str]) -> list[float]:
        """Синхронная обёртка над async API для простых локальных вызовов."""

        return asyncio.run(self.get_relevance_scores_async(query, documents))

    @abstractmethod
    async def get_relevance_scores_async(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """Вернуть relevance score для каждого документа в исходном порядке."""

        raise NotImplementedError


jina_cache = {}


class JinaRerankerClient(BaseRerankerClient):
    """HTTP-клиент к reranker endpoint-у в формате Jina-compatible `/predict`."""

    def __init__(self, endpoint: str) -> None:
        """Нормализовать базовый URL и сохранить конечный predict endpoint."""

        self._url = endpoint.rstrip("/") + "/predict"
        self.cache = jina_cache
        self.semaphore = asyncio.Semaphore(256)

    @staticmethod
    def _extract_scores(data: object) -> list[float]:
        """Нормализовать JSON-ответ backend-а к плоскому списку score-ов."""

        if isinstance(data, dict):
            scores = data.get("scores")
            if isinstance(scores, list):
                return [float(score) for score in scores]

            legacy_scores = data.get("data")
            if (
                isinstance(legacy_scores, list)
                and legacy_scores
                and isinstance(legacy_scores[0], list)
            ):
                return [float(score) for score in legacy_scores[0]]

        raise RuntimeError(f"Unexpected reranker response format: {data}")

    @observe(name="reranker-client.get-relevance-scores")
    async def get_relevance_scores_async(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """Вызвать reranker backend и вернуть score-ы для переданных документов."""
        if (query, tuple(documents)) in self.cache:
            return self.cache[(query, tuple(documents))]

        payload = [{"query": query, "documents": documents}]
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1000)
        ) as session:
            async with self.semaphore:
                async with session.post(self._url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
        res = self._extract_scores(data)
        self.cache[(query, tuple(documents))] = res
        return res


__all__ = [
    "BaseRerankerClient",
    "JinaRerankerClient",
]
