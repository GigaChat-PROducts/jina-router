import asyncio
import logging
import math
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from src.clients.reranker_client import BaseRerankerClient
from src.dataset.product_mapping import DATA_DESCRIPTIONS, ID_TO_PRODUCT

logger = logging.getLogger(__name__)

# Future-product weighting is intentionally disabled for now.
_ENABLE_FUTURE_PRODUCT_MULTIPLIER = False


class QuotaTask(BaseModel):
    name: str
    multiplier: float = 1.0


class QuotaPlanner(ABC):
    @abstractmethod
    async def plan_quotas(
        self,
        *,
        query: str,
        products: list[str],
        base_product: str | None = None,
        current_product: str | None = None,
        future_product: str | None = None,
    ) -> dict[str, dict[str, int]]:
        raise NotImplementedError


def _softmax(scores: list[float], temperature: float) -> np.ndarray:
    if not scores:
        return np.array([], dtype=float)

    safe_temperature = max(temperature, 1e-6)
    logits = np.asarray(scores, dtype=float) / safe_temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    total = np.sum(weights)
    if total <= 0:
        return np.full(len(scores), 1.0 / len(scores), dtype=float)
    return weights / total


def _quota_from_scores(
    scores: list[float], quota: int, temperature: float
) -> list[int]:
    if quota <= 0 or not scores:
        return [0] * len(scores)

    probabilities = _softmax(scores, temperature)
    return [int(value) for value in (probabilities * quota).tolist()]


class StaticQuotaPlanner(QuotaPlanner):
    def __init__(
        self,
        *,
        global_token_limit: int,
        tasks: list[QuotaTask],
        task_temperature: float = 1.0,
    ) -> None:
        self._global_token_limit = global_token_limit
        self._tasks = list(tasks)
        self._task_temperature = task_temperature

    async def plan_quotas(
        self,
        *,
        query: str,
        products: list[str],
        base_product: str | None = None,
        current_product: str | None = None,
        future_product: str | None = None,
    ) -> dict[str, dict[str, int]]:
        _ = (query, base_product, current_product, future_product)
        if self._global_token_limit <= 0:
            return {task.name: {} for task in self._tasks}

        task_scores = [task.multiplier for task in self._tasks]
        task_quotas = _quota_from_scores(
            task_scores,
            self._global_token_limit,
            self._task_temperature,
        )

        if not products:
            return {task.name: {} for task in self._tasks}

        product_quotas_by_task: dict[str, dict[str, int]] = {}
        for task, task_quota in zip(self._tasks, task_quotas, strict=False):
            base, remainder = divmod(task_quota, len(products))
            product_quotas_by_task[task.name] = {
                product: base + (1 if index < remainder else 0)
                for index, product in enumerate(products)
            }

        return product_quotas_by_task


class RerankQuotaPlanner(QuotaPlanner):
    def __init__(
        self,
        *,
        client: BaseRerankerClient,
        global_token_limit: int,
        tasks: list[QuotaTask],
        task_temperature: float = 1.0,
        product_temperature: float = 1.0,
        base_product_multiplier: float = 1.0,
        current_product_multiplier: float = 1.0,
        future_product_multiplier: float = 1.0,
        fallback: QuotaPlanner | None = None,
    ) -> None:
        self._client = client
        self._global_token_limit = global_token_limit
        self._tasks = list(tasks)
        self._task_temperature = task_temperature
        self._product_temperature = product_temperature
        self._base_product_multiplier = base_product_multiplier
        self._current_product_multiplier = current_product_multiplier
        self._future_product_multiplier = future_product_multiplier
        self._fallback = fallback

    @staticmethod
    def _resolve_data_subset(requested: list[str]) -> list[dict[str, str]]:
        name_to_item = {item["name"]: item for item in DATA_DESCRIPTIONS}
        return [name_to_item[name] for name in requested if name in name_to_item]

    @staticmethod
    def _resolve_product_subset(requested: list[str]) -> list[dict[str, str]]:
        result = []
        for product_id in requested:
            product = ID_TO_PRODUCT.get(product_id)
            if product is None:
                continue
            result.append(
                {
                    "id": product_id,
                    "description": product["description"],
                }
            )
        return result

    async def plan_quotas(
        self,
        *,
        query: str,
        products: list[str],
        base_product: str | None = None,
        current_product: str | None = None,
        future_product: str | None = None,
        base_product_multiplier: float | None = None,
        current_product_multiplier: float | None = None,
        future_product_multiplier: float | None = None,
    ) -> dict[str, dict[str, int]]:
        effective_base_multiplier = (
            self._base_product_multiplier
            if base_product_multiplier is None
            else base_product_multiplier
        )
        effective_current_multiplier = (
            self._current_product_multiplier
            if current_product_multiplier is None
            else current_product_multiplier
        )
        effective_future_multiplier = (
            self._future_product_multiplier
            if future_product_multiplier is None
            else future_product_multiplier
        )
        if self._global_token_limit <= 0:
            return {task.name: {} for task in self._tasks}

        task_names = [task.name for task in self._tasks]
        task_multipliers_by_name = {task.name: task.multiplier for task in self._tasks}
        task_items = self._resolve_data_subset(task_names)
        product_items = self._resolve_product_subset(products)
        product_score_multipliers = self._resolve_product_score_multipliers(
            product_names=[item["id"] for item in product_items],
            base_product=base_product,
            current_product=current_product,
            future_product=future_product,
            base_product_multiplier=effective_base_multiplier,
            current_product_multiplier=effective_current_multiplier,
            future_product_multiplier=effective_future_multiplier,
        )

        async def _rank_task_descriptions() -> list[float]:
            if not task_items:
                return []
            return await self._client.get_relevance_scores_async(
                query=query,
                documents=[item["description"] for item in task_items],
            )

        async def _rank_product_descriptions() -> list[float]:
            if not product_items:
                return []
            return await self._client.get_relevance_scores_async(
                query=query,
                documents=[item["description"] for item in product_items],
            )

        try:
            task_scores, product_scores = await asyncio.gather(
                _rank_task_descriptions(),
                _rank_product_descriptions(),
            )
        except Exception as error:
            if self._fallback is None:
                raise
            logger.warning(
                "Failed to compute dynamic RAG quotas, using static token limits: %s",
                error,
            )
            return await self._fallback.plan_quotas(
                query=query,
                products=products,
                base_product=base_product,
                current_product=current_product,
                future_product=future_product,
            )

        weighted_task_scores = [
            self._multiply_score(score, task_multipliers_by_name[item["name"]])
            for item, score in zip(task_items, task_scores, strict=False)
        ]
        task_quotas = {
            item["name"]: quota
            for item, quota in zip(
                task_items,
                _quota_from_scores(
                    weighted_task_scores,
                    self._global_token_limit,
                    self._task_temperature,
                ),
                strict=False,
            )
        }

        weighted_product_scores = [
            self._multiply_score(score, product_score_multipliers[item["id"]])
            for item, score in zip(product_items, product_scores, strict=True)
        ]

        product_quotas_by_task: dict[str, dict[str, int]] = {}
        for task_name, task_quota in task_quotas.items():
            product_quotas_by_task[task_name] = {
                item["id"]: quota
                for item, quota in zip(
                    product_items,
                    _quota_from_scores(
                        weighted_product_scores,
                        task_quota,
                        self._product_temperature,
                    ),
                    strict=False,
                )
            }

        return product_quotas_by_task

    @staticmethod
    def _resolve_product_score_multipliers(
        *,
        product_names: list[str],
        base_product: str | None,
        current_product: str | None,
        future_product: str | None,
        base_product_multiplier: float,
        current_product_multiplier: float,
        future_product_multiplier: float,
    ) -> dict[str, float]:
        role_multiplier_by_product = {name: 1.0 for name in product_names}
        product_name_set = set(product_names)

        def _apply_multiplier(product_name: str | None, multiplier: float) -> None:
            if product_name is None or product_name not in product_name_set:
                return
            role_multiplier_by_product[product_name] *= max(multiplier, 1e-6)

        _apply_multiplier(base_product, base_product_multiplier)
        _apply_multiplier(current_product, current_product_multiplier)
        if _ENABLE_FUTURE_PRODUCT_MULTIPLIER:
            _apply_multiplier(future_product, future_product_multiplier)
        return role_multiplier_by_product

    @staticmethod
    def _multiply_score(score: float, multiplier: float) -> float:
        return score + math.log(max(multiplier, 1e-6))


__all__ = [
    "QuotaPlanner",
    "QuotaTask",
    "RerankQuotaPlanner",
    "StaticQuotaPlanner",
]
