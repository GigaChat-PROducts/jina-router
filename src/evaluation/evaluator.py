import asyncio
import json
import math
from pathlib import Path
from typing import Literal

from tqdm.asyncio import tqdm as async_tqdm

from src.clients.reranker_client import JinaRerankerClient
from src.config import rag_settings
from src.dataset.product_mapping import ID_TO_PRODUCT
from src.dataset.schemas import DatasetItem, Distribution
from src.evaluation.quotas import RerankQuotaPlanner, StaticQuotaPlanner

Metric = Literal["kl_div", "mse"]
TASK_TO_LABEL = {
    "documents": "factology",
    "best_practices": "sales_practices",
}


def get_scores(
    target: list[Distribution] | None,
    predicted: list[Distribution] | None,
    metric: Metric,
) -> float:
    target_map = {item.name: item.probability for item in target or []}
    predicted_map = {item.name: item.probability for item in predicted or []}
    keys = sorted(set(target_map) | set(predicted_map))
    if not keys:
        return 0.0

    target_values = [target_map.get(key, 0.0) for key in keys]
    predicted_values = [predicted_map.get(key, 0.0) for key in keys]

    target_sum = sum(target_values) or 1.0
    predicted_sum = sum(predicted_values) or 1.0
    target_values = [value / target_sum for value in target_values]
    predicted_values = [value / predicted_sum for value in predicted_values]

    if metric == "mse":
        return sum(
            (target_value - predicted_value) ** 2
            for target_value, predicted_value in zip(
                target_values,
                predicted_values,
                strict=False,
            )
        ) / len(keys)
    if metric == "kl_div":
        eps = 1e-12
        return sum(
            target_value * math.log((target_value + eps) / (predicted_value + eps))
            for target_value, predicted_value in zip(
                target_values,
                predicted_values,
                strict=False,
            )
        )
    raise ValueError(f"Unsupported metric: {metric}")


def _to_distributions(
    values: dict[str, int], *, map_name: bool = False
) -> list[Distribution]:
    total = sum(values.values())
    if total <= 0:
        return []

    result = []
    for name, quota in values.items():
        if quota <= 0:
            continue
        final_name = (
            ID_TO_PRODUCT[name]["name"] if map_name and name in ID_TO_PRODUCT else name
        )
        result.append(Distribution(name=final_name, probability=quota / total))
    return result


def _aggregate_predictions(
    quotas: dict[str, dict[str, int]],
) -> tuple[list[Distribution], list[Distribution]]:
    per_task = {task: sum(product_map.values()) for task, product_map in quotas.items()}
    task_distribution = _to_distributions(
        {TASK_TO_LABEL.get(task, task): value for task, value in per_task.items()}
    )

    per_product: dict[str, int] = {}
    for product_map in quotas.values():
        for product_id, quota in product_map.items():
            per_product[product_id] = per_product.get(product_id, 0) + quota
    product_distribution = _to_distributions(per_product, map_name=True)
    return task_distribution, product_distribution


def _build_planner() -> StaticQuotaPlanner | RerankQuotaPlanner:
    token_limit = (
        rag_settings.global_token_limit
        if rag_settings.global_token_limit is not None
        else 10000
    )
    static_planner = StaticQuotaPlanner(
        global_token_limit=token_limit,
        tasks=rag_settings.quota_tasks,
        task_temperature=rag_settings.task_reranking_temperature,
    )
    if not rag_settings.enable_quota_reranking:
        return static_planner

    return RerankQuotaPlanner(
        client=JinaRerankerClient(rag_settings.reranker_endpoint),
        global_token_limit=token_limit,
        tasks=rag_settings.quota_tasks,
        task_temperature=rag_settings.task_reranking_temperature,
        product_temperature=rag_settings.product_reranking_temperature,
        base_product_multiplier=rag_settings.base_product_multiplier,
        current_product_multiplier=rag_settings.current_product_multiplier,
        future_product_multiplier=rag_settings.future_product_multiplier,
        fallback=static_planner,
    )


async def _predict_item(
    item: DatasetItem,
    planner: StaticQuotaPlanner | RerankQuotaPlanner,
) -> tuple[list[Distribution], list[Distribution]]:
    quotas = await planner.plan_quotas(
        query="\n".join(item.dialog),
        products=[item.base_product, *item.products],
        base_product=item.base_product,
        current_product=item.current_product,
    )
    return _aggregate_predictions(quotas)


async def run_evaluation(dataset_path: Path) -> dict[str, float | int]:
    dataset = [
        DatasetItem.model_validate(item)
        for item in json.loads(dataset_path.read_text(encoding="utf-8"))
        if item.get("gt_task_distribution") and item.get("gt_product_distribution")
    ]
    planner = _build_planner()
    tasks = [_predict_item(item, planner) for item in dataset]
    predictions = await async_tqdm.gather(*tasks, total=len(tasks))

    task_kl = []
    task_mse = []
    product_kl = []
    product_mse = []
    for item, (pred_task, pred_product) in zip(dataset, predictions, strict=False):
        task_kl.append(get_scores(item.gt_task_distribution, pred_task, "kl_div"))
        task_mse.append(get_scores(item.gt_task_distribution, pred_task, "mse"))
        product_kl.append(
            get_scores(item.gt_product_distribution, pred_product, "kl_div")
        )
        product_mse.append(
            get_scores(item.gt_product_distribution, pred_product, "mse")
        )

    size = len(dataset) or 1
    return {
        "samples": len(dataset),
        "task_kl_div": sum(task_kl) / size,
        "task_mse": sum(task_mse) / size,
        "product_kl_div": sum(product_kl) / size,
        "product_mse": sum(product_mse) / size,
    }


async def main() -> None:
    path = Path("src/dataset/data/labeled_dataset.json")
    results = await run_evaluation(path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
