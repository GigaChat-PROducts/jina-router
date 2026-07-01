import asyncio
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from tqdm.asyncio import tqdm as async_tqdm

from src.clients.reranker_client import JinaRerankerClient
from src.config import rag_settings
from src.dataset.product_mapping import ID_TO_PRODUCT
from src.dataset.schemas import DatasetItem, Distribution, ItemClass
from src.evaluation.quotas import QuotaTask, RerankQuotaPlanner, StaticQuotaPlanner

Metric = Literal["kl_div", "mse"]


@dataclass
class EvaluationConfig:
    reranker_endpoint: str
    global_token_limit: int
    enable_quota_reranking: bool
    task_reranking_temperature: float
    product_reranking_temperature: float
    base_product_multiplier: float
    current_product_multiplier: float
    future_product_multiplier: float
    quota_tasks: list[QuotaTask]


@dataclass
class PredictionFailure:
    item_id: str
    error: str


@dataclass
class DetailedSampleResult:
    item_id: str
    query: str
    predicted_task: list[dict[str, float]]
    gt_task: list[dict[str, float]]
    task_kl_div: float
    predicted_product: list[dict[str, float]]
    gt_product: list[dict[str, float]]
    product_kl_div: float


@dataclass
class EvaluationResult:
    samples_total: int
    samples_processed: int
    samples_failed: int
    task_kl_div: float
    task_mse: float
    product_kl_div: float
    product_mse: float
    failures: list[PredictionFailure]
    detailed_samples: list[DetailedSampleResult]

    def to_dict(self) -> dict[str, any]:
        return {
            "samples_total": self.samples_total,
            "samples_processed": self.samples_processed,
            "samples_failed": self.samples_failed,
            "task_kl_div": self.task_kl_div,
            "task_mse": self.task_mse,
            "product_kl_div": self.product_kl_div,
            "product_mse": self.product_mse,
            "failures": [asdict(failure) for failure in self.failures],
            "detailed_samples": [asdict(sample) for sample in self.detailed_samples],
        }


def get_scores(
    target: list[Distribution] | None,
    predicted: list[Distribution] | None,
    metric: Metric,
) -> float:
    target_map = {item.name: item.probability for item in target or []}
    predicted_map = {item.name: item.probability for item in predicted or []}
    if len(target_map) != len(predicted_map):
        raise ValueError()
    res = 0

    if metric == "mse":
        res = sum(
            (target_map[key] - predicted_map[key]) ** 2 for key in target_map
        ) / len(target_map)

    elif metric == "kl_div":
        eps = 1e-12
        res = sum(
            target_map[key]
            * math.log((target_map[key] + eps) / (predicted_map[key] + eps))
            for key in target_map
        ) / len(target_map)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return res


def _to_distributions(
    values: dict[str, int], *, map_name: bool = False
) -> list[Distribution]:
    total = sum(values.values())
    if total <= 0:
        return []

    result = []
    for name, quota in values.items():
        if quota <= 0:
            raise ValueError
        result.append(Distribution(name=name, probability=quota / total))
    return result


def _aggregate_predictions(
    quotas: dict[str, dict[str, int]],
) -> tuple[list[Distribution], list[Distribution]]:
    per_task = {task: sum(product_map.values()) for task, product_map in quotas.items()}
    task_distribution = _to_distributions(
        {task: value for task, value in per_task.items()}
    )

    per_product: dict[str, int] = {}
    for product_map in quotas.values():
        for product_id, quota in product_map.items():
            per_product[product_id] = per_product.get(product_id, 0) + quota
    product_distribution = _to_distributions(per_product, map_name=True)
    return task_distribution, product_distribution


def build_default_config() -> EvaluationConfig:
    token_limit = (
        rag_settings.global_token_limit
        if rag_settings.global_token_limit is not None
        else 10000
    )
    return EvaluationConfig(
        reranker_endpoint=rag_settings.reranker_endpoint,
        global_token_limit=token_limit,
        enable_quota_reranking=rag_settings.enable_quota_reranking,
        task_reranking_temperature=rag_settings.task_reranking_temperature,
        product_reranking_temperature=rag_settings.product_reranking_temperature,
        base_product_multiplier=rag_settings.base_product_multiplier,
        current_product_multiplier=rag_settings.current_product_multiplier,
        future_product_multiplier=rag_settings.future_product_multiplier,
        quota_tasks=list(rag_settings.quota_tasks),
    )


def _build_planner(config: EvaluationConfig) -> RerankQuotaPlanner:
    token_limit = config.global_token_limit
    static_planner = StaticQuotaPlanner(
        global_token_limit=token_limit,
        tasks=config.quota_tasks,
        task_temperature=config.task_reranking_temperature,
    )
    if not config.enable_quota_reranking:
        return static_planner

    return RerankQuotaPlanner(
        client=JinaRerankerClient(config.reranker_endpoint),
        global_token_limit=token_limit,
        tasks=config.quota_tasks,
        task_temperature=config.task_reranking_temperature,
        product_temperature=config.product_reranking_temperature,
        base_product_multiplier=config.base_product_multiplier,
        current_product_multiplier=config.current_product_multiplier,
        future_product_multiplier=config.future_product_multiplier,
        fallback=None,
    )


async def _predict_item(
    item: DatasetItem,
    planner: StaticQuotaPlanner | RerankQuotaPlanner,
    semaphore: asyncio.Semaphore,  # Added semaphore parameter
) -> tuple[str, list[Distribution], list[Distribution], str | None]:
    try:
        # Wrap the API/planning execution with the semaphore context manager
        async with semaphore:
            quotas = await planner.plan_quotas(
                query="\n".join(item.dialog),
                products=[item.base_product, *item.products],
                base_product=item.base_product,
                current_product=item.current_product,
            )
        task_distribution, product_distribution = _aggregate_predictions(quotas)
        return item.item_id, task_distribution, product_distribution, None
    except Exception as error:
        return item.item_id, [], [], str(error)


def _aggregate_metric_sums(
    dataset: list[DatasetItem],
    predictions: list[tuple[str, list[Distribution], list[Distribution], str | None]],
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[PredictionFailure],
    list[DetailedSampleResult],
]:
    task_kl = []
    task_mse = []
    product_kl = []
    product_mse = []
    failures: list[PredictionFailure] = []
    detailed_samples: list[DetailedSampleResult] = []

    for item, (item_id, pred_task, pred_product, error) in zip(
        dataset, predictions, strict=False
    ):
        if error is not None:
            failures.append(PredictionFailure(item_id=item_id, error=error))
            continue

        item_task_kl = get_scores(item.gt_task_distribution, pred_task, "kl_div")
        item_task_mse = get_scores(item.gt_task_distribution, pred_task, "mse")
        item_product_kl = get_scores(
            item.gt_product_distribution_with_context, pred_product, "kl_div"
        )
        item_product_mse = get_scores(
            item.gt_product_distribution_with_context, pred_product, "mse"
        )

        task_kl.append(item_task_kl)
        task_mse.append(item_task_mse)
        product_kl.append(item_product_kl)
        product_mse.append(item_product_mse)

        detailed_samples.append(
            DetailedSampleResult(
                item_id=item_id,
                query="\n".join(item.dialog),
                predicted_task=[
                    {"name": d.name, "probability": d.probability} for d in pred_task
                ],
                predicted_product=[
                    {"name": d.name, "probability": d.probability} for d in pred_product
                ],
                gt_task=[
                    {"name": d.name, "probability": d.probability}
                    for d in (item.gt_task_distribution or [])
                ],
                gt_product=[
                    {"name": d.name, "probability": d.probability}
                    for d in (item.gt_product_distribution_with_context or [])
                ],
                task_kl_div=item_task_kl,
                product_kl_div=item_product_kl,
            )
        )

    return task_kl, task_mse, product_kl, product_mse, failures, detailed_samples


def get_dataset(path: Path, mode: ItemClass) -> list[DatasetItem]:
    dataset: list[DatasetItem] = [
        DatasetItem.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
        if item.get("gt_task_distribution") and item.get("gt_product_distribution")
    ]
    dataset = [item for item in dataset if item.item_class == mode]
    return dataset


async def run_evaluation(
    dataset_path: Path,
    mode: ItemClass,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    effective_config = build_default_config() if config is None else config
    dataset = get_dataset(dataset_path, mode)
    planner = _build_planner(effective_config)

    # Initialize the semaphore with a max concurrency limit of 32
    semaphore = asyncio.Semaphore(16)

    # Pass the semaphore down into each task
    tasks = [_predict_item(item, planner, semaphore) for item in dataset]
    predictions = await async_tqdm.gather(*tasks, total=len(tasks))

    (
        task_kl,
        task_mse,
        product_kl,
        product_mse,
        failures,
        detailed_samples,
    ) = _aggregate_metric_sums(
        dataset,
        predictions,
    )

    processed = len(task_kl)
    denominator = processed or 1
    return EvaluationResult(
        samples_total=len(dataset),
        samples_processed=processed,
        samples_failed=len(failures),
        task_kl_div=sum(task_kl) / denominator,
        task_mse=sum(task_mse) / denominator,
        product_kl_div=sum(product_kl) / denominator,
        product_mse=sum(product_mse) / denominator,
        failures=failures,
        detailed_samples=detailed_samples,
    )


async def main() -> None:
    path = Path("src/dataset/data/labeled_dataset.json")
    results = await run_evaluation(path, ItemClass.TEST, build_default_config())
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, ensure_ascii=False, indent=2)
    # print(json.dumps(results.to_dict(), ensure_ascii=False, indent=2))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(__file__).parent / "evaluation_results" / f"evaluation_{timestamp}.json"
    )
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # from dotenv import load_dotenv

    # load_dotenv(".env")
    asyncio.run(main())
