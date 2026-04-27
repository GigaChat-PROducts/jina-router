from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.evaluation.evaluator import EvaluationConfig, EvaluationResult
from src.evaluation.quotas import QuotaTask


class RunConfig(BaseModel):
    reranker_endpoint: str
    global_token_limit: int
    enable_quota_reranking: bool
    task_reranking_temperature: float
    product_reranking_temperature: float
    base_product_multiplier: float
    current_product_multiplier: float
    future_product_multiplier: float
    quota_tasks: list[QuotaTask]

    @classmethod
    def from_eval_config(cls, config: EvaluationConfig) -> "RunConfig":
        return cls(
            reranker_endpoint=config.reranker_endpoint,
            global_token_limit=config.global_token_limit,
            enable_quota_reranking=config.enable_quota_reranking,
            task_reranking_temperature=config.task_reranking_temperature,
            product_reranking_temperature=config.product_reranking_temperature,
            base_product_multiplier=config.base_product_multiplier,
            current_product_multiplier=config.current_product_multiplier,
            future_product_multiplier=config.future_product_multiplier,
            quota_tasks=config.quota_tasks,
        )

    def to_eval_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            reranker_endpoint=self.reranker_endpoint,
            global_token_limit=self.global_token_limit,
            enable_quota_reranking=self.enable_quota_reranking,
            task_reranking_temperature=self.task_reranking_temperature,
            product_reranking_temperature=self.product_reranking_temperature,
            base_product_multiplier=self.base_product_multiplier,
            current_product_multiplier=self.current_product_multiplier,
            future_product_multiplier=self.future_product_multiplier,
            quota_tasks=self.quota_tasks,
        )


class FailureRecord(BaseModel):
    item_id: str
    error: str


class RunRecord(BaseModel):
    run_name: str
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config: RunConfig
    samples_total: int
    samples_processed: int
    samples_failed: int
    task_kl_div: float
    product_kl_div: float
    task_mse: float
    product_mse: float
    failures: list[FailureRecord] = Field(default_factory=list)

    @classmethod
    def from_result(
        cls,
        *,
        run_name: str,
        config: RunConfig,
        result: EvaluationResult,
    ) -> "RunRecord":
        return cls(
            run_name=run_name,
            config=config,
            samples_total=result.samples_total,
            samples_processed=result.samples_processed,
            samples_failed=result.samples_failed,
            task_kl_div=result.task_kl_div,
            product_kl_div=result.product_kl_div,
            task_mse=result.task_mse,
            product_mse=result.product_mse,
            failures=[
                FailureRecord(item_id=failure.item_id, error=failure.error)
                for failure in result.failures
            ],
        )
