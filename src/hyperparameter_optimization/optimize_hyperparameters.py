import asyncio
from pathlib import Path

import optuna

from src.config import settings
from src.evaluation.evaluator import EvaluationConfig, EvaluationResult, run_evaluation
from src.evaluation.quotas import QuotaTask
from src.hyperparameter_optimization.config import OptimizerSettings, optimizer_settings


def calculate_metric(result: EvaluationResult):
    task_kl_div = result.task_kl_div
    product_kl_div = result.product_kl_div
    loss = task_kl_div + 2 * product_kl_div
    return loss


def get_data_path():
    return Path("src/dataset/data/labeled_dataset.json")


def get_eval_config(trial, optimizer_settings: OptimizerSettings):
    document_multiplier = trial.suggest_float(
        *optimizer_settings.document_multiplier.convert()
    )
    quota_tasks = [
        QuotaTask(name="documents", multiplier=document_multiplier),
        QuotaTask(name="best_practices", multiplier=1.0),
    ]
    return EvaluationConfig(
        reranker_endpoint=settings.reranker_endpoint,
        global_token_limit=settings.global_token_limit,
        enable_quota_reranking=settings.enable_quota_reranking,
        task_reranking_temperature=trial.suggest_float(
            *optimizer_settings.task_temperature.convert()
        ),
        product_reranking_temperature=trial.suggest_float(
            *optimizer_settings.product_temperature.convert()
        ),
        base_product_multiplier=trial.suggest_float(
            *optimizer_settings.base_product_multiplier.convert()
        ),
        current_product_multiplier=trial.suggest_float(
            *optimizer_settings.current_product_multiplier.convert()
        ),
        future_product_multiplier=trial.suggest_float(
            *optimizer_settings.future_product_multiplier.convert()
        ),
        quota_tasks=quota_tasks,
    )


def objective(trial):
    eval_config = get_eval_config(trial, optimizer_settings)
    data_path = get_data_path()
    result = asyncio.run(run_evaluation(dataset_path=data_path, config=eval_config))

    trial.set_user_attr("task_kl_div", result.task_kl_div)
    trial.set_user_attr("product_kl_div", result.product_kl_div)
    trial.set_user_attr("samples_processed", result.samples_processed)
    return calculate_metric(result)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name=optimizer_settings.run_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=optimizer_settings.n_trials)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
