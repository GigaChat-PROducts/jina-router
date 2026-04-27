import asyncio
import logging
from pathlib import Path

from src.eval_app.models import RunConfig, RunRecord
from src.eval_app.store import upsert_run
from src.evaluation.evaluator import run_evaluation

logger = logging.getLogger(__name__)


def run_named_evaluation(
    *,
    run_name: str,
    config: RunConfig,
    dataset_path: Path,
    runs_path: Path,
) -> tuple[RunRecord, bool]:
    logger.info("evaluation.start run_name=%s dataset=%s", run_name, dataset_path)
    result = asyncio.run(run_evaluation(dataset_path, config.to_eval_config()))
    record = RunRecord.from_result(run_name=run_name, config=config, result=result)
    _, replaced = upsert_run(runs_path, record)
    logger.info(
        "evaluation.finish run_name=%s replaced=%s processed=%s failed=%s",
        run_name,
        replaced,
        record.samples_processed,
        record.samples_failed,
    )
    return record, replaced
