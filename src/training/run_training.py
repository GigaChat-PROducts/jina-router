import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
from peft import LoraConfig, TaskType, get_peft_model
from transformers.trainer import Trainer

from src.dataset.schemas import ItemClass
from src.training.config import TrainingConfig
from src.training.dataset import TrainingDataset
from src.training.model_constants import MODEL_NAME
from src.training.modeling import JinaForRanking
from src.training.tokenizer import ModelTokenizer, ModelTokenizerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model(base_name: str = MODEL_NAME):
    logger.info(f"Loading base model {base_name}")
    return JinaForRanking.from_pretrained(base_name)


def build_lora_config(cfg: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        target_modules=cfg.lora_target_modules,
        use_rslora=cfg.lora_use_rslora,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export a publishable model bundle with ONNX after training completes.",
    )
    p.add_argument(
        "--final-dir-name",
        default=None,
        help="Name of the final export folder under output_dir. Defaults to config value.",
    )
    args = p.parse_args()

    cfg = TrainingConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.final_dir_name:
        cfg.final_dir_name = args.final_dir_name
    if args.export_onnx:
        cfg.export_onnx = True

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    tokenizer_config = ModelTokenizerConfig()

    tokenizer = ModelTokenizer(tokenizer_config)
    train_ds = TrainingDataset(mode=ItemClass.TRAIN, tokenizer=tokenizer)
    eval_ds = TrainingDataset(mode=ItemClass.VAL, tokenizer=tokenizer)

    model: Any = get_peft_model(load_model(), build_lora_config(cfg))
    base_model = model.get_base_model()
    for parameter in base_model.projector.parameters():
        parameter.requires_grad = True

    training_args = cfg.get_training_arguments()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=train_ds.collate_fn,
    )

    trainer.train()

    model = model.merge_and_unload()
    final_dir = Path(cfg.output_dir) / cfg.final_dir_name
    logger.info("Saving final model bundle to %s", final_dir)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if cfg.export_onnx:
        from src.training.onnx_export import export_model_bundle

        export_model_bundle(
            source_model_dir=final_dir,
            output_dir=final_dir,
            tokenizer_source=final_dir,
            opset=cfg.onnx_opset,
        )


if __name__ == "__main__":
    main()
