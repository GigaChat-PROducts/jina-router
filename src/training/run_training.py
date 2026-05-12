import argparse
import logging
import os

import mlflow
import torch
from transformers import Trainer

from src.dataset.schemas import ItemClass
from src.training.config import TrainingConfig
from src.training.dataset import TrainingDataset
from src.training.modeling import JinaForRanking
from src.training.tokenizer import ModelTokenizer, ModelTokenizerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def collate_fn(batch):
    return {
        "input_ids": torch.Tensor([item.inputs.input_ids.squeeze(0) for item in batch]),
        "attention_mask": torch.Tensor(
            [item.inputs.attention_mask.squeeze(0) for item in batch]
        ),
        "labels": torch.Tensor([item.labels for item in batch]),
    }


def setup_mlflow(cfg: TrainingConfig):
    os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow_tracking_uri
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)


def load_model(base_name: str = "jinaai/jina-reranker-v3"):
    logger.info(f"Loading base model {base_name}")
    model = JinaForRanking.from_pretrained(base_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    cfg = TrainingConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir

    setup_mlflow(cfg)

    tokenizer_config = ModelTokenizerConfig()

    tokenizer = ModelTokenizer(tokenizer_config)
    train_ds = TrainingDataset(mode=ItemClass.TRAIN, tokenizer=tokenizer)
    eval_ds = TrainingDataset(mode=ItemClass.VAL, tokenizer=tokenizer)

    model = load_model()

    training_args = cfg.get_training_arguments()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    with mlflow.start_run(run_name="jina-reranker"):
        mlflow.log_params(cfg.model_dump())
        trainer.train()


if __name__ == "__main__":
    main()
