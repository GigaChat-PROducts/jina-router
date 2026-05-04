import argparse
import logging
import os

import mlflow
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer

from src.dataset.schemas import ItemClass
from src.training.config import TrainingConfig
from src.training.dataset import TrainingDataset
from src.training.modeling import JinaForRanking
from src.training.tokenizer import ModelTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_collate_fn(tokenizer: ModelTokenizer):
    pad_token_id = tokenizer.tokenizer.pad_token_id or 0

    def collate_fn(batch):
        input_ids = [item.inputs.input_ids.squeeze(0) for item in batch]
        attention_mask = [item.inputs.attention_mask.squeeze(0) for item in batch]
        labels = [item.labels for item in batch]

        return {
            "input_ids": pad_sequence(
                input_ids, batch_first=True, padding_value=pad_token_id
            ),
            "attention_mask": pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100.0),
        }

    return collate_fn


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

    tokenizer = ModelTokenizer()
    train_ds = TrainingDataset(mode=ItemClass.TRAIN, tokenizer=tokenizer)
    eval_ds = TrainingDataset(mode=ItemClass.VAL, tokenizer=tokenizer)

    model = load_model()

    training_args = cfg.get_training_arguments()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=build_collate_fn(tokenizer),
    )

    with mlflow.start_run(run_name="jina-reranker"):
        mlflow.log_params(cfg.model_dump())
        if args.mode == "train":
            trainer.train()
        else:
            print(trainer.evaluate())


if __name__ == "__main__":
    main()
