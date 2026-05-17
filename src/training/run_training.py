import argparse
import logging

import torch
from transformers import Trainer

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
        data_collator=train_ds.collate_fn,
    )

    trainer.train()


if __name__ == "__main__":
    main()
