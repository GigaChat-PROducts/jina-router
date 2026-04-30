import os
import json
from pathlib import Path
from dataclasses import dataclass

import torch
from pydantic import BaseModel

from src.training.utils import format_docs_prompts_func
from src.constants.cross_encoder_descriptions import product_name_to_id, ID_TO_DESCRIPTION
from src.dataset import DatasetItem, ItemClass
from src.training.tokenizer import ModelTokenizer, TokenizerOutput

DATASET_PATH = Path(__file__).parent.parent / "dataset" / "data" / "labeled_dataset.json"


class TrainingDatasetItem(BaseModel):
    query: str
    document_keys: list[str]
    labels: list[float]

    def __post_init__(self):
        if len(self.document_keys) != len(self.labels):
            raise ValueError


@dataclass
class EncodedDatasetItem:
    inputs: TokenizerOutput
    labels: torch.Tensor


class TrainingDataset:
    def __init__(self, mode: ItemClass, tokenizer: ModelTokenizer):
        if not os.path.exists(DATASET_PATH):
            raise ValueError(f"Path {DATASET_PATH} does not exist, ensure you downloaded it")
        
        with open(DATASET_PATH) as f:
            data = json.load(f)
        
        self.dataset = []

        def sort_labels(item, keys):
            labels = []
            for key in keys:
                for distribution in item.gt_task_distribution:
                    if product_name_to_id(distribution.name) == key:
                        labels.append(distribution.probability)
            return labels

        for d in data:
            item = DatasetItem(**d)
            if item.item_class != mode:
                continue
            dialog = "\n".join(item.dialog)
            self.dataset.append(TrainingDatasetItem(
                query=dialog,
                document_keys=["documents", "best_practices"],
                labels=sort_labels(item, ["factology", "sales_practices"]),
            ))

            document_keys = [item.base_product] + item.products
            self.dataset.append(TrainingDatasetItem(
                query=dialog,
                document_keys=document_keys,
                labels=sort_labels(item, document_keys),
            ))

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item: TrainingDatasetItem = self.dataset[idx]

        query = item.query
        documents = []
        for key in item.document_keys:
            documents.append(ID_TO_DESCRIPTION[key])
        
        text = format_docs_prompts_func(query=query, docs=documents, special_tokens=self.tokenizer.special_tokens)
        inputs = self.tokenizer.tokenize(text)
        labels = torch.Tensor(item.labels)
        return EncodedDatasetItem(
            inputs=inputs,
            labels=labels,
        )