import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel

from src.constants.cross_encoder_descriptions import (
    ID_TO_DESCRIPTION,
    product_name_to_id,
)
from src.dataset import DatasetItem, ItemClass, Distribution
from src.training.tokenizer import ModelTokenizer, TokenizerOutput

DATASET_PATH = (
    Path(__file__).parent.parent / "dataset" / "data" / "labeled_dataset.json"
)


class TrainingDatasetItem(BaseModel):
    query: str
    document_keys: list[str]
    labels: list[float]

    def __post_init__(self):
        if len(self.document_keys) != len(self.labels):
            raise ValueError


@dataclass
class EncodedDatasetItem:
    query: str
    documents: list[str]
    labels: list[float]


class TrainingDataset:
    def __init__(self, mode: ItemClass, tokenizer: ModelTokenizer):
        if not os.path.exists(DATASET_PATH):
            raise ValueError(
                f"Path {DATASET_PATH} does not exist, ensure you downloaded it"
            )

        with open(DATASET_PATH) as f:
            data = json.load(f)

        self.dataset = []

        def extract_labels(keys: list[str], distributions: list[Distribution]) -> list[float]:
            res = []
            for key in keys:
                for distr in distributions:
                    if distr.name == key:
                        res.append(distr.probability)
                        break
                else:
                    raise ValueError(f"{key=}")
            return res

        task_keys = ["documents", "best_practices"]
        for d in data[:100]:
            item = DatasetItem(**d)
            if item.item_class != mode:
                continue
            dialog = "\n".join(item.dialog)
            task_labels = extract_labels(keys=task_keys, distributions=item.gt_task_distribution)
            self.dataset.append(TrainingDatasetItem(query=dialog, document_keys=task_keys, labels=task_labels))
            product_keys = [item.base_product] + item.products
            product_labels = extract_labels(keys=product_keys, distributions=item.gt_product_distribution)
            self.dataset.append(TrainingDatasetItem(query=dialog, document_keys=product_keys, labels=product_labels))
            
            
        print(f"{mode=}")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item: TrainingDatasetItem = self.dataset[idx]

        query = item.query
        documents = []
        for key in item.document_keys:
            documents.append(ID_TO_DESCRIPTION[key])

        labels = item.labels
        if len(labels) != len(documents):
            raise ValueError(f"{item.model_dump_json()}")
        return EncodedDatasetItem(
            query=query,
            documents=documents,
            labels=labels,
        )


    def collate_fn(self, batch: list[EncodedDatasetItem]):
        max_length = 2560
        texts = []
        labels = []
        for data in batch:
            text = self.tokenizer.format_data(
                query=data.query,
                docs=data.documents,
                max_length=max_length,
            )
            texts.append(text)
            labels.append(torch.Tensor(data.labels))
        
        inputs = self.tokenizer.tokenize(texts)

        return {
            **inputs,
            "labels": labels,
        }
