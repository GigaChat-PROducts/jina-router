import json
import random
from pathlib import Path
from collections import defaultdict

import polars as pl

from src.dataset.schemas import DatasetItem


def format_dialog(dialog: list[str]) -> str:
    return "\n".join(dialog)


def format_distribution(distribution: list) -> str:
    return "\n".join([f"{x.name}: {round(x.probability, 2)}" for x in distribution])


def convert_dataset_item(dataset_item: DatasetItem) -> dict:
    return {
        "item_id": dataset_item.item_id,
        "dialog": format_dialog(dataset_item.dialog),
        "task_distribution": format_distribution(
            dataset_item.gt_task_distribution or []
        ),
        "product_distribution": format_distribution(
            dataset_item.gt_product_distribution or []
        ),
        "prev_dialog": format_dialog(dataset_item.metadata.get("prev_dialog", [])),
        "current_product": dataset_item.current_product or "",
        "product_distribution_with_context": format_distribution(
            dataset_item.gt_product_distribution_with_context or []
        ),
    }


if __name__ == "__main__":
    size = 100

    random.seed(42)
    path = Path(__file__).parent / "data" / "labeled_dataset.json"
    with open(path, "r") as f:
        dataset = json.load(f)
    
    sizes = defaultdict(int)
    for item in dataset:
        sizes[item["item_source"]] += 1
    print(sizes)

    random.shuffle(dataset)
    dataset = dataset[:size]
    converted_dataset = [convert_dataset_item(DatasetItem(**item)) for item in dataset]
    save_path = path.parent / "tagme_labeled_dataset.csv"
    df = pl.DataFrame(converted_dataset)
    df.write_csv(save_path)
