import random 
from pathlib import Path
import json
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
    
    random.shuffle(dataset)
    dataset = dataset[:size]
    converted_dataset = [convert_dataset_item(DatasetItem(**item)) for item in dataset]
    with open(path.parent / "tagme_labeled_dataset.json", "w") as f:
        json.dump(converted_dataset, f, indent=4, ensure_ascii=False)
    