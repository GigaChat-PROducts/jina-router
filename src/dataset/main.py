import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import json_repair
import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from src.dataset.clients import LLM
from src.dataset.config import DatasetConfig
from src.dataset.product_mapping import (
    ALLOWED_COMBINATIONS,
    AVAILABLE_COMBINATIONS,
    ID_TO_PRODUCT,
    product_name_to_id,
)
from src.dataset.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.dataset.schemas import D30Item, DatasetItem, Distribution, ItemClass
from src.utils import format_docs_prompts_func


def create_dataset(source_dataset: list[D30Item], dataset_config: DatasetConfig):
    np.random.seed(dataset_config.random_state)
    random.seed(dataset_config.random_state)
    item_ratio = math.ceil(dataset_config.target_size / len(source_dataset))
    dataset: list[DatasetItem] = []
    random.shuffle(source_dataset)
    for item in tqdm(source_dataset):
        dialog_len = len(item.dialog)
        item_class = np.random.choice(
            [ItemClass.TRAIN, ItemClass.VAL, ItemClass.TEST],
            p=[
                dataset_config.train_size,
                dataset_config.val_size,
                dataset_config.test_size,
            ],
        )
        for _ in range(item_ratio):
            if len(dataset) >= dataset_config.target_size:
                break
            a = random.randint(0, dialog_len - 1)
            b = random.randint(a + 1, dialog_len)
            prev_dialog = item.dialog[:a]
            current_dialog = item.dialog[a:b]
            next_dialog = item.dialog[b:]

            try:
                combination = ALLOWED_COMBINATIONS[
                    random.choice(
                        AVAILABLE_COMBINATIONS[product_name_to_id(item.products[0])]
                    )
                ]
            except IndexError:
                print(
                    f"Продукт '{item.products[0]}' не найден ни в одной комбинации. Пропускаем этот элемент."
                )
                continue

            products = combination["content"]
            products = [p for p in products if random.random() > dataset_config.dropout]
            embedded_item = format_docs_prompts_func(
                query="\n".join(current_dialog),
                docs=[ID_TO_PRODUCT[prod_id]["description"] for prod_id in products],
            )
            payload = {
                "text": embedded_item,
            }
            response = requests.post(
                "http://91.211.217.36:8022/api/v1/reranker/tokenize", json=payload
            ).json()

            dataset.append(
                DatasetItem(
                    item_id=item.item_id + f"_{a}_{b}",
                    item_class=item_class,
                    base_product=combination["main_product"],
                    products=products,
                    dialog=current_dialog,
                    gt_task_distribution=None,
                    gt_product_distribution=None,
                    current_product=None,
                    metadata={
                        "prev_dialog": prev_dialog,
                        "next_dialog": next_dialog,
                        "a": a,
                        "b": b,
                        "dialog_len": dialog_len,
                        "source_product": item.products[0],
                    },
                    len_tokens=len(response["data"]),
                )
            )
    with open("src/dataset/data/dataset.json", "w") as f:
        json.dump(
            [item.model_dump(mode="json") for item in dataset],
            f,
            ensure_ascii=False,
            indent=4,
        )
    return dataset


def label_dataset(dataset: list[DatasetItem], client: LLM):
    message_list = []
    for item in dataset:
        prev_dialog = "\n".join(item.metadata["prev_dialog"])
        products_with_descriptions = [
            ID_TO_PRODUCT[prod_id]["description"]
            for prod_id in [item.base_product] + item.products
        ]
        products_with_descriptions = json.dumps(
            products_with_descriptions, ensure_ascii=False, indent=2
        )
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append(
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    dialog="\n".join(item.dialog),
                    prev_dialog=prev_dialog,
                    products_with_descriptions=products_with_descriptions,
                ),
            }
        )
        message_list.append(messages)
    responses = client.call_sync(message_list=message_list)
    new_dataset = []
    for item, response in zip(dataset, responses):
        try:
            if response is None:
                raise ValueError("Received empty response from LLM")
            response = json_repair.loads(response["response"])
            item.gt_task_distribution = [
                Distribution(**dist) for dist in response["gt_task_distribution"]
            ]
            item.gt_product_distribution = [
                Distribution(**dist) for dist in response["gt_product_distribution"]
            ]
            item.gt_product_distribution_with_context = [
                Distribution(**dist)
                for dist in response["gt_product_distribution_with_context"]
            ]

            item.current_product = response["current_product"]
            new_dataset.append(item)
        except Exception as e:
            print(f"Error processing item {item.item_id}: {e}")
            continue

    with open("src/dataset/data/labeled_dataset.json", "w") as f:
        json.dump(
            [item.model_dump(mode="json") for item in new_dataset],
            f,
            ensure_ascii=False,
            indent=4,
        )


def upload_to_huggingface(dataset_path: Path, repo_id: str):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload labeled dataset at {str(datetime.now())}",
            folder_path=str(dataset_path),
            path_in_repo="/",
        )
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="dataset")
        upload_to_huggingface(dataset_path, repo_id)


def download_dataset(repo_id: str, dataset_path: Path):
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dataset_path)


if __name__ == "__main__":
    load_dotenv(".env")
    with open("src/dataset/data/d30_full_dialogs.json", "r") as f:
        dataset = [D30Item(**item) for item in json.load(f)]
    config = DatasetConfig()
    dataset = create_dataset(dataset, config)
    # label_dataset(
    #     dataset,
    #     client=LLM.from_giga_token(
    #         token=os.environ["GIGACHAT_TOKEN"], model="GigaChat-2-Max", max_threads=5
    #     ),
    # )
    # download_dataset(
    #     dataset_path=Path(__file__).parent / "data",
    #     repo_id="Hinter-Models/product-task-router",
    # )
