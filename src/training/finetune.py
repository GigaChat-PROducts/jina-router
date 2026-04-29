from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.dataset.product_mapping import DATA_DESCRIPTIONS, ID_TO_PRODUCT
from src.dataset.schemas import DatasetItem, Distribution, ItemClass

SPECIAL_TOKENS = {
    "query_embed_token": "<|rerank_token|>",
    "doc_embed_token": "<|embed_token|>",
}
TASK_DOC_NAMES = [item["name"] for item in DATA_DESCRIPTIONS]
TASK_TARGET_NAMES = {"documents": "factology", "best_practices": "sales_practices"}
PRODUCT_NAMES = [item["name"] for item in ID_TO_PRODUCT.values()]


def build_prompt(query: str, docs: list[str], instruction: str) -> str:
    cleaned_query = query
    for token in SPECIAL_TOKENS.values():
        cleaned_query = cleaned_query.replace(token, "")
    cleaned_docs = []
    for doc in docs:
        cleaned_doc = doc
        for token in SPECIAL_TOKENS.values():
            cleaned_doc = cleaned_doc.replace(token, "")
        cleaned_docs.append(cleaned_doc)

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    prompt = (
        f"I will provide you with {len(cleaned_docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {cleaned_query}\n"
    )
    prompt += f"<instruct>\n{instruction}\n</instruct>\n"

    doc_prompts = [
        f'<passage id="{index}">\n{doc}{SPECIAL_TOKENS["doc_embed_token"]}\n</passage>'
        for index, doc in enumerate(cleaned_docs)
    ]
    prompt += "\n".join(doc_prompts) + "\n"
    prompt += f"<query>\n{cleaned_query}{SPECIAL_TOKENS['query_embed_token']}\n</query>"

    return prefix + prompt + suffix


@dataclass(frozen=True)
class TrainingExample:
    item_id: str
    kind: str
    prompt: str
    labels: list[float]
    num_labels: int


class RankingDataset:
    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


class GroupedBatchSampler:
    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, length in enumerate(self.lengths):
            buckets[length].append(index)

        lengths = list(buckets)
        if self.shuffle:
            random.shuffle(lengths)

        for length in lengths:
            indices = buckets[length]
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / max(self.batch_size, 1))


class RankingCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[TrainingExample]) -> dict[str, object]:
        import torch

        prompts = [feature.prompt for feature in features]
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [feature.labels for feature in features], dtype=torch.float32
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }


def _normalize_distribution(items: Iterable[Distribution]) -> list[Distribution]:
    items = list(items)
    total = sum(item.probability for item in items)
    if total <= 0:
        raise ValueError("Distribution is empty or has zero mass")
    return [
        Distribution(name=item.name, probability=item.probability / total)
        for item in items
    ]


def _distribution_vector(
    distributions: list[Distribution] | None,
    order: list[str],
    *,
    allow_missing: bool = False,
) -> list[float]:
    if not distributions:
        raise ValueError("Missing target distribution")
    normalized = _normalize_distribution(distributions)
    values = {item.name: item.probability for item in normalized}
    missing = [name for name in order if name not in values]
    if missing and not allow_missing:
        raise ValueError(f"Target distribution is missing labels: {missing}")
    vector = [values.get(name, 0.0) for name in order]
    total = sum(vector)
    if total <= 0:
        raise ValueError(
            "Target distribution has zero probability mass after alignment"
        )
    return [value / total for value in vector]


def _normalize_product_label_name(name: str) -> str:
    if name in PRODUCT_NAMES:
        return name

    # Some labels include description text after the product name.
    for product_name in PRODUCT_NAMES:
        if name.startswith(product_name + " ") or name.startswith(product_name + "-"):
            return product_name
        if name.startswith(product_name + " -") or name.startswith(product_name + " —"):
            return product_name
    return name


def _normalize_product_distribution(
    distributions: list[Distribution] | None,
) -> list[Distribution] | None:
    if distributions is None:
        return None

    merged: dict[str, float] = defaultdict(float)
    for item in distributions:
        normalized_name = _normalize_product_label_name(item.name)
        merged[normalized_name] += item.probability
    return [
        Distribution(name=name, probability=probability)
        for name, probability in merged.items()
    ]


def _build_task_example(item: DatasetItem) -> TrainingExample:
    docs = [row["description"] for row in DATA_DESCRIPTIONS]
    prompt = build_prompt(
        query="\n".join(item.dialog),
        docs=docs,
        instruction="Rank the task descriptions by how well they match the dialog.",
    )
    target_order = [TASK_TARGET_NAMES[name] for name in TASK_DOC_NAMES]
    labels = _distribution_vector(item.gt_task_distribution, target_order)
    return TrainingExample(
        item_id=f"{item.item_id}::task",
        kind="task",
        prompt=prompt,
        labels=labels,
        num_labels=len(labels),
    )


def _build_product_example(item: DatasetItem) -> TrainingExample:
    product_ids = [item.base_product, *item.products]
    docs = [ID_TO_PRODUCT[product_id]["description"] for product_id in product_ids]
    prompt = build_prompt(
        query="\n".join(item.dialog),
        docs=docs,
        instruction="Rank the products by how well they match the dialog.",
    )
    target_distributions = (
        item.gt_product_distribution_with_context
        if item.gt_product_distribution_with_context
        else item.gt_product_distribution
    )
    target_distributions = _normalize_product_distribution(target_distributions)
    product_names = [ID_TO_PRODUCT[product_id]["name"] for product_id in product_ids]
    if target_distributions is None:
        raise ValueError("Missing product target distribution")

    target_names = {distribution.name for distribution in target_distributions}
    unknown_labels = [name for name in target_names if name not in product_names]
    if unknown_labels:
        raise ValueError(
            f"Product target has labels outside item products: {unknown_labels}; item products={product_names}"
        )

    if len(target_distributions) != len(product_names):
        raise ValueError(
            f"Product target count mismatch: labels={len(target_distributions)}, products={len(product_names)}"
        )

    labels = _distribution_vector(
        target_distributions, product_names, allow_missing=False
    )
    return TrainingExample(
        item_id=f"{item.item_id}::product",
        kind="product",
        prompt=prompt,
        labels=labels,
        num_labels=len(labels),
    )


def build_examples(
    dataset_path: Path,
    split: ItemClass | None = None,
) -> tuple[list[DatasetItem], list[TrainingExample], list[TrainingExample]]:
    raw_items = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset = [
        DatasetItem.model_validate(item)
        for item in raw_items
        if item.get("gt_task_distribution") and item.get("gt_product_distribution")
    ]
    if split is not None:
        dataset = [item for item in dataset if item.item_class == split]

    valid_dataset: list[DatasetItem] = []
    task_examples: list[TrainingExample] = []
    product_examples: list[TrainingExample] = []
    dropped_items = 0
    for item in dataset:
        try:
            task_example = _build_task_example(item)
            product_example = _build_product_example(item)
        except ValueError:
            dropped_items += 1
            continue
        valid_dataset.append(item)
        task_examples.append(task_example)
        product_examples.append(product_example)

    if dropped_items:
        print(f"Dropped {dropped_items} malformed items while building examples")
    return valid_dataset, task_examples, product_examples


def build_metrics(
    model,
    examples: list[TrainingExample],
    tokenizer,
    batch_size: int,
    max_length: int,
    device,
) -> dict[str, float]:
    if not examples:
        return {"kl_div": 0.0, "samples": 0.0}

    import torch
    from torch.utils.data import DataLoader

    dataset = RankingDataset(examples)
    sampler = GroupedBatchSampler(
        lengths=[example.num_labels for example in examples],
        batch_size=batch_size,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=RankingCollator(tokenizer, max_length=max_length),
    )

    total_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = labels.to(device)
            outputs = model(**batch, labels=labels)
            batch_size_value = labels.shape[0]
            total_loss += float(outputs.loss.detach().cpu()) * batch_size_value
            total_samples += batch_size_value

    denominator = max(total_samples, 1)
    return {"kl_div": total_loss / denominator, "samples": float(total_samples)}


def _resolve_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune JinaForRanking with HF Trainer"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("src/dataset/data/labeled_dataset.json"),
    )
    parser.add_argument(
        "--model-name-or-path", type=str, default="models/jina_reranker"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/jina_reranker_finetuned")
    )
    parser.add_argument("--mlflow-experiment", type=str, default="jina-router-finetune")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("mlflow is required to run fine-tuning") from error

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    from src.training.modeling import JinaForRanking

    class GroupedBatchTrainer(Trainer):
        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer requires a train dataset")
            lengths = [example.num_labels for example in self.train_dataset]
            sampler = GroupedBatchSampler(
                lengths=lengths,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    model = JinaForRanking.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    model.config.use_cache = False
    device = _resolve_device()
    model.to(device)

    _, train_task_examples, train_product_examples = build_examples(
        args.dataset_path, ItemClass.TRAIN
    )
    _, val_task_examples, val_product_examples = build_examples(
        args.dataset_path, ItemClass.VAL
    )
    train_examples = train_task_examples + train_product_examples

    collator = RankingCollator(tokenizer=tokenizer, max_length=args.max_length)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    trainer = GroupedBatchTrainer(
        model=model,
        args=training_args,
        train_dataset=RankingDataset(train_examples),
        data_collator=collator,
    )

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset_path": str(args.dataset_path),
                "model_name_or_path": args.model_name_or_path,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "max_length": args.max_length,
                "seed": args.seed,
                "train_task_samples": len(train_task_examples),
                "train_product_samples": len(train_product_examples),
                "val_task_samples": len(val_task_examples),
                "val_product_samples": len(val_product_examples),
            }
        )

        train_result = trainer.train()
        mlflow.log_metrics({"train_loss": float(train_result.training_loss)})

        train_task_metrics = build_metrics(
            model=model,
            examples=train_task_examples,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            device=device,
        )
        train_product_metrics = build_metrics(
            model=model,
            examples=train_product_examples,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            device=device,
        )
        val_task_metrics = build_metrics(
            model=model,
            examples=val_task_examples,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            device=device,
        )
        val_product_metrics = build_metrics(
            model=model,
            examples=val_product_examples,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            device=device,
        )

        mlflow.log_metrics(
            {
                "train_task_kl_div": train_task_metrics["kl_div"],
                "train_product_kl_div": train_product_metrics["kl_div"],
                "val_task_kl_div": val_task_metrics["kl_div"],
                "val_product_kl_div": val_product_metrics["kl_div"],
                "train_task_samples": train_task_metrics["samples"],
                "train_product_samples": train_product_metrics["samples"],
                "val_task_samples": val_task_metrics["samples"],
                "val_product_samples": val_product_metrics["samples"],
            }
        )

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
