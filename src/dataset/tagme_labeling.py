def format_dialog(dialog: list[str]) -> str:
    return "\n".join(dialog)


def format_distribution(distribution: list) -> str:
    return "\n".join([f"{x.name}: {round(x.probability, 2)}" for x in distribution])


def convert_dataset_item(dataset_item) -> dict:
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
