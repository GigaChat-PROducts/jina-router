from collections import defaultdict

from src.constants.cross_encoder_descriptions import (
    ALLOWED_COMBINATIONS,
    DATA_DESCRIPTIONS,
    PRESENTATION_ROWS,
    ID_TO_PRODUCT,
    NAME_TO_TASK,
    product_name_to_id,
)

AVAILABLE_COMBINATIONS = defaultdict(list)
for idx, comb in enumerate(ALLOWED_COMBINATIONS):
    for item in [comb["main_product"]] + comb["content"]:
        AVAILABLE_COMBINATIONS[item].append(idx)


def validate_sets(presentation_rows, allowed_sets):
    valid_ids = {row["id"] for row in presentation_rows}
    for s in allowed_sets:
        for item in s["content"] + [s["main_product"]]:
            if item not in valid_ids:
                raise ValueError(f"Invalid content id '{item}' in set '{s['id']}'")


if __name__ == "__main__":
    validate_sets(PRESENTATION_ROWS, ALLOWED_COMBINATIONS)
    print("All sets are valid.")
