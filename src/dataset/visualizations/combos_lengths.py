import json
from pathlib import Path

from src.constants.cross_encoder_descriptions import ALLOWED_COMBINATIONS

path = Path(__file__).parent / "token_lengths.json"
with open(path, "r") as f:
    token_lengths = json.load(f)

token_lengths = {item["id"]: item["len_tokens"] for item in token_lengths}


for combo in ALLOWED_COMBINATIONS:
    combo["len_tokens"] = sum(
        token_lengths[product] for product in combo["content"] + [combo["main_product"]]
    )

with open("src/dataset/visualizations/combos_lengths.json", "w") as f:
    json.dump(
        ALLOWED_COMBINATIONS,
        f,
        ensure_ascii=False,
        indent=4,
    )
