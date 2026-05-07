import json

import requests
from tqdm import tqdm

from src.constants.cross_encoder_descriptions import PRESENTATION_ROWS

for row in tqdm(PRESENTATION_ROWS):
    payload = {
        "text": row["description"],
    }
    response = requests.post(
        "http://91.211.217.36:8022/api/v1/reranker/tokenize", json=payload
    ).json()
    row["len_tokens"] = len(response["data"])


with open("src/dataset/visualizations/token_lengths.json", "w") as f:
    json.dump(
        PRESENTATION_ROWS,
        f,
        ensure_ascii=False,
        indent=4,
    )
