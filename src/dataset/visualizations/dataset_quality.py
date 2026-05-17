import json
from collections import Counter, defaultdict

import polars as pl

with open("src/dataset/data/tagme_quality.json", "r") as f:
    data = json.load(f)

new_data = defaultdict(lambda: defaultdict(lambda: Counter()))
for item in data:
    for k, v in item["result"].items():
        new_data[item["file_name"]][k][v] += 1

data = []
for item in new_data.values():
    curr = {}
    for k, v in item.items():
        curr[k] = int(v.most_common(1)[0][0])
    data.append(curr)

df = pl.DataFrame(data)

print(df.mean())

for row in df.mean().iter_rows(named=True):
    for col, val in row.items():
        print(f"{col}: {val:.4f}" if isinstance(val, float) else f"{col}: {val}")
