import json

import matplotlib.pyplot as plt
import numpy as np

# Load dataset
with open("src/dataset/data/dataset.json", "r") as f:
    source_dataset = json.load(f)

# Extract token lengths
lengths = [item["len_tokens"] for item in source_dataset]

# Basic stats
print(
    f"Max: {max(lengths)}, Min: {min(lengths)}, Mean: {sum(lengths) / len(lengths):.2f}"
)

# Create bins with 100-token intervals
bin_size = 100
max_len = max(lengths)
bins = np.arange(0, max_len + bin_size, bin_size)

# Plot histogram
plt.figure(figsize=(12, 6))
plt.hist(lengths, bins=bins, edgecolor="black", alpha=0.7, color="skyblue")
plt.xlabel("Token Length (bins of 100)")
plt.ylabel("Frequency")
plt.title("Distribution of Token Lengths")
plt.grid(axis="y", alpha=0.5, linestyle="--")
plt.tight_layout()
plt.savefig("token_length_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Optional: print distribution stats per bin
bin_counts, bin_edges = np.histogram(lengths, bins=bins)
print("\nDistribution per 100-token bin (non-empty only):")
for count, start, end in zip(bin_counts, bin_edges[:-1], bin_edges[1:]):
    if count > 0:
        pct = count / len(lengths) * 100
        print(f"{start:6d}-{end:6d} tokens: {count:6d} items ({pct:5.2f}%)")
