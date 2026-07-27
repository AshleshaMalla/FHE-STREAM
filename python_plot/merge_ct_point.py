#!/usr/bin/env python3
import json
from pathlib import Path

# Load existing ring scaling results
with open("results/ring_scaling_seq_add.json") as f:
    merged = json.load(f)

# Load new CT data
with open("results/ct_seq_add_131072.json") as f:
    ct_data = json.load(f)

# Add CT benchmarks to merged
merged["benchmarks"].extend(ct_data["benchmarks"])

# Save back
with open("results/ring_scaling_seq_add.json", "w") as f:
    json.dump(merged, f, indent=2)

print("Added CT benchmark to ring_scaling_seq_add.json")
