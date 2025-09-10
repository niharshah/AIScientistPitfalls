import random
from datasets import load_dataset, Dataset

# Load SPR_BENCH local CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
spr_dataset = load_dataset("csv", data_files=data_files)

# Process SPR_BENCH dataset: split sequence into tokens and cast label to int
for split in spr_dataset.keys():
    spr_dataset[split] = spr_dataset[split].map(lambda x: {"tokens": x["sequence"].split(), "label": int(x["label"])})

print("SPR_BENCH dataset loaded and preprocessed:")
print(spr_dataset)

# Synthesize a simple synthetic dataset following SPR guidelines:
# Each sequence has L=10 tokens; each token is a shape from {▲, ■, ●, ◆} and a color from {r, g, b, y}.
shapes = ["▲", "■", "●", "◆"]
colors = ["r", "g", "b", "y"]

num_synthetic_samples = 100
synthetic_data = {"id": [], "sequence": [], "tokens": [], "label": []}

for i in range(num_synthetic_samples):
    token_list = []
    for j in range(10):
        # Create token by combining a random shape with a random color
        token = random.choice(shapes) + random.choice(colors)
        token_list.append(token)
    # Join tokens with a space
    seq = " ".join(token_list)
    # For label, use a simple rule: accept (1) if there are at least 3 unique shapes, else reject (0)
    unique_shapes = set([token[0] for token in token_list])
    label = 1 if len(unique_shapes) >= 3 else 0
    synthetic_data["id"].append(f"synthetic_{i}")
    synthetic_data["sequence"].append(seq)
    synthetic_data["tokens"].append(token_list)
    synthetic_data["label"].append(label)

synthetic_dataset = Dataset.from_dict(synthetic_data)
print("Synthetic dataset created:")
print(synthetic_dataset)