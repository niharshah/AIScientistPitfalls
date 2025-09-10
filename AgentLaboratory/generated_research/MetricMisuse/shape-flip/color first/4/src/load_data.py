import pathlib
from datasets import load_dataset, DatasetDict

# Define the path to the local SPR_BENCH dataset directory
data_path = pathlib.Path("SPR_BENCH")

# Load datasets from CSV files using the HuggingFace datasets library
spr_data = DatasetDict()
spr_data["train"] = load_dataset("csv", data_files=str(data_path / "train.csv"), split="train", cache_dir=".cache_dsets")
spr_data["dev"] = load_dataset("csv", data_files=str(data_path / "dev.csv"), split="train", cache_dir=".cache_dsets")
spr_data["test"] = load_dataset("csv", data_files=str(data_path / "test.csv"), split="train", cache_dir=".cache_dsets")

# Simple helper functions to compute color and shape variety
def count_color_variety(seq):
    tokens = seq.strip().split()
    # Color is the second character of the token if present
    colors = {token[1] for token in tokens if len(token) > 1}
    return len(colors)

def count_shape_variety(seq):
    tokens = seq.strip().split()
    # Shape is the first character of the token
    shapes = {token[0] for token in tokens if token}
    return len(shapes)

# Augment each split with additional fields: color_variety and shape_variety
for split in ["train", "dev", "test"]:
    spr_data[split] = spr_data[split].map(lambda x: {
        "color_variety": count_color_variety(x["sequence"]),
        "shape_variety": count_shape_variety(x["sequence"])
    })

# Print basic info to verify loading and transformation
print("Loaded SPR_BENCH splits:", list(spr_data.keys()))
print("Example from train split:")
print(spr_data["train"][0])