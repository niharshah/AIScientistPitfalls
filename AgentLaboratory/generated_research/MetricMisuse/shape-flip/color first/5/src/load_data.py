from datasets import load_dataset, DatasetDict
import pathlib

# Define the path to the SPR_BENCH dataset folder (adjust if needed)
data_path = pathlib.Path("SPR_BENCH")

# Function to compute color and shape variety for each sequence
def add_token_features(example):
    tokens = example["sequence"].split()
    color_set = set()
    shape_set = set()
    for token in tokens:
        # Token may have only shape (1 character) or shape+color (2 characters)
        if token:
            shape_set.add(token[0])
        if len(token) > 1:
            color_set.add(token[1])
    example["color_variety"] = len(color_set)
    example["shape_variety"] = len(shape_set)
    return example

# Load the datasets from local CSV files using HuggingFace's datasets library
spr_dataset = DatasetDict({
    "train": load_dataset("csv", data_files=str(data_path / "train.csv"), split="train", cache_dir=".cache_dsets"),
    "dev": load_dataset("csv", data_files=str(data_path / "dev.csv"), split="train", cache_dir=".cache_dsets"),
    "test": load_dataset("csv", data_files=str(data_path / "test.csv"), split="train", cache_dir=".cache_dsets")
})

# Apply the feature extraction to each split to add color and shape variety information
for split in spr_dataset.keys():
    spr_dataset[split] = spr_dataset[split].map(add_token_features)

# Quick prints to show that dataset prep is complete and to display an example
print("Loaded splits:", list(spr_dataset.keys()))
print("Example from train split:", spr_dataset["train"][0])