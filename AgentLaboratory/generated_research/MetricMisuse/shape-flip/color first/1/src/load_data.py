import pathlib
from datasets import load_dataset, DatasetDict

# Define the local path to the SPR_BENCH folder
data_folder = pathlib.Path("./SPR_BENCH/")

# Create a mapping for the three CSV splits using local paths
data_files = {
    "train": str(data_folder / "train.csv"),
    "dev": str(data_folder / "dev.csv"),
    "test": str(data_folder / "test.csv")
}

# Load the dataset splits into a HuggingFace DatasetDict
spr_dataset = DatasetDict({
    split: load_dataset("csv", data_files=path, split="train", cache_dir=".cache_dsets")
    for split, path in data_files.items()
})

# Inline lambdas to calculate unique color and shape counts for each token sequence.
# For color, we consider the second character (if available) from each token.
# For shape, we consider the first character of each token.
color_count = lambda seq: len({token[1] for token in seq.strip().split() if len(token) > 1})
shape_count = lambda seq: len({token[0] for token in seq.strip().split() if token})

# Add extra columns "color_variety" and "shape_variety" to each dataset split
for split in spr_dataset.keys():
    spr_dataset[split] = spr_dataset[split].map(lambda ex: {
        "color_variety": color_count(ex["sequence"]),
        "shape_variety": shape_count(ex["sequence"])
    })

# Display a sample row from the training data with the new computed features
print("Training sample with computed features:")
print(spr_dataset["train"][0])