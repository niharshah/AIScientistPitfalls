import pathlib
from datasets import load_dataset

# Define the root directory for SPR_BENCH
data_root = pathlib.Path("SPR_BENCH")
files = {
    "train": str(data_root / "train.csv"),
    "dev": str(data_root / "dev.csv"),
    "test": str(data_root / "test.csv")
}

# Load the CSV datasets from local files using HuggingFace's dataset library
spr_dataset = load_dataset("csv", data_files=files, cache_dir=".cache_dsets")

# Add new computed columns for color and shape complexity to each split
# Using dataset.map for a simple transformation
spr_dataset["train"] = spr_dataset["train"].map(lambda x: {
    "color_variety": len({token[1] for token in x["sequence"].split() if len(token) > 1}),
    "shape_variety": len({token[0] for token in x["sequence"].split() if token})
})
spr_dataset["dev"] = spr_dataset["dev"].map(lambda x: {
    "color_variety": len({token[1] for token in x["sequence"].split() if len(token) > 1}),
    "shape_variety": len({token[0] for token in x["sequence"].split() if token})
})
spr_dataset["test"] = spr_dataset["test"].map(lambda x: {
    "color_variety": len({token[1] for token in x["sequence"].split() if len(token) > 1}),
    "shape_variety": len({token[0] for token in x["sequence"].split() if token})
})

# Print dataset splits and an example from the train split
print("Dataset splits loaded:", list(spr_dataset.keys()))
print("\nExample from train split:")
print(spr_dataset["train"][0])