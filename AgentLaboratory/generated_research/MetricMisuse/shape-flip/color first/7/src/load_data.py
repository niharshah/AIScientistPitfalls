from datasets import load_dataset, DatasetDict
import pathlib

# Define the local path for SPR_BENCH benchmark files
data_path = pathlib.Path("./SPR_BENCH/")

# Load the dataset splits using HuggingFace datasets, treating each CSV as a separate split
dset = DatasetDict({
    "train": load_dataset("csv", data_files=str(data_path / "train.csv"), split="train", cache_dir=".cache_dsets"),
    "dev": load_dataset("csv", data_files=str(data_path / "dev.csv"), split="train", cache_dir=".cache_dsets"),
    "test": load_dataset("csv", data_files=str(data_path / "test.csv"), split="train", cache_dir=".cache_dsets")
})

# Add additional fields for color and shape complexity
# Color complexity: count of unique color glyphs (second character in token, if present)
# Shape complexity: count of unique shape glyphs (first character of each token)
dset = dset.map(lambda x: {
    "color_complexity": len(set(token[1] for token in x["sequence"].split() if len(token) > 1)),
    "shape_complexity": len(set(token[0] for token in x["sequence"].split() if token))
})

# Display a sample instance from the training set for verification
print("Available splits:", list(dset.keys()))
print("Sample from Train split:", dset["train"][0])