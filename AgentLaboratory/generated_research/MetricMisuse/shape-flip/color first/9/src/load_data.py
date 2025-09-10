import pathlib
from datasets import load_dataset

# Define the path to the local SPR_BENCH directory containing train.csv, dev.csv, and test.csv
data_path = pathlib.Path("./SPR_BENCH")

# Load the dataset splits using HuggingFace's datasets library
spr_dataset = load_dataset(
    "csv",
    data_files={
        "train": str(data_path / "train.csv"),
        "dev": str(data_path / "dev.csv"),
        "test": str(data_path / "test.csv")
    },
    cache_dir=".hf_cache"
)

# Print the splits and a sample from the training split to verify proper loading
print("Splits loaded:", list(spr_dataset.keys()))
print("A training sample:", spr_dataset["train"][0])

# Define simple helper code to compute unique color and shape variety in a sequence
def count_unique_colors(sequence):
    # Assumes each token is a shape followed by an optional color character.
    return len({token[1] for token in sequence.split() if len(token) > 1})

def count_unique_shapes(sequence):
    # Assumes the first character of each token denotes the shape.
    return len({token[0] for token in sequence.split() if token})

# Demonstrate helper functions on the first training example
sample_sequence = spr_dataset["train"][0]["sequence"]
print("Unique color count in sample:", count_unique_colors(sample_sequence))
print("Unique shape count in sample:", count_unique_shapes(sample_sequence))