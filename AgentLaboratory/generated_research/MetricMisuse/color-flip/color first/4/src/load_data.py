import pathlib
from datasets import load_dataset, DatasetDict

# Set the path for the local SPR_BENCH directory
data_path = pathlib.Path("SPR_BENCH")

# Load the SPR_BENCH datasets from CSV files as a local HuggingFace dataset
spr_datasets = DatasetDict({
    "train": load_dataset(
        "csv", 
        data_files=str(data_path / "train.csv"), 
        split="train", 
        cache_dir=".cache_dsets"
    ),
    "dev": load_dataset(
        "csv", 
        data_files=str(data_path / "dev.csv"), 
        split="train", 
        cache_dir=".cache_dsets"
    ),
    "test": load_dataset(
        "csv", 
        data_files=str(data_path / "test.csv"), 
        split="train", 
        cache_dir=".cache_dsets"
    ),
})

# Print information about the loaded dataset splits and a sample training instance
print("Dataset splits loaded:", list(spr_datasets.keys()))
print("Sample training instance:", spr_datasets["train"][0])