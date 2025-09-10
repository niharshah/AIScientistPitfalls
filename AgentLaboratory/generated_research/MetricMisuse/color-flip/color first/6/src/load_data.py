import pathlib
from datasets import load_dataset, DatasetDict

# Set the local data directory for SPR_BENCH
data_dir = pathlib.Path("SPR_BENCH")

# Load the dataset CSV files using HuggingFace's datasets library
spr_dataset = DatasetDict({
    "train": load_dataset(
        "csv",
        data_files=str(data_dir / "train.csv"),
        split="train",
        cache_dir=".cache_dsets"
    ),
    "dev": load_dataset(
        "csv",
        data_files=str(data_dir / "dev.csv"),
        split="train",
        cache_dir=".cache_dsets"
    ),
    "test": load_dataset(
        "csv",
        data_files=str(data_dir / "test.csv"),
        split="train",
        cache_dir=".cache_dsets"
    )
})

print("Loaded dataset splits:", list(spr_dataset.keys()))
print("Example from train split:", spr_dataset["train"][0])