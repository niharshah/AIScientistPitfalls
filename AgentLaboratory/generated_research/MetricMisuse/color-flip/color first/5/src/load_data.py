from datasets import load_dataset, DatasetDict
from pathlib import Path

# Define the local data path to the SPR_BENCH folder
DATA_PATH = Path("SPR_BENCH")

# Load the SPR_BENCH dataset using HuggingFace datasets library
dset = DatasetDict()
dset["train"] = load_dataset("csv", data_files=str(DATA_PATH / "train.csv"), split="train", cache_dir=".cache_dsets")
dset["dev"] = load_dataset("csv", data_files=str(DATA_PATH / "dev.csv"), split="train", cache_dir=".cache_dsets")
dset["test"] = load_dataset("csv", data_files=str(DATA_PATH / "test.csv"), split="train", cache_dir=".cache_dsets")

# Display the available splits and a sample instance from the training data
print("Loaded dataset splits:", list(dset.keys()))
print("Example from training set:", dset["train"][0])