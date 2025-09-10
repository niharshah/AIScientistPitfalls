from datasets import load_dataset, DatasetDict
import pathlib

# Define the local data directory path
data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")

# Load each CSV file from the local directory using HuggingFace's datasets library
spr_bench = DatasetDict({
    "train": load_dataset("csv", data_files=str(data_path / "train.csv"), split="train", cache_dir=".cache_dsets"),
    "dev": load_dataset("csv", data_files=str(data_path / "dev.csv"), split="train", cache_dir=".cache_dsets"),
    "test": load_dataset("csv", data_files=str(data_path / "test.csv"), split="train", cache_dir=".cache_dsets")
})

print("Loaded SPR_BENCH dataset splits:", list(spr_bench.keys()))
print("\nExample from train dataset:")
print(spr_bench["train"][0])