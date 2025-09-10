from datasets import load_dataset, DatasetDict
import pathlib

# Set local path for the SPR_BENCH dataset folder
data_path = pathlib.Path("./SPR_BENCH/")

# Load the dataset splits using HuggingFace's load_dataset with local CSV files
spr_bench = DatasetDict()
for split, file_name in zip(["train", "dev", "test"], ["train.csv", "dev.csv", "test.csv"]):
    spr_bench[split] = load_dataset("csv", data_files=str(data_path / file_name), split="train", cache_dir=".cache_dsets")

# Print splits and display an example row from the train split
print("Dataset splits:", list(spr_bench.keys()))
print("\nExample from train split:")
print(spr_bench["train"][0])