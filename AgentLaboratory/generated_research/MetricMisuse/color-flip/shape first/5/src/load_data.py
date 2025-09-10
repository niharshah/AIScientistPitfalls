from datasets import load_dataset

# Define file paths for local SPR_BENCH CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "validation": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}

# Load the dataset using HuggingFace's datasets library
spr_bench = load_dataset("csv", data_files=data_files, delimiter=",")

# Display a brief overview of the loaded dataset
print(spr_bench)