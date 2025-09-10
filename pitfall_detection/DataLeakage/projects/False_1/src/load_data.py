from datasets import load_dataset

# Load the SPR_BENCH dataset from local CSV files located in the 'SPR_BENCH' directory.
# The expected file layout is:
# SPR_BENCH/
#  ├─ train.csv   (20000 rows)
#  ├─ dev.csv     (5000 rows)
#  └─ test.csv    (10000 rows)

data_files = {
  "train": "SPR_BENCH/train.csv",
  "dev": "SPR_BENCH/dev.csv",
  "test": "SPR_BENCH/test.csv"
}

# Load the dataset using HuggingFace datasets.
dataset = load_dataset("csv", data_files=data_files)

# Print a sample from each split to verify the data loading.
print("Sample from train split:", dataset["train"][0])
print("Sample from dev split:", dataset["dev"][0])
print("Sample from test split:", dataset["test"][0])