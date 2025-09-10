from datasets import load_dataset

# Load the SPR_BENCH benchmark dataset from local CSV files
dataset = load_dataset("csv", data_files={
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
})

# Print a sample from each split to verify the data loading.
print("Train sample:", dataset["train"][0])
print("Dev sample:", dataset["dev"][0])
print("Test sample:", dataset["test"][0])