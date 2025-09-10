from datasets import load_dataset

# Load the local SPR_BENCH dataset from CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# Print out the number of samples in each split
print("Train samples:", len(dataset["train"]))
print("Dev samples:", len(dataset["dev"]))
print("Test samples:", len(dataset["test"]))