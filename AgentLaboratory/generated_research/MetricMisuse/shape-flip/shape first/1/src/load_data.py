import datasets

# Load the local SPR_BENCH dataset from CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = datasets.load_dataset("csv", data_files=data_files, delimiter=",")

# Display basic information about the dataset splits
print("SPR_BENCH Dataset:")
print("Train instances:", len(dataset["train"]))
print("Dev instances:", len(dataset["dev"]))
print("Test instances:", len(dataset["test"]))