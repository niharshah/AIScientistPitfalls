from datasets import load_dataset

# Load the SPR_BENCH dataset from local CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# Print some basic information about the loaded splits
print("Train split count:", len(dataset["train"]))
print("Dev split count:", len(dataset["dev"]))
print("Test split count:", len(dataset["test"]))

# Show an example from the train split to verify the data format
print("Example train instance:", dataset["train"][0])