from datasets import load_dataset

data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}

dataset = load_dataset("csv", data_files=data_files)

# Simple data preparation: Print out dataset sizes and a sample row from each split.
print("Train size:", len(dataset["train"]))
print("Dev size:", len(dataset["dev"]))
print("Test size:", len(dataset["test"]))

print("\nSample from Train Split:")
print(dataset["train"][0])