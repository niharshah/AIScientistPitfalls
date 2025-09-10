from datasets import load_dataset

# Load local SPR_BENCH dataset stored in CSV format.
# Directory structure:
# SPR_BENCH/
#    ├─ train.csv
#    ├─ dev.csv
#    └─ test.csv
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}

dataset = load_dataset("csv", data_files=data_files)

# Simple check to ensure the dataset is loaded.
print("Train examples:", len(dataset["train"]))
print("Dev examples:", len(dataset["dev"]))
print("Test examples:", len(dataset["test"]))

# If needed for further processing, you can inspect one sample as follows:
print("Example:", dataset["train"][0])