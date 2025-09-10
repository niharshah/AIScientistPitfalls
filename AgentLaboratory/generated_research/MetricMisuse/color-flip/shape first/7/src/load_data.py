from datasets import load_dataset

# Load the SPR_BENCH benchmark data from local CSV files.
dataset = load_dataset(
    "csv", 
    data_files={
        "train": "SPR_BENCH/train.csv",
        "dev": "SPR_BENCH/dev.csv",
        "test": "SPR_BENCH/test.csv"
    },
    delimiter=","
)

# Basic checks to ensure the dataset is loaded correctly.
print("Train samples:", len(dataset["train"]))
print("Dev samples:", len(dataset["dev"]))
print("Test samples:", len(dataset["test"]))

# Print one example from the training set for verification.
print("Example from train set:", dataset["train"][0])