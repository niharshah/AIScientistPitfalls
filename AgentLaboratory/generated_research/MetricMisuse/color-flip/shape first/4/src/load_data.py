from datasets import load_dataset

# Load local SPR_BENCH dataset from CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files, delimiter=",")

# Print dataset info to ensure correct loading
print(dataset)