import datasets

# Load the SPR_BENCH dataset from local CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}

dataset = datasets.load_dataset("csv", data_files=data_files)

print(dataset)