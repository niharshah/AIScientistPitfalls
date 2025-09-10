from datasets import load_dataset

# Load the SPR_BENCH dataset from local CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files, delimiter=",")

# Preprocess: tokenize the sequence by splitting on whitespace and ensure label is an integer.
dataset = dataset.map(lambda x: {"tokens": x["sequence"].split(), "label": int(x["label"])})

# Verify the dataset structure
print(dataset)