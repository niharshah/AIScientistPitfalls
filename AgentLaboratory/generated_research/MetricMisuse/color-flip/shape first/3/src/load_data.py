from datasets import load_dataset

# Load dataset using local CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files, delimiter=",")

# Simple preprocessing: split the sequence string into tokens (shape + optional color) 
# and count number of tokens. This will be useful for further analysis.
dataset = dataset.map(lambda x: {"tokens": x["sequence"].split(), "num_tokens": len(x["sequence"].split())})

print(dataset)