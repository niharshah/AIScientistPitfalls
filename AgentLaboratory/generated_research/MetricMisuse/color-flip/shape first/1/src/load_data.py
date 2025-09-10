from datasets import load_dataset

# Load the SPR_BENCH dataset from local CSV files.
dataset = load_dataset(
    "csv",
    data_files={
        "train": "SPR_BENCH/train.csv",
        "dev": "SPR_BENCH/dev.csv",
        "test": "SPR_BENCH/test.csv"
    },
    delimiter=","
)

# A simple preprocessing step: split the 'sequence' column into a list of tokens.
def split_sequence(example):
    # Each token is expected to be in the format "shape+color" (e.g., "▲r", "■b").
    example["tokens"] = example["sequence"].split()
    return example

# Apply the split to each split of the dataset.
dataset = dataset.map(split_sequence)

print(dataset)