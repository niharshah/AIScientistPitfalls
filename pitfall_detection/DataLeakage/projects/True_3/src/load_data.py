from datasets import load_dataset

# Load the SPR_BENCH dataset from local CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files, delimiter=",")

# Process each example: split the sequence into tokens and convert the label to integer.
def process_example(example):
    # Splitting the sequence into individual tokens based on space.
    example["tokens"] = example["sequence"].split()
    # Convert label to integer if it isn't already
    example["label"] = int(example["label"])
    return example

dataset = dataset.map(process_example)

print(dataset)