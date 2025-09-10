from datasets import load_dataset, DatasetDict

# Define paths for the SPR_BENCH dataset files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}

# Load each split as a HuggingFace dataset from local CSV files
spr_dataset = DatasetDict()
for split, path in data_files.items():
    spr_dataset[split] = load_dataset("csv", data_files=path, split="train", cache_dir=".cache_dsets")

# Simple helper functions to compute complexity metrics
def count_color_complexity(sequence):
    # Count the unique color letters in the sequence (token[1] if available)
    tokens = sequence.strip().split()
    colors = set(token[1] for token in tokens if len(token) > 1)
    return len(colors)

def count_shape_complexity(sequence):
    # Count the unique shape symbols (first char of each token)
    tokens = sequence.strip().split()
    shapes = set(token[0] for token in tokens if token)
    return len(shapes)

# Map the complexity metrics to each dataset split; add as new columns
for split in spr_dataset.keys():
    # Use the dataset.map to compute and add new columns: color_complexity and shape_complexity
    spr_dataset[split] = spr_dataset[split].map(
        lambda example: {
            "color_complexity": count_color_complexity(example["sequence"]),
            "shape_complexity": count_shape_complexity(example["sequence"])
        }
    )

# Display a sample of the train split to verify the dataset preparation
print("Train sample:")
print(spr_dataset["train"][0])