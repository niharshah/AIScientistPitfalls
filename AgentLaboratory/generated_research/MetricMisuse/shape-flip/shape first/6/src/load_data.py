from datasets import load_dataset

# Load the SPR_BENCH dataset from local CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# A simple mapping function to compute shape and color complexity for each example
def compute_complexities(example):
    tokens = example['sequence'].split()
    shape_set = set()
    color_set = set()
    for token in tokens:
        if token:  # Ensure token is not empty
            # The shape is the first character
            shape_set.add(token[0])
            # If the token has a color symbol (second character), add it
            if len(token) > 1:
                color_set.add(token[1])
    example["shape_complexity"] = len(shape_set)
    example["color_complexity"] = len(color_set)
    return example

# Apply the mapping to each split of the dataset
dataset = dataset.map(compute_complexities)

print(dataset)