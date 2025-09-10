from datasets import load_dataset

# Load the SPR_BENCH dataset from local CSV files.
dataset = load_dataset(
    "csv",
    data_files={
        "train": "SPR_BENCH/train.csv",
        "dev": "SPR_BENCH/dev.csv",
        "test": "SPR_BENCH/test.csv"
    }
)

# Define a simple mapping function to compute shape and color complexity.
def compute_complexities(example):
    tokens = example["sequence"].split()
    shapes = set()
    colors = set()
    for token in tokens:
        if token:  # ensure token is not empty
            shapes.add(token[0])
            if len(token) > 1:
                colors.add(token[1])
    example["shape_complexity"] = len(shapes)
    example["color_complexity"] = len(colors)
    return example

# Apply the mapping function to each split.
dataset = dataset.map(compute_complexities)

print(dataset)