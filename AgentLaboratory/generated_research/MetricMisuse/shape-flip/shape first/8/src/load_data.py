from datasets import load_dataset

# Load the SPR_BENCH dataset from local CSV files: train, dev, and test splits.
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# Process each example to compute shape and color complexities.
# Each token is a shape glyph, possibly followed by a 1-letter color.
# We assume the first character of each token is the shape.
# If a token has more than one character, the second character represents the color.
def process_example(example):
    tokens = example["sequence"].split()
    shapes = set(token[0] for token in tokens)
    colors = set(token[1] for token in tokens if len(token) > 1)
    example["shape_complexity"] = len(shapes)
    example["color_complexity"] = len(colors)
    return example

# Apply the processing transformation to all dataset splits.
dataset = dataset.map(process_example)

# Print a summary of the processed dataset to check the new columns.
print(dataset)