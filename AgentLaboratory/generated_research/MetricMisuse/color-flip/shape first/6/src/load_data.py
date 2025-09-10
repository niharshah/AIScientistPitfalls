from datasets import load_dataset

# Load SPR_BENCH dataset from local CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# Define the sets for shape and color glyphs
shape_glyphs = {"▲", "■", "●", "◆"}
color_glyphs = {"r", "g", "b", "y"}

# Compute shape and color complexity for each example
def add_complexity(example):
    tokens = example["sequence"].split()
    unique_shapes = set()
    unique_colors = set()
    for token in tokens:
        if token and token[0] in shape_glyphs:
            unique_shapes.add(token[0])
        if len(token) > 1 and token[1] in color_glyphs:
            unique_colors.add(token[1])
    example["shape_complexity"] = len(unique_shapes)
    example["color_complexity"] = len(unique_colors)
    return example

# Apply the complexity computation to all splits
for split in dataset.keys():
    dataset[split] = dataset[split].map(add_complexity)

print(dataset)