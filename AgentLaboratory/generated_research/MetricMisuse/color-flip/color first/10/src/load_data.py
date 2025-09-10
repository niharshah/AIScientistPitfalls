from datasets import load_dataset

# Load the SPR_BENCH dataset from local CSV files.
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

def process_example(example):
    # Split the sequence string into tokens by whitespace.
    tokens = example["sequence"].split()
    example["tokens"] = tokens

    shapes = []
    colors = []
    # For each token, separate out the shape glyph and color.
    for token in tokens:
        # Assume shape characters are among these glyphs and colors are lowercase letters.
        shape = "".join([c for c in token if c in "▲■●◆"])
        color = "".join([c for c in token if c in "rgby"])
        shapes.append(shape)
        colors.append(color)
    example["shapes"] = shapes
    example["colors"] = colors

    # Calculate complexities as the number of unique shapes and colors.
    example["shape_complexity"] = len(set(shapes))
    example["color_complexity"] = len(set(colors))
    return example

# Apply the processing function to all splits.
dataset = dataset.map(process_example)

print(dataset)