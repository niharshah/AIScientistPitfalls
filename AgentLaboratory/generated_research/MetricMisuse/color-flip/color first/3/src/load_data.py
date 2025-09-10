import datasets
from datasets import load_dataset

# Load the local SPR_BENCH dataset
ds = load_dataset("csv", data_files={"train": "SPR_BENCH/train.csv", 
                                       "dev": "SPR_BENCH/dev.csv", 
                                       "test": "SPR_BENCH/test.csv"})

# Define sets of shape glyphs and color letters
shape_glyphs = {"▲", "■", "●", "◆"}
color_letters = {"r", "g", "b", "y"}

# Mapping function to compute shape and color complexity for each example
def compute_complexities(example):
    tokens = example["sequence"].split()
    unique_shapes = set()
    unique_colors = set()
    for token in tokens:
        # Assuming the first character is the shape glyph
        if token and token[0] in shape_glyphs:
            unique_shapes.add(token[0])
        # If a color is given (as last character) and it is valid
        if len(token) > 1 and token[-1] in color_letters:
            unique_colors.add(token[-1])
    example["shape_complexity"] = len(unique_shapes)
    example["color_complexity"] = len(unique_colors)
    return example

# Apply the mapping to each split in the dataset
ds = ds.map(compute_complexities)

print(ds)