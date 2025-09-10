from datasets import load_dataset, DatasetDict
import pathlib

# Define dataset directory (local folder containing train.csv, dev.csv, and test.csv)
dataset_path = pathlib.Path("SPR_BENCH")

# Function to augment each data instance with weak explanation sketches and meta-features.
def add_explanation(example):
    tokens = example["sequence"].split()
    colors = set()
    shapes = set()
    for token in tokens:
        # Each token consists of a shape glyph and, optionally, a one-letter color.
        if len(token) > 1:
            shapes.add(token[0])
            colors.add(token[1])
        else:
            shapes.add(token)
    # Create a weak explanation string that gives the number of unique colors and shapes.
    explanation = f"Colors: {len(colors)}; Shapes: {len(shapes)}"
    return {"explanation": explanation, "color_count": len(colors), "shape_count": len(shapes)}

# Load the local SPR_BENCH dataset splits using HuggingFace datasets.
spr_data = DatasetDict()
for split in ["train", "dev", "test"]:
    csv_file = str(dataset_path / f"{split}.csv")
    ds = load_dataset("csv", data_files=csv_file, split="train", cache_dir=".cache_dsets")
    ds = ds.map(add_explanation)
    spr_data[split] = ds

# Print confirmation of loaded splits and display an example from the training set.
print("Loaded dataset splits:", list(spr_data.keys()))
print("Example from the train split:", spr_data["train"][0])