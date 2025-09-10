from datasets import load_dataset, DatasetDict
import pathlib

# Set the local dataset path (adjust if necessary)
data_path = pathlib.Path("SPR_BENCH")

# Helper function to count unique color glyphs in a sequence.
def count_color_variety(seq):
    # Each token is expected to be a shape followed by an optional 1-letter color, e.g., "▲r".
    # We only count tokens with a color (length > 1).
    return len(set(token[1] for token in seq.strip().split() if len(token) > 1))

# Helper function to count unique shape glyphs in a sequence.
def count_shape_variety(seq):
    # The first character of each token represents the shape.
    return len(set(token[0] for token in seq.strip().split() if token))

# Load the SPR_BENCH dataset from CSV files using HuggingFace datasets.
def load_spr_data(root):
    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = load_dataset(
            "csv", 
            data_files=str(root / f"{split}.csv"), 
            split="train",           # each CSV is treated as a single split
            cache_dir=".cache_dsets"   # optional cache directory for HF datasets
        )
    return dset

spr_data = load_spr_data(data_path)

# Enhance each split of the dataset by computing color and shape complexities
for split in spr_data.keys():
    spr_data[split] = spr_data[split].map(lambda ex: {
        "color_complexity": count_color_variety(ex["sequence"]),
        "shape_complexity": count_shape_variety(ex["sequence"])
    })

print("Loaded splits:", list(spr_data.keys()))
print("Example from train split:", spr_data["train"][0])