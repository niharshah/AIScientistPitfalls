from datasets import load_dataset

# Load the SPR_BENCH benchmark from local CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
spr_dataset = load_dataset("csv", data_files=data_files, cache_dir=".cache_dsets")

# Add two new columns to each instance: color_count and shape_count.
spr_dataset = spr_dataset.map(lambda ex: {
    **ex,
    "color_count": len({token[1] for token in ex["sequence"].split() if len(token) > 1}),
    "shape_count": len({token[0] for token in ex["sequence"].split() if token})
})

# Print a summary of the loaded dataset splits
print("Dataset splits:", list(spr_dataset.keys()))
print("Example from train split:", spr_dataset["train"][0])