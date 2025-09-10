import pathlib
from datasets import load_dataset, DatasetDict

# Use the relative path to the SPR_BENCH folder
spr_path = pathlib.Path("SPR_BENCH")

# Initialize an empty dictionary to hold available dataset splits.
spr_dataset = {}

# For each expected split, check if the CSV file exists and load it.
for split in ["train", "dev", "test"]:
    csv_file = spr_path / f"{split}.csv"
    if csv_file.exists():
        spr_dataset[split] = load_dataset("csv", data_files=str(csv_file), split="train", cache_dir=".cache_dsets")
        print(f"Loaded {split} split with {len(spr_dataset[split])} records.")
    else:
        print(f"WARNING: {csv_file.resolve()} not found. '{split}' split will be skipped.")

# Convert to a DatasetDict if any splits loaded.
if spr_dataset:
    spr_dataset = DatasetDict(spr_dataset)
    print("\nDataset splits available:", list(spr_dataset.keys()))
    if "train" in spr_dataset:
        example = spr_dataset["train"][0]
        print("\nFirst training example:")
        print("ID:", example.get("id", "N/A"))
        print("Sequence:", example.get("sequence", "N/A"))
        print("Label:", example.get("label", "N/A"))
        
        # Process the first training sample to count unique colors and shapes.
        tokens = example.get("sequence", "").split()
        shape_set = set()
        color_set = set()
        for token in tokens:
            if token:
                shape_set.add(token[0])
            if len(token) > 1:
                color_set.add(token[1])
        print("Unique Shapes Count:", len(shape_set))
        print("Unique Colors Count:", len(color_set))
else:
    print("No dataset splits were loaded. Please ensure the SPR_BENCH folder contains train.csv, dev.csv, and test.csv.")