from datasets import load_dataset
import os

# Base directory where all benchmark datasets are stored.
base_dir = "SPR_BENCH"

# List of four local HuggingFace dataset names to be utilized.
local_datasets = ["SFRFG", "IJSJF", "GURSG", "TSHUY"]

# Dictionary to hold the loaded datasets.
datasets_loaded = {}

# Loop through each dataset directory and load train, dev, and test splits.
for ds_name in local_datasets:
    ds_path = os.path.join(base_dir, ds_name)
    datasets_loaded[ds_name] = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(ds_path, "train.csv"),
            "dev": os.path.join(ds_path, "dev.csv"),
            "test": os.path.join(ds_path, "test.csv")
        },
        delimiter=","
    )
    print(f"Loaded {ds_name} dataset:")
    print(datasets_loaded[ds_name])
    
# The code above reads four local datasets and prints a summary of each dataset loaded.