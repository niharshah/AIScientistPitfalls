from datasets import load_dataset

# Load the SPR_BENCH dataset from local CSV files using HuggingFace datasets.
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

print("Dataset splits:")
for split, data in dataset.items():
    print(f"{split} split has {len(data)} samples")