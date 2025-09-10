from datasets import load_dataset

# Load the SPR_BENCH dataset using local CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "validation": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
ds = load_dataset("csv", data_files=data_files)

# Preprocess each example by splitting the 'sequence' string into a list of tokens and converting label to integer.
ds = ds.map(lambda ex: {"tokens": ex["sequence"].strip().split(), "label": int(ex["label"])})
print(ds)