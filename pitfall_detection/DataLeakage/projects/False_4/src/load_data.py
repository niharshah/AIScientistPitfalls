from datasets import load_dataset

# Load datasets using the HuggingFace 'datasets' library
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
datasets = load_dataset('csv', data_files=data_files)

# Clean data by removing any rows with missing 'sequence' values
train_clean = datasets["train"].filter(lambda x: x['sequence'] is not None)
dev_clean = datasets["dev"].filter(lambda x: x['sequence'] is not None)
test_clean = datasets["test"].filter(lambda x: x['sequence'] is not None)

# Verify the number of rows after cleaning to ensure no missing values
print(f"Number of rows after cleaning - Train: {train_clean.num_rows}, Dev: {dev_clean.num_rows}, Test: {test_clean.num_rows}")