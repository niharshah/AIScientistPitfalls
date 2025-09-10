from datasets import load_dataset

# Define the file paths for the local SPR_BENCH dataset CSVs
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}

# Load the dataset using HuggingFace's load_dataset function for CSV files
dataset = load_dataset("csv", data_files=data_files)

# Print the dataset summary to confirm correct loading
print(dataset)