import datasets

# Load the SPR_BENCH dataset from local CSV files using HuggingFace's datasets library
data_files = {
    "train": "SPR_BENCH/train.csv",
    "validation": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = datasets.load_dataset("csv", data_files=data_files)

# Display the loaded dataset structure to confirm proper loading
print(dataset)