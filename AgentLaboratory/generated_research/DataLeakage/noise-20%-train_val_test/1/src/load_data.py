from datasets import load_dataset

# Load the SPR_BENCH CSV files from the local directory into a single dataset dictionary
data_files = {
    "train": "./SPR_BENCH/train.csv",
    "dev": "./SPR_BENCH/dev.csv",
    "test": "./SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# Print basic statistics of the loaded dataset for verification
print("Dataset keys:", list(dataset.keys()))
print("Train split sample:")
print(dataset["train"][0])
print("Dev split sample:")
print(dataset["dev"][0])
print("Test split sample:")
print(dataset["test"][0])