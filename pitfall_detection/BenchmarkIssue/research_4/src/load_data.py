from datasets import load_dataset

# Define the names of 4 local HuggingFace datasets (benchmarks)
dataset_names = ["EWERV", "URCJF", "PHRTV", "IJSJF"]

# Dictionary to store the datasets after loading
data_dict = {}

# For each dataset, load the train, dev, and test splits from CSV files
for name in dataset_names:
    # Define the file paths for each split
    data_files = {
        "train": f"SPR_BENCH/{name}/train.csv",
        "dev": f"SPR_BENCH/{name}/dev.csv",
        "test": f"SPR_BENCH/{name}/test.csv"
    }
    # Load the dataset using the HuggingFace datasets library
    dataset = load_dataset("csv", data_files=data_files, delimiter=",")
    data_dict[name] = dataset

# Print out the number of instances per split for each dataset
for name, dataset in data_dict.items():
    counts = {split: len(dataset[split]) for split in dataset.keys()}
    print(f"Dataset {name}: {counts}")