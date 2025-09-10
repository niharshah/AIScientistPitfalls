from datasets import load_dataset

data_files = {
    "train": "SPR_BENCH/train.csv",
    "validation": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}

dataset = load_dataset("csv", data_files=data_files, delimiter=",")

print("Train dataset sample:", dataset["train"][0])
print("Validation dataset sample:", dataset["validation"][0])
print("Test dataset sample:", dataset["test"][0])