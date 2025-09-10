import datasets

# List of 4 local HuggingFace datasets directories (benchmark names)
benchmarks = ["SFRFG", "IJSJF", "GURSG", "TEXHE"]

# Dictionary to store loaded datasets
loaded_datasets = {}

# Loop over each benchmark and load CSV splits using HuggingFace datasets
for b in benchmarks:
    folder = f"SPR_BENCH/{b}"
    data_files = {
        "train": f"{folder}/train.csv",
        "dev": f"{folder}/dev.csv",
        "test": f"{folder}/test.csv"
    }
    ds = datasets.load_dataset("csv", data_files=data_files)
    loaded_datasets[b] = ds
    print(f"Benchmark {b} loaded:")
    print("  Train size:", len(ds["train"]))
    print("  Dev size:", len(ds["dev"]))
    print("  Test size:", len(ds["test"]))

# Display one example from the first benchmark (SFRFG) for verification.
print("\nExample from SFRFG train split:")
print(loaded_datasets["SFRFG"]["train"][0])