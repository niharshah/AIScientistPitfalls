from datasets import load_dataset

# Load the SPR_BENCH dataset using the load_dataset function from HuggingFace
dataset = load_dataset("SPR_BENCH")

# Print sample data from each split to verify correct loading
print("Sample from Train set:\n", dataset["train"][0])
print("\nSample from Validation set:\n", dataset["validation"][0])
print("\nSample from Test set:\n", dataset["test"][0])