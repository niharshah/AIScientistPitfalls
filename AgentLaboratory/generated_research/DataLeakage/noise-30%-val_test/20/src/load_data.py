from datasets import load_dataset

# Load the SPR_BENCH dataset
dataset = load_dataset("SPR_BENCH")

# Access train and test splits only, since dev doesn't exist
train_data = dataset['train']
test_data = dataset['test']

# Verify the structure of the dataset
print("Train data sample:", train_data[0])
print("Test data sample:", test_data[0])