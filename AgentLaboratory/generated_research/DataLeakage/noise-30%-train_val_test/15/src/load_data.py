from datasets import load_dataset

# Load the train and test splits separately from the SPR_BENCH dataset
train_data = load_dataset("SPR_BENCH", split='train')
test_data = load_dataset("SPR_BENCH", split='test')

# Print the first entry from the train split to verify loading
print("Train split sample:")
print(train_data[0])

# Print the first entry from the test split to verify loading
print("\nTest split sample:")
print(test_data[0])