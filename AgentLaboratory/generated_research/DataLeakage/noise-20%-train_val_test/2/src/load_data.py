from datasets import load_dataset

# Load the SPR_BENCH dataset from HuggingFace
dataset = load_dataset('SPR_BENCH')

# Access train, validation, and test datasets
train_set = dataset['train']
validation_set = dataset['validation']
test_set = dataset['test']

# Display the first few entries of each dataset to verify the data
print("Train Dataset Sample:")
print(train_set[:5], "\n")

print("Validation Dataset Sample:")
print(validation_set[:5], "\n")

print("Test Dataset Sample:")
print(test_set[:5])