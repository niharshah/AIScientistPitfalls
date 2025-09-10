from datasets import load_dataset

# Load the SPR_BENCH dataset using HuggingFace datasets library
dataset = load_dataset('csv', data_files={
    'train': 'SPR_BENCH/train.csv',
    'dev': 'SPR_BENCH/dev.csv',
    'test': 'SPR_BENCH/test.csv'
})

# Display dataset information and the first few samples from each split
print("Dataset information:")
print(dataset)

print("\nFirst few rows of the training dataset:")
print(dataset['train'][0:5])

print("\nFirst few rows of the development dataset:")
print(dataset['dev'][0:5])

print("\nFirst few rows of the test dataset:")
print(dataset['test'][0:5])