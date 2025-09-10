from datasets import load_dataset

# Load the datasets from local files assuming they are stored in 'SPR_BENCH/'
dataset = load_dataset('csv', data_files={'train': 'SPR_BENCH/train.csv', 'dev': 'SPR_BENCH/dev.csv', 'test': 'SPR_BENCH/test.csv'})

# Print the first few rows from each dataset to verify they are loaded correctly
print("Training Dataset Sample:")
print(dataset['train'][:5])

print("\nDevelopment Dataset Sample:")
print(dataset['dev'][:5])

print("\nTest Dataset Sample:")
print(dataset['test'][:5])