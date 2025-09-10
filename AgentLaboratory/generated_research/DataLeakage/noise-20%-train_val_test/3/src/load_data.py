from datasets import load_dataset

# Define the path to the local CSV files for the SPR_BENCH benchmark
data_files = {
    'train': 'SPR_BENCH/train.csv',
    'dev': 'SPR_BENCH/dev.csv',
    'test': 'SPR_BENCH/test.csv'
}

# Load the datasets using Hugging Face `datasets` library
datasets = load_dataset('csv', data_files=data_files, data_dir='.')

# Overview of datasets to ensure they are loaded correctly
print("Training Dataset Details:")
print(datasets['train'])

print("\nDevelopment Dataset Details:")
print(datasets['dev'])

print("\nTest Dataset Details:")
print(datasets['test'])