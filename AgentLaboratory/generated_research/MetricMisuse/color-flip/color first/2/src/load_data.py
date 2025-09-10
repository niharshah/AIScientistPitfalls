from datasets import load_dataset

# Load the local SPR_BENCH dataset from CSV files
dataset = load_dataset('csv', data_files={
    'train': './SPR_BENCH/train.csv',
    'dev': './SPR_BENCH/dev.csv',
    'test': './SPR_BENCH/test.csv'
}, delimiter=',')

# Print a sample row from the training set and dataset sizes for quick inspection
print("Sample train instance:", dataset['train'][0])
print("Number of training instances:", len(dataset['train']))
print("Number of dev instances:", len(dataset['dev']))
print("Number of test instances:", len(dataset['test']))