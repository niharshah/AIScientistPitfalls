from datasets import load_dataset

# Load the SPR_BENCH benchmark from local CSV files
data_files = {
    'train': 'SPR_BENCH/train.csv',
    'dev': 'SPR_BENCH/dev.csv',
    'test': 'SPR_BENCH/test.csv'
}
dataset = load_dataset('csv', data_files=data_files, delimiter=',')

# Simple data preparation: add a new column that splits the sequence into tokens.
dataset = dataset.map(lambda row: {'tokens': row['sequence'].split()})

print(dataset)