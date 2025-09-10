from datasets import Dataset, DatasetDict
import pandas as pd

# Load the datasets from local CSV files
train_df = pd.read_csv('SPR_BENCH/train.csv', encoding='utf-8')
dev_df = pd.read_csv('SPR_BENCH/dev.csv', encoding='utf-8')
test_df = pd.read_csv('SPR_BENCH/test.csv', encoding='utf-8')

# Convert each DataFrame into a HuggingFace Dataset
train_data = Dataset.from_pandas(train_df)
dev_data = Dataset.from_pandas(dev_df)
test_data = Dataset.from_pandas(test_df)

# Create a DatasetDict for easy management
dataset = DatasetDict({
    'train': train_data,
    'dev': dev_data,
    'test': test_data
})

# Display basic information about the datasets
print(dataset)