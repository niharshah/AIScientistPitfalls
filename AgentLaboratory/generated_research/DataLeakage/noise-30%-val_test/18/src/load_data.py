import pandas as pd
from datasets import Dataset

# Load the SPR_BENCH dataset from local CSV files
train_data = pd.read_csv('SPR_BENCH/train.csv')
dev_data = pd.read_csv('SPR_BENCH/dev.csv')
test_data = pd.read_csv('SPR_BENCH/test.csv')

# Convert pandas DataFrame to HuggingFace Dataset format
train_dataset = Dataset.from_pandas(train_data)
dev_dataset = Dataset.from_pandas(dev_data)
test_dataset = Dataset.from_pandas(test_data)

# Display the information to verify the datasets
print("HuggingFace Datasets:")
print("Train Dataset size:", len(train_dataset))
print("Dev Dataset size:", len(dev_dataset))
print("Test Dataset size:", len(test_dataset))

# Print sample entries from the train dataset to verify data format
print("Sample entries from train dataset:")
print(train_dataset[0])
print(train_dataset[1])
print(train_dataset[2])