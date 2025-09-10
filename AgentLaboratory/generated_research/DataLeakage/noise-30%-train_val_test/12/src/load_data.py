import pandas as pd

# Reload the datasets to ensure we have them for processing
train_data = pd.read_csv('SPR_BENCH/train.csv')
dev_data = pd.read_csv('SPR_BENCH/dev.csv')
test_data = pd.read_csv('SPR_BENCH/test.csv')

# Define a tokenizer to split the sequence into tokens based on spaces
def tokenize(sequence):
    return sequence.split()

# Apply tokenization to all datasets
train_data['tokens'] = train_data['sequence'].map(tokenize)
dev_data['tokens'] = dev_data['sequence'].map(tokenize)
test_data['tokens'] = test_data['sequence'].map(tokenize)

# Display the tokenized samples to ensure the process worked
print("\nSample Tokenization - Train Data:")
print(train_data.head())

print("\nSample Tokenization - Dev Data:")
print(dev_data.head())

print("\nSample Tokenization - Test Data:")
print(test_data.head())