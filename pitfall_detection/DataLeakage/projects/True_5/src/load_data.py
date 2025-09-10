import random
import pandas as pd

# Function to generate a symbolic sequence
def generate_sequence(length):
    symbols = ['▲', '■', '●', '◆']
    colors = ['r', 'g', 'b', 'y']
    return ' '.join(random.choice(symbols) + random.choice(colors) for _ in range(length))

# Generate synthetic data
num_sequences = 30000
sequence_lengths = [5, 10, 15, 20]
synthetic_data = []

for length in sequence_lengths:
    for _ in range(num_sequences // len(sequence_lengths)):
        synthetic_data.append(generate_sequence(length))

# Function to apply labeling rules
def apply_rules(sequence):
    tokens = sequence.split()
    rule_1 = all(token[0] == tokens[0][0] for token in tokens)
    rule_2 = tokens[0][-1] == tokens[-1][-1]
    rule_3 = all(tokens[i][-1] != tokens[i + 1][-1] for i in range(len(tokens) - 1))
    return 1 if rule_1 or rule_2 or rule_3 else 0

# Generate labels for synthetic data
updated_labels = [apply_rules(seq) for seq in synthetic_data]

# Create a DataFrame for synthetic data
synthetic_df = pd.DataFrame({'id': [f'SYN_{i}' for i in range(len(synthetic_data))],
                             'sequence': synthetic_data,
                             'label': updated_labels})

# Load existing datasets from local CSV files
train_data = pd.read_csv('SPR_BENCH/train.csv')
dev_data = pd.read_csv('SPR_BENCH/dev.csv')
test_data = pd.read_csv('SPR_BENCH/test.csv')

# Combine synthetic data with the existing datasets
train_data_extended = pd.concat([train_data, synthetic_df.iloc[:20000]], ignore_index=True)
dev_data_extended = pd.concat([dev_data, synthetic_df.iloc[20000:25000]], ignore_index=True)
test_data_extended = pd.concat([test_data, synthetic_df.iloc[25000:30000]], ignore_index=True)

# Print sizes of the extended datasets
print("Extended Train Data Size:", train_data_extended.shape)
print("Extended Dev Data Size:", dev_data_extended.shape)
print("Extended Test Data Size:", test_data_extended.shape)