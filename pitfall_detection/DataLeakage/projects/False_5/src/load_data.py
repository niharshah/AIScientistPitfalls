from datasets import load_dataset

# Load the CSV dataset using the load_dataset function from HuggingFace's datasets library
dataset = load_dataset('csv', data_files={'train': 'SPR_BENCH/train.csv', 
                                          'dev': 'SPR_BENCH/dev.csv', 
                                          'test': 'SPR_BENCH/test.csv'})

# Function to create symbol to integer mapping from the entire dataset
def create_symbol_mapping(dataset):
    symbol_set = set()
    for split in dataset:
        for seq in dataset[split]['sequence']:
            symbol_set.update(seq.split())
    return {symbol: idx for idx, symbol in enumerate(sorted(symbol_set))}

# Creating the symbol-to-number mapping
symbol_to_num = create_symbol_mapping(dataset)
print("\nSymbol-to-Number Mapping:")
print(symbol_to_num)

# Function to encode each dataset split using mapping
def encode_and_update_dataset(dataset, symbol_to_num):
    for split in dataset.keys():
        dataset[split] = dataset[split].map(lambda row: {'encoded_sequence': [symbol_to_num[symbol] for symbol in row['sequence'].split()]})
    return dataset

# Encoding the sequence data
encoded_dataset = encode_and_update_dataset(dataset, symbol_to_num)

# Print the encoded train dataset to verify
print("\nEncoded Train Dataset Example:")
print(encoded_dataset['train'][:3])