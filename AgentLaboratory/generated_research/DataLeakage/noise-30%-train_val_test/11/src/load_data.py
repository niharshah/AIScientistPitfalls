from datasets import load_dataset

# Load the SPR_BENCH dataset using HuggingFace datasets library
dataset = load_dataset('csv', data_files={'train': 'SPR_BENCH/train.csv', 
                                          'dev': 'SPR_BENCH/dev.csv', 
                                          'test': 'SPR_BENCH/test.csv'})

# Display the loaded dataset structure
print("Dataset structure:")
print(dataset)