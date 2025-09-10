import pandas as pd
from datasets import load_dataset

# Load the local SPR_BENCH dataset using HuggingFace's datasets library
dataset = load_dataset('csv', data_files={'train': 'SPR_BENCH/train.csv'})

# Convert the train dataset to a Pandas DataFrame for easier manipulation
train_data = dataset['train'].to_pandas()

# Initialize an empty list to store graph representations of sequences
graph_data = []

# Iterate over each row in the train dataset
for index, row in train_data.iterrows():
    sequence = row['sequence'].split()
    nodes = list(set(sequence))
    edges = [(sequence[i], sequence[i+1]) for i in range(len(sequence)-1)]
    graph_data.append({'id': row['id'], 'nodes': nodes, 'edges': edges, 'label': row['label']})

# Convert list of graph data to a DataFrame
graph_df = pd.DataFrame(graph_data)

# Save the processed graph data into a CSV file
graph_df.to_csv('SPR_BENCH/graph_processed_train.csv', index=False)

# Print the first few rows to verify correctness
print("First few rows of the graph-processed DataFrame:")
print(graph_df.head())