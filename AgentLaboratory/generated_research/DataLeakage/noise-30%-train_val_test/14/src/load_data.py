from datasets import load_dataset
import networkx as nx

# Load the dataset using HuggingFace Datasets
dataset = load_dataset('csv', data_files={
    'train': 'SPR_BENCH/train.csv',
    'dev': 'SPR_BENCH/dev.csv',
    'test': 'SPR_BENCH/test.csv'
})

# Assign the splits to variables
train_data = dataset['train']
dev_data = dataset['dev']
test_data = dataset['test']

# Take a sample sequence from train data
sequence_sample = train_data[0]['sequence']

# Initialize a graph
G = nx.DiGraph()

# Define mapping for shapes and colors
shape_map = {'▲': 'triangle', '■': 'square', '●': 'circle', '◆': 'diamond'}
color_map = {'r': 'red', 'b': 'blue', 'g': 'green', 'y': 'yellow'}

# Tokenize the sequence
tokens = sequence_sample.split()

# Add nodes and edges to the graph based on tokens
for idx, token in enumerate(tokens):
    shape = shape_map.get(token[0], 'unknown')
    color = color_map.get(token[1], 'none') if len(token) > 1 else 'none'
    node_id = f"node_{idx}"
    
    # Add the current node with attributes
    G.add_node(node_id, shape=shape, color=color)
    
    # Connect to the previous node if one exists
    if idx > 0:
        prev_node_id = f"node_{idx-1}"
        G.add_edge(prev_node_id, node_id)

# Print node details
print("Graph Nodes and Attributes:")
for node in G.nodes(data=True):
    print(f"{node}")

# Print edge details
print("\nGraph Edges:")
print(list(G.edges()))