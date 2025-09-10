import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
import numpy as np

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

# Function to convert sequences to node features
def sequence_to_features(sequence):
    shape_map = {'▲': 0, '■': 1, '●': 2, '◆': 3}
    color_map = {'r': 0, 'b': 1, 'g': 2, 'y': 3}
    tokens = sequence.split()
    features = np.zeros((len(tokens), 2), dtype=int)
    for i, token in enumerate(tokens):
        shape_feature = shape_map.get(token[0], 0)
        color_feature = color_map.get(token[1], 0) if len(token) > 1 else 0
        features[i] = [shape_feature, color_feature]
    return features.flatten()

# Preprocess datasets
train_features = np.array([sequence_to_features(seq['sequence']) for seq in train_data if 'sequence' in seq])
dev_features = np.array([sequence_to_features(seq['sequence']) for seq in dev_data if 'sequence' in seq])
test_features = np.array([sequence_to_features(seq['sequence']) for seq in test_data if 'sequence' in seq])

# Extract labels
train_labels = [seq['label'] for seq in train_data if 'label' in seq]
dev_labels = [seq['label'] for seq in dev_data if 'label' in seq]
test_labels = [seq['label'] for seq in test_data if 'label' in seq]

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
dev_labels_encoded = label_encoder.transform(dev_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Reshape features to 2D arrays for GaussianNB
train_features = train_features.reshape((len(train_labels), -1))
dev_features = dev_features.reshape((len(dev_labels), -1))
test_features = test_features.reshape((len(test_labels), -1))

# Train a Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(train_features, train_labels_encoded)

# Evaluate the model on dev data
print("Evaluating model performance on the validation set:")
dev_preds = gnb.predict(dev_features)
dev_accuracy = accuracy_score(dev_labels_encoded, dev_preds)
print(f"Validation Accuracy: {dev_accuracy}")

# Evaluate the model on test data
print("Evaluating model performance on the test set:")
test_preds = gnb.predict(test_features)
test_accuracy = accuracy_score(test_labels_encoded, test_preds)
print(f"Test Accuracy: {test_accuracy}")

# Visualize sample graphs from the test set
for idx in range(min(len(test_data), 2)):  # Using range to avoid index errors
    G = nx.DiGraph()
    features = sequence_to_features(test_data[idx]['sequence']).reshape(-1, 2)
    for i, (shape, color) in enumerate(features):
        color_hex = '#' + ''.join([f"{int(63 * (3 - color)):02x}" for _ in range(3)])  # simple color mapping
        G.add_node(i, shape=shape, color=color_hex)
        if i > 0:
            G.add_edge(i-1, i)
    
    plt.figure(figsize=(6, 4))
    node_colors = [G.nodes[i]['color'] for i in G.nodes()]
    nx.draw(G, with_labels=True, node_color=node_colors, cmap=plt.get_cmap('viridis'))
    plt.title(f"Graph for Test Sequence {idx + 1}")
    plt.savefig(f"Figure_{idx + 1}.png")
    plt.close()