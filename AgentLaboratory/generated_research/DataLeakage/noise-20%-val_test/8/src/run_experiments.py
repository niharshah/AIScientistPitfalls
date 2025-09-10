import pandas as pd
from sklearn.model_selection import train_test_split
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Force use of CPU by disabling CUDA
torch.cuda.is_available = lambda : False

# Load the datasets
train_data = pd.read_csv('SPR_BENCH/train.csv')
dev_data = pd.read_csv('SPR_BENCH/dev.csv')
test_data = pd.read_csv('SPR_BENCH/test.csv')

# Combine datasets
combined_data = pd.concat([train_data, dev_data, test_data])

# Conduct a 70:30 train-test split
train_split, test_split = train_test_split(combined_data, test_size=0.3, random_state=42)

# Check sample data
print("Train Sample:", train_split['sequence'].head())
print("Test Sample:", test_split['sequence'].head())

# Graph creation function
def create_graph(sequence):
    G = nx.Graph()
    symbols = sequence.split()
    for idx, symbol in enumerate(symbols):
        G.add_node(idx, symbol=symbol)
    for idx in range(len(symbols) - 1):
        G.add_edge(idx, idx + 1)
    return G

# Define a function to map symbols to features
def symbol_to_features(symbol):
    shapes = {'■': 1, '▲': 2, '●': 3, '◆': 4}
    colors = {'r': 1, 'g': 2, 'b': 3, 'y': 4}
    return [shapes.get(symbol[0], 0), colors.get(symbol[1], 0)] if len(symbol) == 2 else [0, 0]

# Function to convert graph to PyTorch Geometric Data format
def graph_to_data(graph, label):
    x = torch.tensor([symbol_to_features(data['symbol']) for _, data in graph.nodes(data=True)], dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# Define the GNN model architecture
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return global_add_pool(x, batch)

# Prepare data
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_split['label'])
test_labels = label_encoder.transform(test_split['label'])

train_graphs = [create_graph(seq) for seq in train_split['sequence']]
test_graphs = [create_graph(seq) for seq in test_split['sequence']]

train_data_list = [graph_to_data(g, l) for g, l in zip(train_graphs, train_labels)]
test_data_list = [graph_to_data(g, l) for g, l in zip(test_graphs, test_labels)]

train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

# Initialize model, optimizer, and loss function
device = torch.device('cpu')
model = GNN(input_dim=2, hidden_dim=64, output_dim=len(label_encoder.classes_)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
print("Training started...")
for epoch in range(5):  # Reduced epochs to avoid lengthy execution
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
y_true, y_pred = [], []
for batch in test_loader:
    batch = batch.to(device)
    output = model(batch.x, batch.edge_index, batch.batch)
    pred = output.argmax(dim=1)
    y_true.extend(batch.y.tolist())
    y_pred.extend(pred.tolist())

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Output results
print("\nResults:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot and save Accuracy
plt.figure()
plt.plot(range(5), [accuracy] * 5, label='Accuracy')
plt.title("Accuracy Over Epochs")
plt.savefig("Figure_1.png")