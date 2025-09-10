import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Ensure torch only uses CPU
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

# Load datasets from CSV files
train_df = pd.read_csv('SPR_BENCH/train.csv', encoding='utf-8')
dev_df = pd.read_csv('SPR_BENCH/dev.csv', encoding='utf-8')
test_df = pd.read_csv('SPR_BENCH/test.csv', encoding='utf-8')

# Define function to construct graph data from sequences
def construct_graph(sequence, label):
    G = nx.DiGraph()
    for i, token in enumerate(sequence):
        G.add_node(i, token=token)
    for i in range(len(sequence) - 1):
        G.add_edge(i, i + 1)
    x = torch.tensor([[ord(c.lower()) - ord('a') + 1] for c in sequence], dtype=torch.float)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# Convert datasets into graph structures
train_graphs = [construct_graph(seq, label) for seq, label in zip(train_df['sequence'], train_df['label'])]
dev_graphs = [construct_graph(seq, label) for seq, label in zip(dev_df['sequence'], dev_df['label'])]
test_graphs = [construct_graph(seq, label) for seq, label in zip(test_df['sequence'], test_df['label'])]

# Setup DataLoader
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=32, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# Define the ANSGN model
class ANSGN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANSGN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = x.unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        x = global_mean_pool(x, batch)
        return self.fc(x)

# Initialize model, loss function, and optimizer
input_dim = 1
output_dim = len(train_df['label'].unique())
model = ANSGN(input_dim=input_dim, hidden_dim=32, output_dim=output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop function with concise code and CPU usage
print("Training ANSGN model on CPU to evaluate its effectiveness on symbolic sequence data.")
for epoch in range(3):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x.to(device), batch.edge_index.to(device), batch.batch)
        loss = criterion(out, batch.y.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {epoch_loss / len(train_loader):.4f}")

# Evaluate the model
print("Evaluating model to validate symbolic sequence recognition capabilities.")
model.eval()
correct, total = 0, 0
for batch in test_loader:
    out = model(batch.x.to(device), batch.edge_index.to(device), batch.batch)
    preds = out.argmax(dim=1)
    correct += (preds == batch.y.to(device)).sum().item()
    total += batch.y.size(0)

accuracy = correct / total * 100
print(f"Test set model accuracy: {accuracy:.2f}%")

# Plot and save training loss figure
for epoch in range(3):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x.to(device), batch.edge_index.to(device), batch.batch)
        loss = criterion(out, batch.y.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {epoch_loss / len(train_loader):.4f}")
plt.ylabel('Loss')
plt.title('Figure_1: Training Loss over Epochs')
plt.legend()
plt.savefig('Figure_1.png')

# Plot and save distribution of predictions figure
plt.figure()
predictions = []
for batch in test_loader:
    out = model(batch.x.to(device), batch.edge_index.to(device), batch.batch)
    predictions.extend(out.argmax(dim=1).tolist())

plt.hist(predictions, bins=output_dim, align='left', rwidth=0.8)
plt.xlabel('Predicted Labels')
plt.ylabel('Frequency')
plt.title('Figure_2: Distribution of Predicted Labels on Test Set')
plt.savefig('Figure_2.png')