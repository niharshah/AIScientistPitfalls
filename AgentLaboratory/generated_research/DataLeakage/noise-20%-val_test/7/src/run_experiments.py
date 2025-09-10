import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# Force usage of CPU
torch.cuda.is_available = lambda : False
device = torch.device('cpu')

# Load the dataset
dataset = load_dataset('csv', data_files={'train': './SPR_BENCH/train.csv', 'dev': './SPR_BENCH/dev.csv', 'test': './SPR_BENCH/test.csv'})

print("Training Dataset Sample:")
print(dataset['train'][:5])

# Encode symbolic sequences into graph data
def encode_sequences(sequences, labels):
    data_list = []
    for seq, label in zip(sequences, labels):
        node_features = []
        edge_index = []
        symbols = seq.split()
        for i, sym in enumerate(symbols):
            value = float(ord(sym[0]) % 10) / 10.0  # Basic encoding of feature
            node_features.append([value])
            if i > 0:
                edge_index.extend([[i - 1, i], [i, i - 1]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(node_features, dtype=torch.float)
        data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
        data_list.append(data)
    return data_list

# Define the GNN model with attention
class SPRModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SPRModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=2, concat=True)
        self.conv3 = GATConv(hidden_channels * 2, out_channels, heads=1, concat=False)
        self.fc = nn.Linear(out_channels, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)

# Prepare the datasets
train_sequences = dataset['train']['sequence']
dev_sequences = dataset['dev']['sequence']
test_sequences = dataset['test']['sequence']

train_labels = torch.tensor(dataset['train']['label'], dtype=torch.long)
dev_labels = torch.tensor(dataset['dev']['label'], dtype=torch.long)
test_labels = torch.tensor(dataset['test']['label'], dtype=torch.long)

train_data = encode_sequences(train_sequences, train_labels)
dev_data = encode_sequences(dev_sequences, dev_labels)
test_data = encode_sequences(test_sequences, test_labels)

# Initialize model, optimizer, and loss function
model = SPRModel(in_channels=1, hidden_channels=32, out_channels=16).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Load data into DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Training loop
num_epochs = 5
loss_values = []

print("Starting training process...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}")
    loss_values.append(avg_loss)

# Validation
print("Evaluating on validation set...")
model.eval()
correct = 0
with torch.no_grad():
    for data in dev_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
val_acc = correct / len(dev_data)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Test Evaluation
print("Evaluating on test set...")
correct = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
test_acc = correct / len(test_data)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Generate and save figures showing results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, num_epochs + 1), loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.savefig('Figure_1.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(['Validation', 'Test'], [val_acc, test_acc])
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Validation and Test Accuracy')
plt.savefig('Figure_2.png')
plt.show()