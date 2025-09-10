import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from datasets import load_dataset

# Ensure using CPU for all operations
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

# Load datasets
data_files = {
    'train': 'SPR_BENCH/train.csv',
    'dev': 'SPR_BENCH/dev.csv',
    'test': 'SPR_BENCH/test.csv'
}
datasets = load_dataset('csv', data_files=data_files, data_dir='.')

# Function to create synthetic dataset
def create_synthetic_dataset(dataset):
    data_list = []
    for row in dataset:
        sequence = row['sequence'].split()
        if not sequence:
            continue
        num_tokens = len(sequence)
        token_embeddings = torch.tensor(np.random.rand(num_tokens, 64), dtype=torch.float32)
        edge_index = torch.tensor([[i, i + 1] for i in range(num_tokens - 1)], dtype=torch.long).t().contiguous()
        try:
            label = torch.tensor([float(row['label'])], dtype=torch.float32)
        except ValueError:
            print(f"Warning: Could not convert label '{row['label']}' to float. Skipping this entry.")
            continue
        graph = Data(x=token_embeddings, edge_index=edge_index, y=label)
        data_list.append(graph)
    return data_list

# Define Graph-Based SPR Model
class ProbabilisticGraphModel(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ProbabilisticGraphModel, self).__init__()
        self.gcn1 = GCNConv(embedding_dim, 128)
        self.gcn2 = GCNConv(128, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, data):
        x = self.gcn1(data.x, data.edge_index).relu()
        x = self.gcn2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.fc(x).squeeze()

# Prepare datasets
train_data = create_synthetic_dataset(datasets['train'])
dev_data = create_synthetic_dataset(datasets['dev'])
test_data = create_synthetic_dataset(datasets['test'])

# Initialize model, optimizer, and loss
model = ProbabilisticGraphModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Setup DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Training function
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds = torch.sigmoid(out) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return accuracy, f1

# Training and evaluation
train_acc_list, dev_acc_list = [], []
train_f1_list, dev_f1_list = [], []

for epoch in range(3):  # Reduced number of epochs to avoid timeout
    print(f"\nEpoch {epoch}: Training and evaluation.")
    train_one_epoch(model, train_loader, optimizer, criterion)
    train_acc, train_f1 = evaluate(model, train_loader)
    dev_acc, dev_f1 = evaluate(model, dev_loader)
    train_acc_list.append(train_acc)
    dev_acc_list.append(dev_acc)
    train_f1_list.append(train_f1)
    dev_f1_list.append(dev_f1)
    print(f"Epoch {epoch}: Train Acc {train_acc:.4f}, Train F1 {train_f1:.4f}; Dev Acc {dev_acc:.4f}, Dev F1 {dev_f1:.4f}")

# Final evaluation on test data
print("\nFinal evaluation on test data:")
test_accuracy, test_f1 = evaluate(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")

# Plot and save results
def plot_results(train_results, dev_results, y_label, figure_name):
    plt.figure()
    plt.plot(train_results, label='Train')
    plt.plot(dev_results, label='Dev')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(f'Training and Development {y_label}')
    plt.legend()
    plt.savefig(figure_name)

# Plot and save accuracy
plot_results(train_acc_list, dev_acc_list, 'Accuracy', 'Figure_1.png')
print("Figure 1 shows accuracy over epochs for train and dev sets.")

# Plot and save F1 score
plot_results(train_f1_list, dev_f1_list, 'F1 Score', 'Figure_2.png')
print("Figure 2 shows F1 Score over epochs for train and dev sets.")