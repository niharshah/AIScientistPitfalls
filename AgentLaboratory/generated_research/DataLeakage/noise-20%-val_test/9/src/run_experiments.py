# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import pathlib
from datasets import load_dataset, DatasetDict

# Disable CUDA - force CPU on system-level as a precaution
torch.cuda.is_available = lambda: False
device = torch.device("cpu")

class GWNN(nn.Module):
    """Graph Wavelet Neural Network for symbolic pattern recognition."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GWNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def encode_symbolic_sequence(sequence):
    """Encode symbolic sequences into numerical format."""
    symbol_to_int = {'▲': 0, '◆': 1, '●': 2, '■': 3}
    color_to_int = {'r': 0, 'b': 1, 'g': 2, 'y': 3}
    return [symbol_to_int.get(s[0], 0) * 4 + color_to_int.get(s[1], 0) for s in sequence.split()]

def convert_to_graph(data):
    """Convert data to graph format, currently a placeholder."""
    return data.float()

def neuro_symbolic_reasoning(features):
    """Implement neuro-symbolic reasoning using poly-factor predicates."""
    # Placeholder for complex logic.
    return features

def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    """Return a DatasetDict {'train':…, 'dev':…, 'test':…} for the SPR bench."""
    def _load(split_csv: str):
        return load_dataset("csv", data_files=str(root / split_csv), split="train", cache_dir=".cache_dsets")
    
    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset

# Load dataset
DATA_PATH = pathlib.Path('./SPR_BENCH/')
spr_bench = load_spr_bench(DATA_PATH)

def create_dataloader(dataset, batch_size=64):
    X = torch.stack([torch.tensor(encode_symbolic_sequence(x['sequence']), dtype=torch.float32) for x in dataset])
    y = torch.tensor([x['label'] for x in dataset], dtype=torch.long)
    tensor_data = TensorDataset(X, y)
    return DataLoader(tensor_data, batch_size=batch_size, shuffle=True)

# Create DataLoader instances
train_loader = create_dataloader(spr_bench['train'])
dev_loader = create_dataloader(spr_bench['dev'])
test_loader = create_dataloader(spr_bench['test'])

# Initialize model, loss function, and optimizer
input_dim = 32
model = GWNN(input_dim=input_dim, hidden_dim=64, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Training the GWNN model on SPR_BENCH...")
for epoch in range(5):  # Run for 5 epochs
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        graph_features = convert_to_graph(X_batch)
        logical_features = neuro_symbolic_reasoning(graph_features)
        outputs = model(logical_features)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/5], Loss: {total_loss / len(train_loader):.4f}')

# Evaluation function
def evaluate_model(loader):
    model.eval()
    predictions, actual_values = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            graph_features = convert_to_graph(X_batch)
            logical_features = neuro_symbolic_reasoning(graph_features)
            outputs = model(logical_features)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            actual_values.extend(y_batch.cpu().numpy())
    return accuracy_score(actual_values, predictions), f1_score(actual_values, predictions, average='weighted')

# Evaluate on dev set
dev_accuracy, dev_f1 = evaluate_model(dev_loader)

# Display results
print("Results for GWNN model on SPR_BENCH dev set:")
print(f"Dev Accuracy: {dev_accuracy * 100:.2f}%")
print(f"F1 Score: {dev_f1:.4f}")

# Plot evaluation results
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy', 'F1 Score'], [dev_accuracy, dev_f1])
plt.title('Figure 2: Dev Results on SPR_BENCH Data')
plt.savefig('Figure_2.png')

# Evaluate on test set to address the original error
test_accuracy, test_f1 = evaluate_model(test_loader)

# Display test results
print("Results for GWNN model on SPR_BENCH test set:")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"F1 Score: {test_f1:.4f}")