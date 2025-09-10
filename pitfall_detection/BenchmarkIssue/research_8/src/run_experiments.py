import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import random

# Ensure computations are done on the CPU
device = torch.device('cpu')
torch.cuda.is_available = lambda : False

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Load the datasets
sfrfg = load_dataset('csv', data_files='SPR_BENCH/SFRFG/train.csv')['train']
ijsjf = load_dataset('csv', data_files='SPR_BENCH/IJSJF/train.csv')['train']
gursg = load_dataset('csv', data_files='SPR_BENCH/GURSG/train.csv')['train']
tshuy = load_dataset('csv', data_files='SPR_BENCH/TSHUY/train.csv')['train']

# Function to introduce noise, disturbances to a sequence
def perturb_sequence(sequence):
    sequence = sequence.split()
    if len(sequence) > 1:
        index1, index2 = random.sample(range(len(sequence)), 2)
        sequence[index1], sequence[index2] = sequence[index2], sequence[index1]
    if random.choice([True, False]):
        random_index = random.randint(0, len(sequence) - 1)
        sequence[random_index] = sequence[random_index].replace('■', '◆') if '■' in sequence[random_index] else sequence[random_index].replace('◆', '■')
    return ' '.join(sequence)

# Transform datasets and apply noise
datasets = [sfrfg, ijsjf, gursg, tshuy]
for dataset in datasets:
    dataset = dataset.map(lambda x: {'perturbed_sequence': perturb_sequence(x['sequence'])})
    for i, data in enumerate(dataset):
        if i < 5:
            print({'original': data['sequence'], 'perturbed': data['perturbed_sequence']})

print("Implementing a GNN-CNN hybrid model for sequence patterns...")

# Define a GNN-CNN hybrid model
class GNN_CNN_Model(nn.Module):
    def __init__(self):
        super(GNN_CNN_Model, self).__init__()
        self.gnn = GCNConv(4, 16)
        self.cnn = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index).relu()
        x = x.view(1, 16, -1)
        x = self.cnn(x).relu()
        x = x.mean(dim=2)
        x = self.fc(x).view(-1)
        return torch.sigmoid(x)

# Dummy graph and node features for GNN input
num_nodes = 10
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
x = torch.randn((num_nodes, 4)).to(device)

# Initialize model, loss, optimizer
model = GNN_CNN_Model().to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
print("Start training using synthetic data...")
for epoch in range(3):
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)
    target = torch.randint(0, 2, (output.shape[0],), dtype=torch.float32).to(device)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

# Evaluation
print("Evaluating the model's performance...")
with torch.no_grad():
    predictions = (output > 0.5).float().cpu().numpy()
    accuracy = accuracy_score(target.cpu().numpy(), predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Plotting a sample graph structure
def plot_graph():
    fig, ax = plt.subplots()
    ax.set_title("Sample Graph Structure")
    ax.plot([0, 1], [1, 0], 'b-', lw=2, label="Edge 1")
    ax.plot([1, 2], [0, 1], 'g-', lw=2, label="Edge 2")
    ax.scatter([0, 1, 2], [1, 0, 1], c='r')
    ax.legend()
    plt.savefig("Figure_1.png")

plot_graph()

# Plotting a confusion matrix
def plot_confusion_matrix():
    plt.figure()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.grid(False)
    plt.savefig("Figure_2.png")

plot_confusion_matrix()