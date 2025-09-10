import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import re

# Ensure no CUDA is used
os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device('cpu')

# Load datasets
train_data = load_dataset("SPR_BENCH", split='train')
test_data = load_dataset("SPR_BENCH", split='test')

# Redesign process to properly handle sequences using regex to ignore non-symbol characters
def process_row(row):
    sequence = row['sequence']
    # Filter only symbol characters for accurate counting
    filtered_seq = re.sub(r'[^■●▲◆]', '', sequence)
    feature_vector = [filtered_seq.count(char) for char in ['■', '●', '▲', '◆']]
    # Placeholder parity and positional attributes
    feature_vector.extend([len(filtered_seq) % 2, len(filtered_seq)])  # Example for Parity and Position
    x = torch.tensor(feature_vector, dtype=torch.float32)
    y = torch.tensor(row['label'], dtype=torch.long)
    return x, y

# Dataset class to wrap our custom processing
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = [process_row(row) for row in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Define a simple MLP model with adjustments for more dimensions
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 64),  # We have 6 input features now
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.layers(x)

# Train the model
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for features, labels in loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluate model accuracy
def evaluate(model, loader):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            ys.extend(labels.numpy())
            preds.extend(predicted.numpy())
    return accuracy_score(ys, preds)

    total_loss = 0
train_dataset = SimpleDataset(train_data)
test_dataset = SimpleDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate model, loss, and optimizer
model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, criterion)
    train_losses.append(loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# Evaluate the model
print("Evaluating model performance:")
train_accuracy = evaluate(model, train_loader)
test_accuracy = evaluate(model, test_loader)

# Report accuracies
print("The following results indicate model's capability and generalization strength.")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot and save training loss trajectory
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig('Figure_1.png')

# Plot and save accuracy comparison
plt.figure()
plt.bar(['Train', 'Test'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy on Train and Test Sets')
plt.savefig('Figure_2.png')