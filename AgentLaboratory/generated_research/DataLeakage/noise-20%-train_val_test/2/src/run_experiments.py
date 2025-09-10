import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from datasets import load_dataset

# Force PyTorch to use CPU
device = torch.device('cpu')
torch.cuda.is_available = lambda: False  # Mock to prevent accidental CUDA use

# Load the SPR_BENCH dataset
dataset = load_dataset('csv', data_files={
    'train': './SPR_BENCH/train.csv',
    'dev': './SPR_BENCH/dev.csv',
    'test': './SPR_BENCH/test.csv'
})

train_set = dataset['train']
validation_set = dataset['dev']
test_set = dataset['test']

# Display samples of datasets
print("Train Dataset Sample:")
print(train_set[:2], "\n")

print("Validation Dataset Sample:")
print(validation_set[:2], "\n")

print("Test Dataset Sample:")
print(test_set[:2])

# Dataset class
class SymbolDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = torch.tensor([ord(char) for char in self.data[idx]['sequence'].replace(' ', '')[:32]], dtype=torch.float32)
        label = torch.tensor(self.data[idx]['label'], dtype=torch.long)
        return sequence, label

# Prepare datasets
train_data = SymbolDataset(train_set)
val_data = SymbolDataset(validation_set)
test_data = SymbolDataset(test_set)

# DataLoader
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)
test_loader = DataLoader(test_data, batch_size=8)

# Simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
input_dim = 32
hidden_dim = 64
output_dim = 2
model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)

# Optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for seq, lbl in loader:
        seq, lbl = seq.to(device), lbl.to(device)
        optimizer.zero_grad()
        out = model(seq)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for seq, lbl in loader:
            seq = seq.to(device)
            out = model(seq)
            predictions.extend(out.argmax(dim=1).tolist())
            labels.extend(lbl.tolist())
    return accuracy_score(labels, predictions)

# Training and Evaluation
num_epochs = 5
train_accuracies, val_accuracies = [], []

print("Training the model...")
for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, criterion)
    train_accuracy = evaluate(model, train_loader)
    val_accuracy = evaluate(model, val_loader)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Test accuracy
test_accuracy = evaluate(model, test_loader)
print("Final Test Accuracy:", test_accuracy)

# Plot accuracy
plt.figure()
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Validation Accuracy')
plt.savefig('Figure_1.png')

plt.figure()
plt.bar(['Test Accuracy'], [test_accuracy])
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.ylim(0, 1)
plt.savefig('Figure_2.png')