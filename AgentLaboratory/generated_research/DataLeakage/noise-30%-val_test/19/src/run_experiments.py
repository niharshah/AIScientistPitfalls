import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datasets import load_dataset

# Ensure CUDA is not used
torch.cuda.is_available = lambda : False

# Set device to CPU
device = torch.device("cpu")

# Load the SPR_BENCH dataset
dataset = load_dataset("csv", data_files={"train": "./SPR_BENCH/train.csv", "validation": "./SPR_BENCH/dev.csv", "test": "./SPR_BENCH/test.csv"})

# Verify dataset loading
print("Sample from Train set:\n", dataset["train"][0])
print("\nSample from Validation set:\n", dataset["validation"][0])
print("\nSample from Test set:\n", dataset["test"][0])

# Create a token-to-index mapping
unique_tokens = set()
for split in ["train", "validation", "test"]:
    for example in dataset[split]:
        unique_tokens.update(example['sequence'].split())

token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
vocab_size = len(token_to_idx)

# Define Dataset class for loading data
class SPRDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tokens = example['sequence'].split()
        indices = [token_to_idx[token] for token in tokens]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(example['label'], dtype=torch.long)

# Create datasets and dataloaders
batch_size = 32
train_dataset = SPRDataset(dataset['train'])
dev_dataset = SPRDataset(dataset['validation'])
test_dataset = SPRDataset(dataset['test'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Use a simple neural network architecture for initial testing
class SimpleSPRModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64):
        super(SimpleSPRModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.fc1(x.mean(dim=1)))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# Initialize model, criterion, and optimizer
model = SimpleSPRModel(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training and evaluation functions
def train_epoch(loader):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions)

# Training loop
epochs = 5
training_losses = []
validation_accuracies = []

for epoch in range(epochs):
    print(f"Running Epoch {epoch + 1}:")
    train_loss = train_epoch(train_loader)
    training_losses.append(train_loss)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

    valid_acc = evaluate(dev_loader)
    validation_accuracies.append(valid_acc)
    print(f"Epoch {epoch + 1}, Validation Accuracy: {valid_acc * 100:.2f}%")

# Test evaluation
print("\nEvaluating on Test Data...")
test_acc = evaluate(test_loader)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Visualize results
print("Generating result figures...")
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_losses, label='Training Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, [acc * 100 for acc in validation_accuracies], label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy Over Epochs')
plt.legend()
plt.savefig("Figure_1.png")
plt.close()

plt.figure()
plt.bar(['Validation Accuracy', 'Test Accuracy'], [validation_accuracies[-1] * 100, test_acc * 100], color=['blue', 'orange'])
plt.title('Final Accuracy Metrics')
plt.xlabel('Dataset Split')
plt.ylabel('Accuracy (%)')
plt.savefig("Figure_2.png")
plt.close()