import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Ensure CPU only by disabling CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False

from datasets import load_dataset

# Load the datasets
train_dataset = load_dataset('SPR_BENCH', split='train')
validation_dataset = load_dataset('SPR_BENCH', split='validation')
test_dataset_1 = load_dataset('SPR_BENCH', split='test[:50%]')
test_dataset_2 = load_dataset('SPR_BENCH', split='test[50%:]')

def calculate_label_distribution(dataset, dataset_name):
    label_counts = Counter(dataset['label'])
    print(f"\n{dataset_name} Label Distribution:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count / len(dataset):.2%}")

# Calculate label distributions
calculate_label_distribution(train_dataset, "Train Dataset")
calculate_label_distribution(validation_dataset, "Validation Dataset")
calculate_label_distribution(test_dataset_1, "Test Dataset 1")
calculate_label_distribution(test_dataset_2, "Test Dataset 2")

def process_sequence(sequence):
    max_length = 100
    char_to_int = {str(c): i for i, c in enumerate(sorted(set(''.join(sequence))), start=1)}
    seq_tensor = torch.tensor([char_to_int.get(c, 0) for c in sequence], dtype=torch.float32)
    if seq_tensor.size(0) > max_length:
        seq_tensor = seq_tensor[:max_length]
    else:
        seq_tensor = torch.cat([seq_tensor, torch.zeros(max_length - seq_tensor.size(0), dtype=torch.float32)])
    return seq_tensor

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

def create_data_loader(dataset, batch_size):
    def collate_fn(batch):
        sequences = [process_sequence(s['sequence']) for s in batch]
        labels = torch.tensor([s['label'] for s in batch], dtype=torch.long)
        sequences_tensor = torch.stack(sequences)
        return {'sequence': sequences_tensor, 'label': labels}
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

input_dim = 100
hidden_dim = 64
output_dim = len(set(train_dataset['label']))
model = SimpleNN(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['sequence'])
        loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch['sequence'])
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(batch['label'].numpy())
            all_preds.extend(predicted.numpy())
    return all_labels, all_preds

def calculate_metrics(labels, preds, dataset_name):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=1)
    recall = recall_score(labels, preds, average='macro', zero_division=1)
    f1 = f1_score(labels, preds, average='macro', zero_division=1)
    print(f"{dataset_name} - Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}")

epochs = 5
batch_size = 64
train_loader = create_data_loader(train_dataset, batch_size)
validation_loader = create_data_loader(validation_dataset, batch_size)

for epoch in range(epochs):
    print(f"\nTraining Epoch {epoch+1}/{epochs}")
    train_loss = train_model(model, train_loader, criterion, optimizer)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}")

print("\nEvaluation results on Test Dataset 1")
labels_1, preds_1 = evaluate_model(model, create_data_loader(test_dataset_1, batch_size))
calculate_metrics(labels_1, preds_1, "Test Dataset 1")

print("\nEvaluation results on Test Dataset 2")
labels_2, preds_2 = evaluate_model(model, create_data_loader(test_dataset_2, batch_size))
calculate_metrics(labels_2, preds_2, "Test Dataset 2")

def plot_results(labels, preds, title, filename):
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(labels)), labels, label='True Labels', marker='o', alpha=0.6)
    plt.scatter(range(len(preds)), preds, label='Predictions', marker='x', alpha=0.6)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Labels')
    plt.legend(frameon=False)
    plt.savefig(filename)
    plt.close()

plot_results(labels_1, preds_1, "Predictions vs True Labels: Test Dataset 1", "Figure_1.png")
plot_results(labels_2, preds_2, "Predictions vs True Labels: Test Dataset 2", "Figure_2.png")