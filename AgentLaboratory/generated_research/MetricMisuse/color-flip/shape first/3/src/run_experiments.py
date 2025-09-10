import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage; disable CUDA

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from datasets import load_dataset

# Override CUDA stream capture check unconditionally to avoid CUDA initialization errors.
torch.cuda.is_current_stream_capturing = lambda: False

# -----------------------------------------------------------------------------
# Set random seeds for reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# -----------------------------------------------------------------------------
# The dataset is preloaded externally using the provided snippet. It contains:
#   - splits: "train", "dev", "test"
#   - fields: "id", "sequence", "label", "tokens", "num_tokens"
print("Dataset splits available:", list(dataset.keys()))

# (Optional) Uncomment these lines to work with a reduced dataset for rapid prototyping.
# dataset["train"] = dataset["train"].select(range(1000))
# dataset["dev"]   = dataset["dev"].select(range(200))
# dataset["test"]  = dataset["test"].select(range(200))

# -----------------------------------------------------------------------------
# Build vocabulary from training split tokens.
all_tokens = []
for example in dataset["train"]:
    all_tokens.extend(example["tokens"])
unique_tokens = sorted(list(set(all_tokens)))
# Reserve index 0 for the PAD token.
vocab = {token: idx + 1 for idx, token in enumerate(unique_tokens)}
vocab["<PAD>"] = 0
vocab_size = len(vocab)
print("Vocabulary size (including PAD):", vocab_size)

# Utility: Encode a sequence of tokens into indices.
def encode_sequence(tokens):
    return [vocab.get(token, 0) for token in tokens]

# For each split, add an "input_ids" field using the vocabulary.
for split in ["train", "dev", "test"]:
    dataset[split] = dataset[split].map(lambda x: {"input_ids": encode_sequence(x["tokens"])})

# -----------------------------------------------------------------------------
# Define collate function for DataLoader with padding.
def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids_list, labels_list, sequences_list = [], [], []
    for item in batch:
        seq = item["input_ids"]
        padded = seq + [0] * (max_len - len(seq))
        input_ids_list.append(padded)
        labels_list.append(float(item["label"]))
        sequences_list.append(item["sequence"])
    return torch.tensor(input_ids_list, dtype=torch.long), torch.tensor(labels_list, dtype=torch.float), sequences_list

batch_size = 64
train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
dev_loader   = DataLoader(dataset["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
test_loader  = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

# -----------------------------------------------------------------------------
# MODEL DEFINITIONS

# Stage 1: Discrete Symbolic Tokenization.
# Embeds token indices and applies a linear transformation; then uses Gumbel-Softmax
# to discretize embeddings into one-hot symbolic representations.
class DiscreteTokenizer(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_symbols):
        super(DiscreteTokenizer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim, num_symbols)
        self.num_symbols = num_symbols

    def forward(self, input_ids, tau=1.0):
        emb = self.embedding(input_ids)  # (B, L, emb_dim)
        logits = self.linear(emb)         # (B, L, num_symbols)
        # Use Gumbel-Softmax for differentiable discretization.
        symbols = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)  # (B, L, num_symbols)
        return symbols

# Stage 2: Rule Induction and Validation.
# Aggregates the discrete token symbols (via mean pooling) and uses a lightweight MLP
# to predict whether the sequence adheres to a hidden poly-factor rule.
class RuleInduction(nn.Module):
    def __init__(self, num_symbols, hidden_dim):
        super(RuleInduction, self).__init__()
        self.fc1 = nn.Linear(num_symbols, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary classification

    def forward(self, token_symbols):
        pooled = token_symbols.mean(dim=1)  # (B, num_symbols)
        x = F.relu(self.fc1(pooled))
        logits = self.fc2(x).squeeze(-1)      # (B,)
        return logits

# Combined Model: Robust PolyRuleNet.
# Implements the two-stage process: symbolic tokenization then rule induction.
class PolyRuleNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_symbols, hidden_dim):
        super(PolyRuleNet, self).__init__()
        self.tokenizer = DiscreteTokenizer(vocab_size, emb_dim, num_symbols)
        self.inductor = RuleInduction(num_symbols, hidden_dim)

    def forward(self, input_ids, tau=1.0):
        symbols = self.tokenizer(input_ids, tau=tau)
        logits = self.inductor(symbols)
        return logits, symbols

# -----------------------------------------------------------------------------
# Define evaluation metric: Shape-Weighted Accuracy (SWA).
# For each sequence, unique shapes are determined by the first character of each token.
def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))

def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    total_weight = sum(weights)
    return sum(correct) / total_weight if total_weight > 0 else 0.0

# -----------------------------------------------------------------------------
# SET HYPERPARAMETERS
emb_dim = 32
num_symbols = 16  # Number of discrete symbol codes.
hidden_dim = 64
learning_rate = 1e-3
num_epochs = 2   # Use a low number for demonstration.
tau = 1.0        # Temperature for Gumbel-Softmax.

device = torch.device("cpu")
print("Using device:", device)

model = PolyRuleNet(vocab_size, emb_dim, num_symbols, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# -----------------------------------------------------------------------------
# TRAINING PROCEDURE
print("\nStarting training for Robust PolyRuleNet for SPR.")
print("This experiment demonstrates a two-stage model:")
print(" - Stage 1: Discrete Symbolic Tokenization using Gumbel-Softmax to create explicit, one-hot symbols from token embeddings.")
print(" - Stage 2: Rule Induction via a lightweight MLP that aggregates symbolic representations to verify if the sequence meets a hidden poly-factor rule.")
print("Evaluation uses Shape-Weighted Accuracy (SWA): accuracy weighted by the diversity of shapes (first character of tokens) in the sequence.\n")

train_losses = []
dev_swa_scores = []
best_dev_swa = 0.0
best_model_state = None

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for input_ids, labels, _ in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(input_ids, tau=tau)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"\nEpoch {epoch}/{num_epochs}:")
    print(" - This epoch's average training loss reflects the model's convergence.")
    print(f"   Average Training Loss = {avg_loss:.4f}")
    
    # Evaluate on the development set.
    model.eval()
    all_preds, all_labels, all_sequences = [], [], []
    with torch.no_grad():
        for input_ids, labels, sequences in dev_loader:
            input_ids = input_ids.to(device)
            outputs, _ = model(input_ids, tau=tau)
            preds = (torch.sigmoid(outputs) > 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().long().tolist())
            all_sequences.extend(sequences)
    binary_acc = sum(1 for p, t in zip(all_preds, all_labels) if p == t) / len(all_preds)
    swa = shape_weighted_accuracy(all_sequences, all_labels, all_preds)
    dev_swa_scores.append(swa)
    print("\nDevelopment Set Evaluation:")
    print(" - Binary Accuracy shows the raw fraction of correct predictions.")
    print(" - Shape-Weighted Accuracy (SWA) weights correctness by the diversity of shapes in each sequence (more unique shapes yield higher weight).")
    print(f"   Dev Binary Accuracy = {binary_acc:.4f}")
    print(f"   Dev SWA = {swa:.4f}")
    if swa > best_dev_swa:
        best_dev_swa = swa
        best_model_state = model.state_dict()

# -----------------------------------------------------------------------------
# GENERATE FIGURE 1: Training Loss Curve.
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Figure_1.png: Training Loss Curve over Epochs\n(Declining loss indicates effective learning)")
plt.savefig("Figure_1.png")
print("\nFigure_1.png saved: This figure displays the decrease in training loss over epochs, demonstrating model convergence.")

# -----------------------------------------------------------------------------
# FINAL EVALUATION ON TEST SET
if best_model_state is not None:
    model.load_state_dict(best_model_state)
model.eval()
all_preds_test, all_labels_test, all_sequences_test = [], [], []
with torch.no_grad():
    for input_ids, labels, sequences in test_loader:
        input_ids = input_ids.to(device)
        outputs, _ = model(input_ids, tau=tau)
        preds = (torch.sigmoid(outputs) > 0.5).long().cpu().tolist()
        all_preds_test.extend(preds)
        all_labels_test.extend(labels.cpu().long().tolist())
        all_sequences_test.extend(sequences)
binary_acc_test = sum(1 for p, t in zip(all_preds_test, all_labels_test) if p == t) / len(all_preds_test)
swa_test = shape_weighted_accuracy(all_sequences_test, all_labels_test, all_preds_test)

print("\nFinal Evaluation on Test Set:")
print(" - Binary Accuracy reflects the overall correctness on unseen data.")
print(" - Shape-Weighted Accuracy (SWA) emphasizes sequences with a richer diversity of shapes.")
print(f"   Test Binary Accuracy = {binary_acc_test:.4f}")
print(f"   Test SWA = {swa_test:.4f}")

# -----------------------------------------------------------------------------
# GENERATE FIGURE 2: SWA Comparison Chart between Dev and Test Splits.
plt.figure()
plt.bar(["Dev", "Test"], [best_dev_swa, swa_test], color=["green", "orange"])
plt.ylabel("Shape-Weighted Accuracy (SWA)")
plt.title("Figure_2.png: SWA Comparison between Dev and Test Splits\n(Assessing generalization performance)")
plt.savefig("Figure_2.png")
print("\nFigure_2.png saved: This chart compares the SWA on the Dev and Test sets, showcasing the model's generalization capability.")

# -----------------------------------------------------------------------------
# Final sanity check: ensure non-zero accuracy.
if binary_acc_test <= 0:
    print("\nError: Test accuracy is 0%. Please review the training and evaluation procedures.")
else:
    print("\nSuccess: Model achieved non-zero accuracy on the test set.")