import os
# Force CPU usage by disabling CUDA devices.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.metrics import confusion_matrix

# ----------------------------
# Reproducibility and Device
# ----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cpu")
print("Using device:", device)

# ----------------------------
# Dataset Loading (assumed to be pre-loaded by provided code, but we include here for completeness)
# ----------------------------
benchmarks = ["SFRFG", "IJSJF", "GURSG", "TSHUY"]
all_datasets = {}
for bm in benchmarks:
    files = {
        "train": f"SPR_BENCH/{bm}/train.csv",
        "dev": f"SPR_BENCH/{bm}/dev.csv",
        "test": f"SPR_BENCH/{bm}/test.csv"
    }
    dataset = load_dataset("csv", data_files=files)
    all_datasets[bm] = dataset

for bm, ds in all_datasets.items():
    print(f"Benchmark: {bm}")
    for split in ds.keys():
        print(f"  {split}: {len(ds[split])} instances")

# ----------------------------
# Data Preprocessing: Build vocabulary and define SPRDataset
# ----------------------------
class SPRDataset(Dataset):
    def __init__(self, examples, vocab, max_len):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex["sequence"].strip().split()
        indices = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        if len(indices) < self.max_len:
            indices = indices + [self.vocab["<pad>"]] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        label = int(ex["label"])
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

processed_data = {}
for bm in benchmarks:
    ds = all_datasets[bm]
    vocab = {"<pad>": 0, "<unk>": 1}
    token_set = set()
    max_len = 0
    for ex in ds["train"]:
        tokens = ex["sequence"].strip().split()
        max_len = max(max_len, len(tokens))
        token_set.update(tokens)
    for token in sorted(token_set):
        if token not in vocab:
            vocab[token] = len(vocab)
    processed_data[bm] = {
        "max_len": max_len,
        "vocab": vocab,
        "train": SPRDataset(ds["train"], vocab, max_len),
        "dev": SPRDataset(ds["dev"], vocab, max_len),
        "test": SPRDataset(ds["test"], vocab, max_len)
    }

# ----------------------------
# Model Definition: PositionalEncoding and SimpleTransformer
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, emb_dim)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, num_layers, num_classes, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=emb_dim*2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(emb_dim, num_classes)
        # Extra chain-of-thought head to produce symbolic intermediate representations
        self.ct_head = nn.Linear(emb_dim, emb_dim)
    def forward(self, src):
        # src shape: (batch_size, seq_len)
        emb = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        emb = self.pos_encoder(emb)
        emb = emb.transpose(0, 1)  # (seq_len, batch, emb_dim)
        transformer_out = self.transformer_encoder(emb)
        transformer_out = transformer_out.transpose(0, 1)  # (batch, seq_len, emb_dim)
        pooled = transformer_out.mean(dim=1)  # simple average pooling for sequence
        ct_vector = torch.tanh(self.ct_head(pooled))
        logits = self.fc(self.dropout(pooled))
        return logits, ct_vector

# ----------------------------
# Hyperparameters for all experiments
# ----------------------------
emb_dim = 32
nhead = 4
num_layers = 1
num_classes = 2
batch_size = 32
num_epochs = 3
learning_rate = 0.001
fusion_scale = 2.0  # scaling factor for symbolic fusion

# Dictionary to store final test accuracies and predictions for each benchmark
results = {}
model_store = {}
all_preds_dict = {}
all_labels_dict = {}

# ----------------------------
# Loop over benchmarks and run experiments
# ----------------------------
for bm in benchmarks:
    print("\n===================================================================")
    print(f"Starting experiment for benchmark {bm}.")
    print("This experiment trains a lightweight transformer classifier on the training split, uses the dev split for tuning,")
    print("and finally evaluates on the hidden test split. The symbolic module computes a verification score based on the")
    print("fraction of tokens starting with '▲'. The score is fused with the neural logits by adjusting the positive class logit.")
    print("This experiment is designed to assess accuracy and generalization across sequence lengths and rule complexity.\n")
    
    # Retrieve processed data for benchmark
    data = processed_data[bm]
    train_dataset = data["train"]
    dev_dataset = data["dev"]
    test_dataset = data["test"]
    max_len = data["max_len"]
    vocab = data["vocab"]
    vocab_size = len(vocab)
    
    # Create DataLoaders (num_workers=0 for stability)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model, optimizer, and loss function
    model = SimpleTransformer(vocab_size, emb_dim, nhead, num_layers, num_classes, max_len)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Monkey-patch optimizer to avoid CUDA graph capture health check when using CPU.
    if device.type != "cuda":
        optimizer._cuda_graph_capture_health_check = lambda: None
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_tokens, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_tokens)
            # Build reverse vocab mapping: index -> token
            rev_vocab = {i: token for token, i in vocab.items()}
            symbolic_scores = []
            # Calculate symbolic verification score for each instance: fraction of tokens starting with '▲'
            batch_tokens_cpu = batch_tokens.cpu().numpy()
            for tokens in batch_tokens_cpu:
                token_list = [rev_vocab[idx] for idx in tokens if idx != vocab["<pad>"]]
                if len(token_list) == 0:
                    score = 0.0
                else:
                    score = sum(1 for tok in token_list if tok.startswith("▲")) / len(token_list)
                symbolic_scores.append(score)
            symbolic_scores = torch.tensor(symbolic_scores, dtype=torch.float32, device=device)
            # Fuse symbolic scores with neural logits: adjust positive class logit (index 1)
            adjusted_logits = logits.clone()
            adjusted_logits[:, 1] = adjusted_logits[:, 1] + fusion_scale * (symbolic_scores - 0.5)
            loss = criterion(adjusted_logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_tokens.size(0)
        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Training Loss: {avg_loss:.4f}")
    
    # Evaluation on Dev Set
    print("\nEvaluating on Dev split to measure generalization on held-out tuning data:")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_tokens, batch_labels in dev_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            logits, _ = model(batch_tokens)
            rev_vocab = {i: token for token, i in vocab.items()}
            symbolic_scores = []
            batch_tokens_cpu = batch_tokens.cpu().numpy()
            for tokens in batch_tokens_cpu:
                token_list = [rev_vocab[idx] for idx in tokens if idx != vocab["<pad>"]]
                if len(token_list) == 0:
                    score = 0.0
                else:
                    score = sum(1 for tok in token_list if tok.startswith("▲")) / len(token_list)
                symbolic_scores.append(score)
            symbolic_scores = torch.tensor(symbolic_scores, dtype=torch.float32, device=device)
            adjusted_logits = logits.clone()
            adjusted_logits[:, 1] = adjusted_logits[:, 1] + fusion_scale * (symbolic_scores - 0.5)
            preds = adjusted_logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
    dev_accuracy = (correct / total) * 100
    print(f"Dev Accuracy: {dev_accuracy:.2f}%")
    
    # Evaluation on Test Set (hidden data evaluation)
    print("\nEvaluating on Test split to assess final model generalization on unseen data:")
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_tokens, batch_labels in test_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            logits, _ = model(batch_tokens)
            rev_vocab = {i: token for token, i in vocab.items()}
            symbolic_scores = []
            batch_tokens_cpu = batch_tokens.cpu().numpy()
            for tokens in batch_tokens_cpu:
                token_list = [rev_vocab[idx] for idx in tokens if idx != vocab["<pad>"]]
                if len(token_list) == 0:
                    score = 0.0
                else:
                    score = sum(1 for tok in token_list if tok.startswith("▲")) / len(token_list)
                symbolic_scores.append(score)
            symbolic_scores = torch.tensor(symbolic_scores, dtype=torch.float32, device=device)
            adjusted_logits = logits.clone()
            adjusted_logits[:, 1] = adjusted_logits[:, 1] + fusion_scale * (symbolic_scores - 0.5)
            preds = adjusted_logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
    test_accuracy = (correct / total) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    results[bm] = test_accuracy
    model_store = model_store if 'model_store' in globals() else {}
    model_store[bm] = model
    all_preds_dict[bm] = all_preds
    all_labels_dict[bm] = all_labels

# ----------------------------
# Generate Figures
# Figure_1: Bar plot of final test accuracies across benchmarks.
# ----------------------------
plt.figure(figsize=(8,6))
bm_names = list(results.keys())
accuracy_values = [results[bm] for bm in bm_names]
plt.bar(bm_names, accuracy_values, color='skyblue')
plt.xlabel("Benchmark")
plt.ylabel("Test Accuracy (%)")
plt.title("Figure_1.png: Final Test Accuracies across Benchmarks")
for i, acc in enumerate(accuracy_values):
    plt.text(i, acc + 1, f"{acc:.1f}%", ha="center")
plt.ylim(0, 105)
plt.savefig("Figure_1.png")
print("\nFigure_1.png saved: Bar plot of final test accuracies across benchmarks.\n")

# Figure_2: Confusion Matrix for the first benchmark (SFRFG)
print("Generating Figure_2.png: Confusion matrix for benchmark SFRFG.")
bm0 = benchmarks[0]
test_dataset = processed_data[bm0]["test"]
vocab = processed_data[bm0]["vocab"]
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
true_labels = []
pred_labels = []
model = model_store[bm0]
model.eval()
with torch.no_grad():
    for batch_tokens, batch_labels in test_loader:
        batch_tokens = batch_tokens.to(device)
        logits, _ = model(batch_tokens)
        rev_vocab = {i: token for token, i in vocab.items()}
        symbolic_scores = []
        for tokens in batch_tokens.cpu().numpy():
            token_list = [rev_vocab[idx] for idx in tokens if idx != vocab["<pad>"]]
            if len(token_list) == 0:
                score = 0.0
            else:
                score = sum(1 for tok in token_list if tok.startswith("▲")) / len(token_list)
            symbolic_scores.append(score)
        symbolic_scores = torch.tensor(symbolic_scores, dtype=torch.float32, device=device)
        adjusted_logits = logits.clone()
        adjusted_logits[:, 1] = adjusted_logits[:, 1] + fusion_scale * (symbolic_scores - 0.5)
        preds = adjusted_logits.argmax(dim=1)
        true_labels.extend(batch_labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Figure_2.png: Confusion Matrix for Benchmark " + bm0)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["0", "1"])
plt.yticks(tick_marks, ["0", "1"])
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("Figure_2.png")
print("Figure_2.png saved: Confusion matrix for benchmark", bm0)

# ----------------------------
# Final Summary: Report final test accuracies for each benchmark.
# ----------------------------
print("\nFinal Test Accuracies per Benchmark:")
for bm in benchmarks:
    print(f"{bm}: {results[bm]:.2f}%")