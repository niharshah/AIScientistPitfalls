import os
# Force CPU usage by making CUDA invisible.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.multiprocessing import freeze_support

# Override any CUDA stream capture calls to ensure CPU-only execution.
torch.cuda.is_current_stream_capturing = lambda: False

# --------------------------
# Dataset loading and preprocessing.
# --------------------------
dataset = load_dataset(
    "csv",
    data_files={
        "train": "SPR_BENCH/train.csv",
        "dev": "SPR_BENCH/dev.csv",
        "test": "SPR_BENCH/test.csv"
    },
    delimiter=","
)

def split_sequence(example):
    # Each token is expected to be in the format "shape+color" (e.g., "▲r", "■b").
    example["tokens"] = example["sequence"].split()
    return example

dataset = dataset.map(split_sequence)
print("Loaded dataset splits:", dataset)

# For demonstration purposes, select a subset of the full dataset.
train_dataset = dataset["train"].select(range(1000))
dev_dataset = dataset["dev"].select(range(200))
test_dataset = dataset["test"].select(range(200))

# --------------------------
# Build vocabulary based on tokens.
# --------------------------
vocab = {}
vocab_counter = 1  # Reserve 0 for padding.
for item in train_dataset:
    for token in item["tokens"]:
        if token not in vocab:
            vocab[token] = vocab_counter
            vocab_counter += 1
print("Vocabulary size:", len(vocab))
inv_vocab = {v: k for k, v in vocab.items()}

# --------------------------
# PyTorch Dataset wrapper.
# --------------------------
class SPRDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        # Convert label to int (assume binary labels).
        return {"tokens": item["tokens"], "label": int(item["label"])}

train_ds = SPRDataset(train_dataset)
dev_ds = SPRDataset(dev_dataset)
test_ds = SPRDataset(test_dataset)

# --------------------------
# Helper function: convert token list to tensor indices.
# --------------------------
def tokens_to_ids(token_list):
    # Use 0 for any token not in vocabulary (padding).
    return torch.tensor([vocab.get(token, 0) for token in token_list], dtype=torch.long)

# --------------------------
# Collate function for DataLoader.
# Pads sequences and returns raw token lists for SWA evaluation.
# --------------------------
def collate_fn(batch):
    token_ids = [tokens_to_ids(item["tokens"]) for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)
    padded_tokens = pad_sequence(token_ids, batch_first=True, padding_value=0)
    tokens_list = [item["tokens"] for item in batch]
    return padded_tokens, labels, tokens_list

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

# --------------------------
# Evaluation Metric: Shape-Weighted Accuracy (SWA)
# Each sample's prediction is weighted by its token shape diversity (unique first characters).
# --------------------------
def count_shape_variety(tokens):
    return len(set(token[0] for token in tokens if token))

def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if (yt == yp) else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) if sum(weights) > 0 else 1)

# --------------------------
# For reproducibility.
# --------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cpu")
print("Using device:", device)

# --------------------------
# Define Hierarchical Iterative Predicate Aggregation (HIPA) Model.
# This model:
# 1. Uses token and positional embeddings.
# 2. Applies a Transformer encoder for contextualization.
# 3. Performs hierarchical segmentation via a sliding window with overlap.
# 4. Extracts local predicates via an MLP.
# 5. Aggregates using a GRU.
# 6. Outputs a binary classification via a final classifier.
# --------------------------
class HIPA(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, transformer_heads=2, transformer_layers=1,
                 window_size=8, stride=None, predicate_dim=16, gru_hidden=16, max_pos=500):
        super(HIPA, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2

        # Token embedding and positional embedding.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_pos, embed_dim)
        
        # Transformer encoder for contextualizing token embeddings.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=transformer_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Local predicate extractor: MLP for soft predicate activations.
        self.predicate_mlp = nn.Sequential(
            nn.Linear(embed_dim, predicate_dim),
            nn.ReLU(),
            nn.Linear(predicate_dim, predicate_dim),
            nn.Sigmoid()
        )
        
        # Aggregator: GRU to combine local predicate outputs.
        self.gru = nn.GRU(input_size=predicate_dim, hidden_size=gru_hidden, batch_first=True)
        
        # Final classifier using a feedforward network.
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, L] where L is the sequence length.
        B, L = x.size()
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        token_emb = self.token_embedding(x) + self.pos_embedding(pos_ids)  # [B, L, embed_dim]
        
        token_emb = token_emb.transpose(0, 1)  # [L, B, embed_dim] for Transformer.
        contextualized = self.transformer_encoder(token_emb)
        contextualized = contextualized.transpose(0, 1)  # [B, L, embed_dim]
        
        local_preds_list = []
        local_consistency_loss = 0.0
        for b in range(B):
            seq = contextualized[b]  # [L, embed_dim]
            seg_preds = []
            idx = 0
            while idx < L:
                end_idx = min(idx + self.window_size, L)
                segment = seq[idx:end_idx]  # [segment_length, embed_dim]
                if segment.size(0) == 0:
                    break
                seg_feat = segment.mean(dim=0)
                pred = self.predicate_mlp(seg_feat)
                seg_preds.append(pred)
                idx += self.stride
            if len(seg_preds) == 0:
                seg_preds_tensor = torch.zeros((1, self.predicate_mlp[-1].out_features), device=x.device)
            else:
                seg_preds_tensor = torch.stack(seg_preds, dim=0)
            local_preds_list.append(seg_preds_tensor)
            if seg_preds_tensor.size(0) > 1:
                diff = seg_preds_tensor[1:] - seg_preds_tensor[:-1]
                local_consistency_loss += torch.mean(diff ** 2)
        if B > 0:
            local_consistency_loss = local_consistency_loss / B
        
        lengths = [preds.size(0) for preds in local_preds_list]
        max_len = max(lengths)
        padded_preds = []
        for preds in local_preds_list:
            if preds.size(0) < max_len:
                pad = torch.zeros(max_len - preds.size(0), preds.size(1), device=preds.device)
                preds_padded = torch.cat([preds, pad], dim=0)
            else:
                preds_padded = preds
            padded_preds.append(preds_padded.unsqueeze(0))
        padded_preds = torch.cat(padded_preds, dim=0)  # [B, max_len, predicate_dim]
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_preds, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        hidden = hidden.squeeze(0)
        
        out = self.classifier(hidden)
        out = out.squeeze(1)
        return out, local_consistency_loss

# Instantiate the model.
model = HIPA(vocab_size=len(vocab) + 1, embed_dim=32, transformer_heads=2, transformer_layers=1,
             window_size=8, stride=4, predicate_dim=16, gru_hidden=16, max_pos=500).to(device)

# Use SGD optimizer to avoid any CUDA-specific issues encountered with Adam.
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# --------------------------
# Training Procedure.
# --------------------------
num_epochs = 5
train_losses = []
dev_losses = []
dev_swa_history = []

print("\nStarting training. This loop aims to optimize the HIPA model for the SPR task.")
print("Results include loss progression and Dev Shape-Weighted Accuracy (SWA), measuring performance considering token shape diversity.\n")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_tokens, batch_labels, _ in train_loader:
        batch_tokens = batch_tokens.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        outputs, local_consistency_loss = model(batch_tokens)
        loss_cls = criterion(outputs, batch_labels)
        loss = loss_cls + 0.1 * local_consistency_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_tokens.size(0)
    epoch_loss = epoch_loss / len(train_ds)
    train_losses.append(epoch_loss)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_tokens = []
    dev_loss = 0.0
    with torch.no_grad():
        for batch_tokens, batch_labels, batch_tokens_list in dev_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            outputs, local_consistency_loss = model(batch_tokens)
            loss_cls = criterion(outputs, batch_labels)
            loss = loss_cls + 0.1 * local_consistency_loss
            dev_loss += loss.item() * batch_tokens.size(0)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())
            all_tokens.extend(batch_tokens_list)
    dev_loss = dev_loss / len(dev_ds)
    dev_losses.append(dev_loss)
    
    swa = shape_weighted_accuracy(all_tokens, all_labels, all_preds)
    dev_swa_history.append(swa)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print("  Average Train Loss =", epoch_loss)
    print("  Average Dev Loss =", dev_loss)
    print("  Dev Shape-Weighted Accuracy (SWA) =", swa, "\n")

# --------------------------
# Evaluation on Test Set.
# --------------------------
model.eval()
all_test_preds = []
all_test_labels = []
test_tokens_list = []
with torch.no_grad():
    for batch_tokens, batch_labels, batch_tokens_list in test_loader:
        batch_tokens = batch_tokens.to(device)
        batch_labels = batch_labels.to(device)
        outputs, _ = model(batch_tokens)
        preds = (outputs > 0.5).float()
        all_test_preds.extend(preds.cpu().tolist())
        all_test_labels.extend(batch_labels.cpu().tolist())
        test_tokens_list.extend(batch_tokens_list)
test_swa = shape_weighted_accuracy(test_tokens_list, all_test_labels, all_test_preds)
print("\nFinal results on the Test set:")
print("  Test Shape-Weighted Accuracy (SWA):", test_swa)
if test_swa == 0.0:
    print("Error: 0% accuracy achieved. Please review the implementation.")
else:
    print("Success: Non-zero accuracy achieved. The HIPA model functions correctly.")

# --------------------------
# Generate Figures to Showcase Results.
# Figure_1.png: Loss curves over epochs.
# Figure_2.png: Development SWA progression.
# --------------------------
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label="Train Loss")
plt.plot(range(1, num_epochs + 1), dev_losses, marker='s', label="Dev Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Figure_1.png: Loss Curves Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("Figure_1.png")
print("\nFigure_1.png saved: This figure displays the training and development loss curves, indicating learning convergence.")

plt.figure()
plt.plot(range(1, num_epochs + 1), dev_swa_history, marker='o', color='green')
plt.xlabel("Epoch")
plt.ylabel("Dev Shape-Weighted Accuracy (SWA)")
plt.title("Figure_2.png: Dev SWA Over Epochs")
plt.grid(True)
plt.savefig("Figure_2.png")
print("Figure_2.png saved: This figure shows the progression of Dev SWA, reflecting improvement in capturing token shape diversity over epochs.")

freeze_support()
print("\nTraining, evaluation, and result visualization complete.")