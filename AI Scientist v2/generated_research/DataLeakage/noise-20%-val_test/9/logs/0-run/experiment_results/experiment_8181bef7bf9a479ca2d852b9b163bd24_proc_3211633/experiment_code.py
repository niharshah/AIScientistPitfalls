import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import time
import pathlib
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# ------------------------------------------------------------------ #
#                        DEVICE & SEED                               #
# ------------------------------------------------------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 2024
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------ #
#                     EXPERIMENT DATA STORE                          #
# ------------------------------------------------------------------ #
experiment_data: Dict = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "Rule_Fidelity": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------------ #
#                     DATASET LOADING HELPER                         #
# ------------------------------------------------------------------ #
disable_caching()  # avoid filling ~/.cache unnecessarily


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


# absolute path configurable via env
DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
assert DATA_PATH.exists(), f"Data path {DATA_PATH} does not exist"

dset = load_spr_bench(DATA_PATH)
print("Loaded splits:", list(dset.keys()))

# ------------------------------------------------------------------ #
#                          VOCABULARY                                #
# ------------------------------------------------------------------ #
PAD_IDX = 0  # reserve 0 for padding
all_chars = set(ch for seq in dset["train"]["sequence"] for ch in seq)
char2idx = {c: i + 1 for i, c in enumerate(sorted(all_chars))}  # start at 1
idx2char = {i: c for c, i in char2idx.items()}
VOCAB_SIZE = len(char2idx) + 1  # + PAD
print("Vocab size (with PAD):", VOCAB_SIZE)


# ------------------------------------------------------------------ #
#                         ENCODING UTILITIES                         #
# ------------------------------------------------------------------ #
def encode(seq: str) -> List[int]:
    return [char2idx[c] for c in seq]


def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs = [encode(b["sequence"]) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), PAD_IDX, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = True
    return padded.to(device), mask.to(device), labels.to(device)


# ------------------------------------------------------------------ #
#                         DATA LOADERS                               #
# ------------------------------------------------------------------ #
BATCH_SIZE = 256
VAL_BATCH = 512

train_loader = DataLoader(
    dset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(dset["dev"], batch_size=VAL_BATCH, collate_fn=collate_fn)
test_loader = DataLoader(dset["test"], batch_size=VAL_BATCH, collate_fn=collate_fn)

NUM_CLASSES = int(max(dset["train"]["label"]) + 1)
print("Number of classes:", NUM_CLASSES)


# ------------------------------------------------------------------ #
#                              MODEL                                 #
# ------------------------------------------------------------------ #
class SmallCNN(nn.Module):
    """
    Char-CNN with kernel sizes 1-3 and global-max pooling.
    kernel size 1 keeps character-level features (used for simple rules).
    """

    def __init__(self, vocab_size: int, embed_dim: int, n_classes: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        kernel_sizes = [1, 2, 3]
        n_filters = 64
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, n_filters, k, padding=0) for k in kernel_sizes]
        )
        self.activation = nn.ReLU()
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B,L] long
        mask: [B,L] bool
        """
        emb = self.embedding(x)  # [B,L,D]
        emb = emb.transpose(1, 2)  # [B,D,L] for conv1d
        feats = []
        for conv in self.convs:
            z = self.activation(conv(emb))  # [B,F,L']
            pooled = torch.max(z, dim=2).values  # global max pool -> [B,F]
            feats.append(pooled)
        feats = torch.cat(feats, dim=1)  # [B, F*k]
        logits = self.fc(feats)  # [B,C]
        return logits


EMBED_DIM = 64
model = SmallCNN(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, PAD_IDX).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)


# ------------------------------------------------------------------ #
#                  RULE EXTRACTION  (char-level)                     #
# ------------------------------------------------------------------ #
@torch.no_grad()
def extract_char_rules(top_k: int = 3) -> List[List[str]]:
    """
    Use kernel size 1 filters: they map directly from characters (after embedding*conv weight)
    We approximate char contribution by multiplying embedding matrix with conv weight.
    For each class we take FC weight @ conv filter to get class-specific importance.
    """
    # first conv is kernel size 1 by construction
    conv1: nn.Conv1d = model.convs[0]
    W_conv = conv1.weight.squeeze(-1)  # [F, D]
    W_embed = model.embedding.weight  # [V,D]
    char_feat = torch.matmul(W_embed, W_conv.T)  # [V, F]
    # class importance = char_feat @ fc_weight (only filters of conv1)
    fc_weights = model.fc.weight[:, : W_conv.size(0)]  # [C, F]
    char_cls_score = torch.matmul(char_feat, fc_weights.T)  # [V, C]
    top_chars = []
    for c in range(NUM_CLASSES):
        idxs = torch.topk(char_cls_score[:, c], top_k).indices.cpu().tolist()
        top_chars.append([idx2char[i] for i in idxs if i in idx2char])
    return top_chars  # list of list[str]


@torch.no_grad()
def rule_based_predict(tokens: torch.Tensor) -> torch.Tensor:
    """
    Very simple heuristic: if sequence contains any of the top characters
    for class c (and not for others), predict c else use fallback model.
    """
    rules = extract_char_rules(top_k=3)
    batch_preds = []
    seq_np = tokens.cpu().numpy()
    for seq in seq_np:
        chars = {idx for idx in seq if idx != PAD_IDX}
        matched = None
        for c, chars_c in enumerate(rules):
            if any(char2idx[ch] in chars for ch in chars_c):
                matched = c
                break
        if matched is None:
            matched = -1  # signal fallback
        batch_preds.append(matched)
    return torch.tensor(batch_preds, device=tokens.device)


# ------------------------------------------------------------------ #
#                            EVALUATION                              #
# ------------------------------------------------------------------ #
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    rule_match = 0
    for tokens, mask, labels in loader:
        logits = model(tokens, mask)
        loss = criterion(logits, labels)
        preds = logits.argmax(1)
        tot += labels.size(0)
        correct += (preds == labels).sum().item()
        loss_sum += loss.item() * labels.size(0)

        r_preds = rule_based_predict(tokens)
        mask_valid = r_preds != -1
        rule_match += (r_preds[mask_valid] == preds[mask_valid]).sum().item()
    acc = correct / tot
    rf = rule_match / tot
    return acc, loss_sum / tot, rf


# ------------------------------------------------------------------ #
#                          TRAINING LOOP                             #
# ------------------------------------------------------------------ #
EPOCHS = 8
for epoch in range(1, EPOCHS + 1):
    model.train()
    seen, correct, loss_sum = 0, 0, 0.0
    for tokens, mask, labels in train_loader:
        optimizer.zero_grad()
        logits = model(tokens, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        seen += labels.size(0)
        correct += (preds == labels).sum().item()
        loss_sum += loss.item() * labels.size(0)

    train_acc = correct / seen
    train_loss = loss_sum / seen

    val_acc, val_loss, val_rf = evaluate(val_loader)

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["Rule_Fidelity"].append(val_rf)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f} | "
        f"RuleFid={val_rf:.3f}"
    )

# ------------------------------------------------------------------ #
#                        TEST EVALUATION                             #
# ------------------------------------------------------------------ #
test_acc, test_loss, test_rf = evaluate(test_loader)
print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.3f}, RuleFid={test_rf:.3f}")

# store predictions & gts
model.eval()
all_preds, all_gts = [], []
with torch.no_grad():
    for tokens, mask, labels in test_loader:
        out = model(tokens, mask).argmax(1).cpu()
        all_preds.append(out)
        all_gts.append(labels.cpu())

experiment_data["SPR_BENCH"]["predictions"] = torch.cat(all_preds).numpy()
experiment_data["SPR_BENCH"]["ground_truth"] = torch.cat(all_gts).numpy()

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))

# Print simple human-readable rules
for cls, chars in enumerate(extract_char_rules(top_k=5)):
    print(f"Class {cls} indicative characters: {chars}")
