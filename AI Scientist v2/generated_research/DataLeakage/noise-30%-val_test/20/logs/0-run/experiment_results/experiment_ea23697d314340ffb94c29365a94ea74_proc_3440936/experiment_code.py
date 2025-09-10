import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score
from datetime import datetime

# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- DATA LOADING -----------------
# Try to import the helper; if unavailable just build synthetic data
def try_load_spr(root):
    try:
        from datasets import load_dataset, DatasetDict

        if not os.path.isdir(root):
            raise FileNotFoundError

        def _ld(split_file):
            return load_dataset(
                "csv",
                data_files=os.path.join(root, split_file),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = {}
        for sp in ["train", "dev", "test"]:
            d[sp] = _ld(f"{sp}.csv")
        return d
    except Exception as e:
        print("Could not load SPR_BENCH – generating synthetic dataset:", e)
        np.random.seed(0)
        vocab = list("ABCDE")

        def gen(n):
            seqs, labels = [], []
            for _ in range(n):
                length = np.random.randint(6, 16)
                s = "".join(np.random.choice(vocab, length))
                # Simple rule: label 1 if count('A') is even, else 0
                lbl = int((s.count("A") % 2) == 0)
                seqs.append(s)
                labels.append(lbl)
            return {"sequence": seqs, "label": labels, "id": [str(i) for i in range(n)]}

        return {"train": gen(2000), "dev": gen(500), "test": gen(500)}


DATA_PATH = os.environ.get("SPR_DATA_PATH", "./SPR_BENCH")
dsets = try_load_spr(DATA_PATH)

# ----------------- VOCAB BUILDING -----------------
all_chars = set(ch for seq in dsets["train"]["sequence"] for ch in seq)
stoi = {ch: i + 1 for i, ch in enumerate(sorted(all_chars))}  # 0 = PAD
itos = {i: ch for ch, i in stoi.items()}
pad_id = 0
vocab_size = len(stoi) + 1
print(f"Vocab size: {vocab_size}")


# ----------------- DATASET CLASS -----------------
class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [int(l) for l in hf_split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = [stoi[ch] for ch in self.seqs[idx]]
        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lengths = [len(ex["input_ids"]) for ex in batch]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
    for i, ex in enumerate(batch):
        seq = ex["input_ids"]
        input_ids[i, : len(seq)] = seq
    return {"input_ids": input_ids.to(device), "labels": labels.to(device)}


batch_size = 256
train_loader = DataLoader(
    SPRDataset(dsets["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRDataset(dsets["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(dsets["test"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ----------------- MODEL -----------------
class TextCNN(nn.Module):
    def __init__(
        self, vocab, emb_dim=64, num_classes=2, kernel_sizes=(3, 4, 5), num_channels=64
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_channels, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, x):  # x: (B, T)
        emb = self.embedding(x).transpose(1, 2)  # (B, emb, T)
        convs = [F.relu(conv(emb)) for conv in self.convs]
        pools = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]
        feats = torch.cat(pools, 1)
        feats = self.dropout(feats)
        return self.fc(feats)


model = TextCNN(vocab_size).to(device)

# ----------------- TRAINING PREP -----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ----------------- TRAINING LOOP -----------------
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds = logits.argmax(1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].detach().cpu().numpy())
    train_loss = total_loss / len(train_loader.dataset)
    train_f1 = f1_score(all_labels, all_preds, average="macro")

    # ----- validation -----
    model.eval()
    val_loss_tot = 0.0
    v_preds, v_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"])
            v_loss = criterion(logits, batch["labels"])
            val_loss_tot += v_loss.item() * batch["labels"].size(0)
            v_preds.extend(logits.argmax(1).cpu().numpy())
            v_labels.extend(batch["labels"].cpu().numpy())
    val_loss = val_loss_tot / len(val_loader.dataset)
    val_f1 = f1_score(v_labels, v_preds, average="macro")
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macroF1 = {val_f1:.4f}"
    )

    # log
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_macro_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_macro_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["timestamps"].append(str(datetime.utcnow()))

# ----------------- TEST EVALUATION -----------------
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch["input_ids"])
        test_preds.extend(logits.argmax(1).cpu().numpy())
        test_labels.extend(batch["labels"].cpu().numpy())
test_f1 = f1_score(test_labels, test_preds, average="macro")
print(f"Test Macro-F1: {test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels

# ----------------- SAVE DATA -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
