import os, pathlib, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from typing import List, Dict
from collections import Counter

# ------------------------------------------------------------------
#  mandatory working directory & device handling
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
#  try to load real SPR_BENCH, else create toy synthetic dataset
# ------------------------------------------------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr_data = load_spr_bench(DATA_PATH)
except Exception as e:
    print("Could not load SPR_BENCH, falling back to synthetic data ->", e)

    def random_seq(n_tokens=8):
        shapes = list("ABCDE")
        colors = list("XYZUV")
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(n_tokens)
        )

    def make_split(n_rows):
        return [
            {
                "id": i,
                "sequence": random_seq(random.randint(5, 12)),
                "label": random.randint(0, 3),
            }
            for i in range(n_rows)
        ]

    spr_data = {
        "train": make_split(800),
        "dev": make_split(200),
        "test": make_split(200),
    }

    # define simple versions of metrics if SPR utility unavailable
    def shape_weighted_accuracy(seq, y_true, y_pred):
        return f1_score(y_true, y_pred, average="micro")

    color_weighted_accuracy = shape_weighted_accuracy


# ------------------------------------------------------------------
#  build vocabulary & label mapping
# ------------------------------------------------------------------
def extract_tokens(sequence: str) -> List[str]:
    return sequence.strip().split()


token_counter = Counter()
for row in spr_data["train"]:
    token_counter.update(extract_tokens(row["sequence"]))
vocab = {
    tok: i + 1 for i, (tok, _) in enumerate(token_counter.most_common())
}  # 0 is PAD
vocab["<UNK>"] = len(vocab) + 1
pad_id = 0
unk_id = vocab["<UNK>"]
num_classes = len({row["label"] for row in spr_data["train"]})
print(f"Vocab size={len(vocab)}  num_classes={num_classes}")


def encode(sequence: str) -> List[int]:
    return [vocab.get(tok, unk_id) for tok in extract_tokens(sequence)]


# label mapping (ensure 0..C-1)
labels_sorted = sorted(
    {row["label"] for split in ["train", "dev", "test"] for row in spr_data[split]}
)
label2id = {lab: i for i, lab in enumerate(labels_sorted)}
for split in ["train", "dev", "test"]:
    for row in spr_data[split]:
        row["label"] = label2id[row["label"]]


# ------------------------------------------------------------------
#  dataset / dataloader
# ------------------------------------------------------------------
class SPRDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "input_ids": torch.tensor(encode(row["sequence"]), dtype=torch.long),
            "labels": torch.tensor(row["label"], dtype=torch.long),
            "sequence": row["sequence"],
        }


def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    seqs = [item["sequence"] for item in batch]
    for i, item in enumerate(batch):
        input_ids[i, : len(item["input_ids"])] = item["input_ids"]
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "sequence": seqs,
    }


batch_size = 64
train_loader = DataLoader(
    SPRDataset(spr_data["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
dev_loader = DataLoader(
    SPRDataset(spr_data["dev"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    SPRDataset(spr_data["test"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


# ------------------------------------------------------------------
#  model
# ------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        return self.fc(h.squeeze(0))


model = GRUClassifier(
    len(vocab), emb_dim=64, hidden_dim=128, num_classes=num_classes, pad_idx=pad_id
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------------
#  experiment data store
# ------------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------------------------
#  training loop
# ------------------------------------------------------------------
def evaluate(loader):
    model.eval()
    losses, ys, yhats, seqs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            losses.append(loss.item() * len(batch["labels"]))
            preds = logits.argmax(-1).cpu().numpy()
            yhats.extend(preds)
            ys.extend(batch["labels"].cpu().numpy())
            seqs.extend(batch["sequence"])
    avg_loss = sum(losses) / len(loader.dataset)
    macroF1 = f1_score(ys, yhats, average="macro")
    swa = shape_weighted_accuracy(seqs, ys, yhats)
    cwa = color_weighted_accuracy(seqs, ys, yhats)
    return avg_loss, macroF1, swa, cwa, ys, yhats


num_epochs = 5
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch["labels"])
    train_loss = epoch_loss / len(train_loader.dataset)
    val_loss, val_f1, val_swa, val_cwa, _, _ = evaluate(dev_loader)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  validation_loss = {val_loss:.4f}  MacroF1={val_f1:.4f}  SWA={val_swa:.4f}  CWA={val_cwa:.4f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        (epoch, None)
    )  # placeholder
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, val_f1))

# ------------------------------------------------------------------
#  final evaluation on test set
# ------------------------------------------------------------------
test_loss, test_f1, test_swa, test_cwa, ys, yhats = evaluate(test_loader)
print(
    f"TEST  loss={test_loss:.4f}  MacroF1={test_f1:.4f}  SWA={test_swa:.4f}  CWA={test_cwa:.4f}"
)

experiment_data["SPR_BENCH"]["predictions"] = yhats
experiment_data["SPR_BENCH"]["ground_truth"] = ys
experiment_data["SPR_BENCH"]["final_metrics"] = {
    "MacroF1": test_f1,
    "SWA": test_swa,
    "CWA": test_cwa,
}

# ------------------------------------------------------------------
#  save everything
# ------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data to", os.path.join(working_dir, "experiment_data.npy"))
