import os, sys, pathlib, random, time, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import Dataset as HFDataset, DatasetDict

try:
    from SPR import load_spr_bench  # if SPR.py is on PYTHONPATH
except Exception:
    load_spr_bench = None  # fallback later

# ----------------- housekeeping -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------- experiment_data skeleton ---------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------------- helpers -----------------------
SHAPES = ["S", "T", "C", "H"]  # Square, Triangle, Circle, Hex …
COLORS = ["r", "g", "b", "y"]


def synthetic_spr_split(n_rows: int, seed: int = 0):
    random.seed(seed)
    seqs, labels, ids = [], [], []
    for i in range(n_rows):
        length = random.randint(4, 12)
        tokens = [random.choice(SHAPES) + random.choice(COLORS) for _ in range(length)]
        seq = " ".join(tokens)
        # synthetic "rule": (n_shapes_mod_3, n_colors_mod_3) compressed as a string
        n_shape_types = len(set(t[0] for t in tokens))
        n_color_types = len(set(t[1] for t in tokens))
        label = f"{n_shape_types%3}_{n_color_types%3}"
        ids.append(f"syn_{i:06d}")
        seqs.append(seq)
        labels.append(label)
    return {"id": ids, "sequence": seqs, "label": labels}


def get_dataset(
    root_path: str = "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/",
) -> DatasetDict:
    spr_path = pathlib.Path(root_path)
    if load_spr_bench and spr_path.exists():
        print("Loading real SPR_BENCH …")
        return load_spr_bench(spr_path)
    # fallback synthetic
    print("Real SPR_BENCH not found, generating synthetic data …")
    train = synthetic_spr_split(6000, seed=1)
    dev = synthetic_spr_split(2000, seed=2)
    test = synthetic_spr_split(3000, seed=3)
    return DatasetDict(
        {
            "train": HFDataset.from_dict(train),
            "dev": HFDataset.from_dict(dev),
            "test": HFDataset.from_dict(test),
        }
    )


# ---------------- tokeniser / vocab --------------
PAD, UNK = "<pad>", "<unk>"


def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    for s in seqs:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode_seq(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab[UNK]) for tok in seq.strip().split()]


# --------------- torch Dataset -------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset, vocab, label2id):
        self.data = hf_dataset
        self.vocab = vocab
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[int(idx)]
        seq_ids = torch.tensor(
            encode_seq(sample["sequence"], self.vocab), dtype=torch.long
        )
        label_id = torch.tensor(self.label2id[sample["label"]], dtype=torch.long)
        return {"input_ids": seq_ids, "label": label_id}


def collate_fn(batch):
    # pad to max length in batch
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        if pad_len:
            ids = torch.cat([ids, torch.full((pad_len,), 0, dtype=torch.long)])
        input_ids.append(ids)
        labels.append(item["label"])
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    return {"input_ids": input_ids, "label": labels}


# ----------------- model ------------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hid_dim * 2, num_classes)

    def forward(self, x):
        # x -> (B, L)
        emb = self.embedding(x)  # (B,L,E)
        _, (h, _) = self.lstm(emb)  # h: (2, B, H)
        h_cat = torch.cat([h[0], h[1]], dim=-1)  # (B, 2H)
        out = self.fc(h_cat)  # (B, C)
        return out


# ---------------- training helpers --------------
def accuracy(preds, labels):
    return (preds == labels).sum().item() / len(labels)


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_acc, total = 0, 0, 0
    all_preds, all_labels = [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        preds = logits.argmax(1)
        total_loss += loss.item() * len(batch["label"])
        total_acc += (preds == batch["label"]).sum().item()
        total += len(batch["label"])
        all_preds.append(preds.cpu())
        all_labels.append(batch["label"].cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return total_loss / total, total_acc / total, all_preds, all_labels


# ------------- main flow ------------------------
spr = get_dataset()
all_train_seqs = spr["train"]["sequence"]
vocab = build_vocab(all_train_seqs)
label_set = sorted(set(spr["train"]["label"]))
label2id = {lbl: i for i, lbl in enumerate(label_set)}
num_classes = len(label2id)
print(f"Vocab size={len(vocab)}, #classes={num_classes}")

train_ds = SPRTorchDataset(spr["train"], vocab, label2id)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id)
test_ds = SPRTorchDataset(
    spr["test"], vocab, label2id
)  # note: may include unseen labels

batch_size = 128
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

model = BiLSTMEncoder(
    len(vocab), emb_dim=32, hid_dim=64, num_classes=num_classes, pad_idx=0
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, running_acc, seen = 0, 0, 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        preds = logits.argmax(1)
        running_loss += loss.item() * len(batch["label"])
        running_acc += (preds == batch["label"]).sum().item()
        seen += len(batch["label"])
    train_loss = running_loss / seen
    train_acc = running_acc / seen

    val_loss, val_acc, _, _ = evaluate(model, dev_loader, criterion)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
        f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
    )

# ---------------- final evaluation ----------------
test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels

train_labels_set = set(spr["train"]["label"])
test_unseen_mask = np.array(
    [lbl not in train_labels_set for lbl in spr["test"]["label"]]
)
if test_unseen_mask.any():
    ura = accuracy(test_preds[test_unseen_mask], test_labels[test_unseen_mask])
else:
    ura = float("nan")  # no unseen rules in synthetic tiny split

print(f"\nTest loss = {test_loss:.4f}, Test ACC = {test_acc:.4f}")
print(f"Unseen-Rule Accuracy (URA) = {ura:.4f}")

# ------------- save everything ------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
