import os, pathlib, random, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from collections import Counter

# ---------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# experiment data container -------------------------------------------------
experiment_data = {
    "num_training_epochs": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# -------------------------  Device -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------  Dataset loading --------------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy

    DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "./SPR_BENCH"))
    spr = load_spr_bench(DATA_PATH)
except Exception:
    print("SPR_BENCH not found, generating synthetic data…")
    shapes, colours = ["A", "B", "C"], ["r", "g", "b"]

    def _synth(n):
        seqs, labels = [], []
        for _ in range(n):
            length = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colours) for _ in range(length)
            )
            lbl = int(any(tok[0] == "A" for tok in seq.split()))
            seqs.append(seq)
            labels.append(lbl)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr = {"train": _synth(2000), "dev": _synth(300), "test": _synth(300)}

    def shape_weighted_accuracy(seqs, y_t, y_p):
        return sum(int(t == p) for t, p in zip(y_t, y_p)) / len(y_t)

    color_weighted_accuracy = shape_weighted_accuracy

# ------------------------- Vocabulary --------------------------------------
train_seqs = spr["train"]["sequence"]
counter = Counter(tok for s in train_seqs for tok in s.split())
vocab = {"<PAD>": 0, "<UNK>": 1}
for tok in counter:
    vocab[tok] = len(vocab)
pad_idx, unk_idx = vocab["<PAD>"], vocab["<UNK>"]
num_classes = len(set(spr["train"]["label"]))


def encode(seq):
    return [vocab.get(tok, unk_idx) for tok in seq.split()]


# ------------------------- Dataset / DataLoader ----------------------------
class SPRDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.ids, self.seqs, self.labels = (
            split["id"],
            split["sequence"],
            split["label"],
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    lengths = [len(b["seq"]) for b in batch]
    maxlen = max(lengths)
    seqs = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        seqs[i, : lengths[i]] = b["seq"]
    return {
        "seq": seqs,
        "lengths": torch.tensor(lengths),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


train_loader = DataLoader(SPRDataset(spr["train"]), 64, True, collate_fn=collate)
val_loader = DataLoader(SPRDataset(spr["dev"]), 128, False, collate_fn=collate)
test_loader = DataLoader(SPRDataset(spr["test"]), 128, False, collate_fn=collate)


# ------------------------- Model -------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hid, num_cls, pad):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad)
        self.gru = nn.GRU(embed_dim, hid, bidirectional=True, batch_first=True)
        self.lin = nn.Linear(2 * hid, num_cls)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.lin(h)


model = GRUClassifier(len(vocab), 32, 64, num_classes, pad_idx).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------- Helpers -----------------------------------------
def evaluate(loader):
    model.eval()
    tot_loss, preds, labels, seqs = 0, [], [], []
    with torch.no_grad():
        for batch in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(bt["seq"], bt["lengths"])
            tot_loss += criterion(out, bt["label"]).item() * len(bt["label"])
            pr = out.argmax(-1).cpu().tolist()
            preds.extend(pr)
            labels.extend(bt["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
    loss = tot_loss / len(labels)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    hwa = 2 * swa * cwa / (swa + cwa + 1e-8)
    return loss, swa, cwa, hwa, preds, labels


# ------------------------- Training loop -----------------------------------
MAX_EPOCHS, patience = 20, 3
best_hwa, no_improve = -1, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        bt = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(bt["seq"], bt["lengths"])
        loss = criterion(out, bt["label"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(bt["label"])
    train_loss = epoch_loss / len(train_loader.dataset)

    val_loss, swa, cwa, hwa, _, _ = evaluate(val_loader)
    print(f"Epoch {epoch:02d}: val_loss={val_loss:.4f} HWA={hwa:.4f}")

    ed = experiment_data["num_training_epochs"]["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["val"].append(hwa)
    ed["timestamps"].append(time.time())

    if hwa > best_hwa + 1e-4:
        best_hwa, no_improve = hwa, 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        no_improve += 1
    if no_improve >= patience:
        print("Early stopping triggered.")
        break

# ------------------------- Final evaluation --------------------------------
model.load_state_dict(torch.load(os.path.join(working_dir, "best_model.pt")))
test_loss, swa, cwa, hwa, preds, labels = evaluate(test_loader)
print(f"Test -> loss:{test_loss:.4f} SWA:{swa:.4f} CWA:{cwa:.4f} HWA:{hwa:.4f}")

ed = experiment_data["num_training_epochs"]["SPR_BENCH"]
ed["predictions"], ed["ground_truth"] = preds, labels
ed["metrics"]["test"] = hwa

# ------------------------- Confusion matrix --------------------------------
try:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_confusion.png"))
    plt.close()
except Exception as e:
    print("Could not plot confusion matrix:", e)

# ------------------------- Save experiment data ----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data to working_dir.")
