import os, pathlib, random, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from collections import Counter

# -----------------------  Hyper-parameter sweep ----------------------------
LR_CANDIDATES = [3e-4, 5e-4, 1e-3, 2e-3]
EPOCHS = 5

# -----------------------  Experiment bookkeeping ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"lr_tuning": {}}

# -----------------------  Device ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------  Load / build dataset -----------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy

    DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "./SPR_BENCH"))
    spr = load_spr_bench(DATA_PATH)
except Exception:
    print("SPR_BENCH not found, generating synthetic data…")
    shapes, colours = ["A", "B", "C"], ["r", "g", "b"]

    def synth(n):
        seqs, labels = [], []
        for _ in range(n):
            length = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colours) for _ in range(length)
            )
            label = int(any(tok[0] == "A" for tok in seq.split()))
            seqs.append(seq)
            labels.append(label)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr = {"train": synth(2000), "dev": synth(300), "test": synth(300)}

    def shape_weighted_accuracy(seqs, y_t, y_p):
        return sum(int(t == p) for t, p in zip(y_t, y_p)) / len(y_t)

    color_weighted_accuracy = shape_weighted_accuracy

# -----------------------  Vocab -------------------------------------------
train_seqs = spr["train"]["sequence"]
counter = Counter(tok for seq in train_seqs for tok in seq.split())
vocab = {"<PAD>": 0, "<UNK>": 1}
for tok in counter:
    vocab[tok] = len(vocab)
pad_idx, unk_idx = vocab["<PAD>"], vocab["<UNK>"]
num_classes = len(set(spr["train"]["label"]))


def encode(seq):
    return [vocab.get(tok, unk_idx) for tok in seq.split()]


# -----------------------  Dataset & Loader --------------------------------
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
    lengths = [len(item["seq"]) for item in batch]
    maxlen = max(lengths)
    seqs = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        seqs[i, : lengths[i]] = item["seq"]
    labels = torch.stack([item["label"] for item in batch])
    return {
        "seq": seqs,
        "lengths": torch.tensor(lengths),
        "label": labels,
        "raw_seq": [b["raw_seq"] for b in batch],
    }


train_ds, val_ds, test_ds = (
    SPRDataset(spr["train"]),
    SPRDataset(spr["dev"]),
    SPRDataset(spr["test"]),
)
train_loader = DataLoader(train_ds, 64, True, collate_fn=collate)
val_loader = DataLoader(val_ds, 128, False, collate_fn=collate)
test_loader = DataLoader(test_ds, 128, False, collate_fn=collate)


# -----------------------  Model definition --------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h)


# -----------------------  Evaluation routine ------------------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total_loss, preds, labels, seqs = 0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            out = model(batch["seq"], batch["lengths"])
            loss = criterion(out, batch["label"])
            total_loss += loss.item() * len(batch["label"])
            pr = out.argmax(1).cpu().tolist()
            preds += pr
            labels += batch["label"].cpu().tolist()
            seqs += batch["raw_seq"]
    loss_avg = total_loss / len(labels)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    hwa = 2 * swa * cwa / (swa + cwa + 1e-8)
    return loss_avg, swa, cwa, hwa, preds, labels


# -----------------------  LR sweep ----------------------------------------
best_hwa, best_state, best_lr = -1, None, None
for lr in LR_CANDIDATES:
    print(f"\n=== Training with learning rate {lr} ===")
    model = GRUClassifier(len(vocab), 32, 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    key = f"lr_{lr}"
    experiment_data["lr_tuning"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
    }
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optimizer.zero_grad()
            out = model(batch["seq"], batch["lengths"])
            loss = criterion(out, batch["label"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch["label"])
        train_loss = epoch_loss / len(train_ds)
        val_loss, swa, cwa, hwa, _, _ = evaluate(model, val_loader)
        print(f"Epoch {epoch}: val_loss={val_loss:.4f} HWA={hwa:.4f}")
        # log
        experiment_data["lr_tuning"][key]["losses"]["train"].append(train_loss)
        experiment_data["lr_tuning"][key]["losses"]["val"].append(val_loss)
        experiment_data["lr_tuning"][key]["metrics"]["val"].append(hwa)
        experiment_data["lr_tuning"][key]["timestamps"].append(time.time())
        # track best model
        if hwa > best_hwa:
            best_hwa, best_state, best_lr = hwa, model.state_dict(), lr

print(f"\nBest LR {best_lr} with validation HWA {best_hwa:.4f}")

# -----------------------  Test evaluation ---------------------------------
best_model = GRUClassifier(len(vocab), 32, 64, num_classes).to(device)
best_model.load_state_dict(best_state)
test_loss, swa, cwa, hwa, preds, labels = evaluate(best_model, test_loader)
print(f"Test  -> loss:{test_loss:.4f} SWA:{swa:.4f} CWA:{cwa:.4f} HWA:{hwa:.4f}")

# store predictions & gt for best lr
key = f"lr_{best_lr}"
experiment_data["lr_tuning"][key]["predictions"] = preds
experiment_data["lr_tuning"][key]["ground_truth"] = labels
experiment_data["lr_tuning"][key]["metrics"]["test"] = hwa

# -----------------------  Confusion matrix --------------------------------
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

# -----------------------  Save experiment data ----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data to", working_dir)
