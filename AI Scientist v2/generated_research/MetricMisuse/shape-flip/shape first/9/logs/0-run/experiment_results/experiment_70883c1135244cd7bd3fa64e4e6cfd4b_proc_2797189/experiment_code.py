import os, pathlib, random, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from collections import Counter

# -------------------- experiment store -------------------------------------
experiment_data = {
    "gru_num_layers": {
        "SPR_BENCH": {
            "param_values": [],
            "metrics": {"train": [], "val": [], "test": []},
            "losses": {"train": [], "val": [], "test": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
            "best_param": None,
        }
    }
}

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- device ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- data --------------------------------------------------
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
            labels.append(int(any(tok[0] == "A" for tok in seq.split())))
            seqs.append(seq)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr = {"train": synth(2000), "dev": synth(300), "test": synth(300)}

    def shape_weighted_accuracy(seqs, y_t, y_p):
        return sum(int(t == p) for t, p in zip(y_t, y_p)) / len(y_t)

    color_weighted_accuracy = shape_weighted_accuracy

# -------------------- vocab -------------------------------------------------
train_seqs = spr["train"]["sequence"]
counter = Counter(tok for seq in train_seqs for tok in seq.split())
vocab = {"<PAD>": 0, "<UNK>": 1}
for tok in counter:
    vocab[tok] = len(vocab)
pad_idx, unk_idx = vocab["<PAD>"], vocab["<UNK>"]
num_classes = len(set(spr["train"]["label"]))


def encode(seq):
    return [vocab.get(tok, unk_idx) for tok in seq.split()]


# -------------------- dataset ----------------------------------------------
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
    lengths = [len(it["seq"]) for it in batch]
    maxlen = max(lengths)
    seqs = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, it in enumerate(batch):
        seqs[i, : lengths[i]] = it["seq"]
    labels = torch.stack([it["label"] for it in batch])
    return {
        "seq": seqs,
        "lengths": torch.tensor(lengths),
        "label": labels,
        "raw_seq": [it["raw_seq"] for it in batch],
    }


train_loader = DataLoader(SPRDataset(spr["train"]), 64, True, collate_fn=collate)
val_loader = DataLoader(SPRDataset(spr["dev"]), 128, False, collate_fn=collate)
test_loader = DataLoader(SPRDataset(spr["test"]), 128, False, collate_fn=collate)


# -------------------- model -------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, pad_idx
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h)


# -------------------- evaluation -------------------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, labels, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["seq"], batch["lengths"])
            loss = criterion(out, batch["label"])
            total_loss += loss.item() * len(batch["label"])
            p = out.argmax(-1).cpu().tolist()
            preds.extend(p)
            labels.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
    avg_loss = total_loss / len(labels)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    hwa = 2 * swa * cwa / (swa + cwa + 1e-8)
    return avg_loss, swa, cwa, hwa, preds, labels


# -------------------- hyperparameter sweep ---------------------------------
param_grid = [1, 2, 3]
EPOCHS, embed_dim, hidden_dim = 5, 32, 64
best_val_hwa, best_param, best_preds, best_labels = -1, None, None, None

for num_layers in param_grid:
    print(f"\n----- Training with num_layers={num_layers} -----")
    experiment_data["gru_num_layers"]["SPR_BENCH"]["param_values"].append(num_layers)
    model = GRUClassifier(
        len(vocab), embed_dim, hidden_dim, num_layers, num_classes, pad_idx
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses, val_hwas = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            out = model(batch["seq"], batch["lengths"])
            loss = criterion(out, batch["label"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch["label"])
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, _, _, val_hwa, _, _ = evaluate(model, val_loader, criterion)
        print(
            f"Epoch {epoch}: train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | HWA {val_hwa:.4f}"
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_hwas.append(val_hwa)
        experiment_data["gru_num_layers"]["SPR_BENCH"]["timestamps"].append(time.time())

    # log per-run data
    experiment_data["gru_num_layers"]["SPR_BENCH"]["losses"]["train"].append(
        train_losses
    )
    experiment_data["gru_num_layers"]["SPR_BENCH"]["losses"]["val"].append(val_losses)
    experiment_data["gru_num_layers"]["SPR_BENCH"]["metrics"]["val"].append(val_hwas)

    test_loss, swa, cwa, test_hwa, preds, labels = evaluate(
        model, test_loader, criterion
    )
    experiment_data["gru_num_layers"]["SPR_BENCH"]["losses"]["test"].append(test_loss)
    experiment_data["gru_num_layers"]["SPR_BENCH"]["metrics"]["test"].append(test_hwa)
    experiment_data["gru_num_layers"]["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["gru_num_layers"]["SPR_BENCH"]["ground_truth"].append(labels)

    if val_hwa > best_val_hwa:
        best_val_hwa, best_param = val_hwa, num_layers
        best_preds, best_labels = preds, labels

experiment_data["gru_num_layers"]["SPR_BENCH"]["best_param"] = best_param
print(f"\nBest num_layers={best_param} with val HWA={best_val_hwa:.4f}")

# -------------------- confusion matrix -------------------------------------
try:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(best_labels, best_preds)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (best num_layers={best_param})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_confusion.png"))
    plt.close()
except Exception as e:
    print("Confusion matrix failed:", e)

# -------------------- save --------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data to working_dir.")
