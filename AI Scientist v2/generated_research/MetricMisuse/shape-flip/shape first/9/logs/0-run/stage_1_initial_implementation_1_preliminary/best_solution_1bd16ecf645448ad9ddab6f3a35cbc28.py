import os, pathlib, random, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from collections import Counter

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# -----------------------  GPU / Device handling  ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------  Dataset loading  ---------------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy

    DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "./SPR_BENCH"))
    spr = load_spr_bench(DATA_PATH)
except Exception as e:
    # fallback synthetic data
    print("SPR_BENCH not found, generating synthetic data…")
    shapes = ["A", "B", "C"]
    colours = ["r", "g", "b"]

    def synth(n):
        seqs, labels = [], []
        for _ in range(n):
            length = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colours) for _ in range(length)
            )
            lbl = int(
                any(tok[0] == "A" for tok in seq.split())
            )  # rule: contains shape A
            seqs.append(seq)
            labels.append(lbl)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr = {"train": synth(2000), "dev": synth(300), "test": synth(300)}

    def shape_weighted_accuracy(seqs, y_t, y_p):
        return sum(int(t == p) for t, p in zip(y_t, y_p)) / len(y_t)

    color_weighted_accuracy = shape_weighted_accuracy

# -----------------------  Vocabulary build  --------------------------------
train_seqs = (
    spr["train"]["sequence"]
    if isinstance(spr["train"], dict)
    else spr["train"]["sequence"]
)
counter = Counter(tok for seq in train_seqs for tok in seq.split())
vocab = {"<PAD>": 0, "<UNK>": 1}
for tok, _ in counter.items():
    vocab[tok] = len(vocab)
pad_idx = vocab["<PAD>"]
unk_idx = vocab["<UNK>"]
num_classes = (
    len(set(spr["train"]["label"]))
    if isinstance(spr["train"], dict)
    else len(set(spr["train"]["label"]))
)


def encode(seq):
    return [vocab.get(tok, unk_idx) for tok in seq.split()]


# -----------------------  Torch Dataset ------------------------------------
class SPRDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.ids = split["id"]
        self.seqs = split["sequence"]
        self.label = split["label"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.label[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    lengths = [len(item["seq"]) for item in batch]
    maxlen = max(lengths)
    seqs = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        seqs[i, : lengths[i]] = item["seq"]
    labels = torch.stack([item["label"] for item in batch])
    rawseq = [item["raw_seq"] for item in batch]
    return {
        "seq": seqs,
        "lengths": torch.tensor(lengths),
        "label": labels,
        "raw_seq": rawseq,
    }


train_ds = SPRDataset(spr["train"])
val_ds = SPRDataset(spr["dev"])
test_ds = SPRDataset(spr["test"])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate)


# -----------------------  Model --------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.classifier(h)


model = GRUClassifier(len(vocab), 32, 64, num_classes, pad_idx).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------  Training loop ------------------------------------
EPOCHS = 5


def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_seqs = []
    loss_total = 0
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch_t["seq"], batch_t["lengths"])
            loss_total += criterion(out, batch_t["label"]).item() * len(
                batch_t["label"]
            )
            preds = out.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_t["label"].cpu().tolist())
            all_seqs.extend(batch["raw_seq"])
    avg_loss = loss_total / len(all_labels)
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    hwa = 2 * swa * cwa / (swa + cwa + 1e-8)
    return avg_loss, swa, cwa, hwa, all_preds, all_labels, all_seqs


for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch_t["seq"], batch_t["lengths"])
        loss = criterion(out, batch_t["label"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch_t["label"])
    train_loss = epoch_loss / len(train_ds)
    val_loss, swa, cwa, hwa, _, _, _ = evaluate(val_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | HWA={hwa:.4f}")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(hwa)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

# -----------------------  Final evaluation ---------------------------------
test_loss, swa, cwa, hwa, preds, labels, seqs = evaluate(test_loader)
print(f"Test  -> loss:{test_loss:.4f} SWA:{swa:.4f} CWA:{cwa:.4f} HWA:{hwa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = labels
experiment_data["SPR_BENCH"]["metrics"]["test"] = hwa

# -----------------------  Confusion matrix plot ----------------------------
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
except:
    pass

# -----------------------  Save experiment data -----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data to working_dir.")
