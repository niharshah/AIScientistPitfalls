# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from collections import Counter

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"batch_size": {"SPR_BENCH": {}}}

# -----------------------  GPU / Device handling  ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------  Dataset loading  ---------------------------------
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

    def shape_weighted_accuracy(_, y_t, y_p):  # simple accuracy for synthetic task
        return sum(int(t == p) for t, p in zip(y_t, y_p)) / len(y_t)

    color_weighted_accuracy = shape_weighted_accuracy

# -----------------------  Vocabulary build  --------------------------------
train_seqs = spr["train"]["sequence"]
counter = Counter(tok for seq in train_seqs for tok in seq.split())
vocab = {"<PAD>": 0, "<UNK>": 1}
for tok in counter:
    vocab[tok] = len(vocab)
pad_idx, unk_idx = vocab["<PAD>"], vocab["<UNK>"]
num_classes = len(set(spr["train"]["label"]))


def encode(seq):
    return [vocab.get(tok, unk_idx) for tok in seq.split()]


# -----------------------  Torch Dataset ------------------------------------
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
    rawseq = [item["raw_seq"] for item in batch]
    return {
        "seq": seqs,
        "lengths": torch.tensor(lengths),
        "label": labels,
        "raw_seq": rawseq,
    }


train_ds, val_ds, test_ds = (
    SPRDataset(spr["train"]),
    SPRDataset(spr["dev"]),
    SPRDataset(spr["test"]),
)


# -----------------------  Model --------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, classes, pad):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim * 2, classes)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(h)


# -----------------------  Evaluation helper  -------------------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    loss_total, preds, labels, seqs = 0.0, [], [], []
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
            p = out.argmax(-1).cpu().tolist()
            preds.extend(p)
            labels.extend(batch_t["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
    avg_loss = loss_total / len(labels)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    hwa = 2 * swa * cwa / (swa + cwa + 1e-8)
    return avg_loss, swa, cwa, hwa, preds, labels


# -----------------------  Hyper-parameter sweep ----------------------------
BATCH_SIZES = [16, 32, 64, 128, 256]
EPOCHS = 5

for bs in BATCH_SIZES:
    print(f"\n=== Training with batch_size={bs} ===")
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

    # Model / optimiser init
    model = GRUClassifier(len(vocab), 32, 64, num_classes, pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Logs
    run_log = {
        "batch_size": bs,
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": []},
        "test_metrics": None,
        "predictions": None,
        "ground_truth": None,
        "timestamps": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
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
            running_loss += loss.item() * len(batch_t["label"])
        train_loss = running_loss / len(train_ds)
        val_loss, swa, cwa, hwa, _, _ = evaluate(model, val_loader)
        print(f"  Ep{epoch} | val_loss {val_loss:.4f} | HWA {hwa:.4f}")
        run_log["losses"]["train"].append(train_loss)
        run_log["losses"]["val"].append(val_loss)
        run_log["metrics"]["val"].append(hwa)
        run_log["timestamps"].append(time.time())

    # Final test evaluation
    test_loss, swa, cwa, hwa, preds, labels = evaluate(model, test_loader)
    print(f"Test HWA={hwa:.4f}")
    run_log["test_metrics"] = hwa
    run_log["predictions"] = preds
    run_log["ground_truth"] = labels

    # Store in experiment_data
    experiment_data["batch_size"]["SPR_BENCH"][bs] = run_log

    # Confusion matrix plot
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"CM bs={bs}")
        plt.xlabel("Pred")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"SPR_confusion_bs{bs}.png"))
        plt.close()
    except Exception:
        pass

# -----------------------  Save experiment data -----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data to working_dir.")
