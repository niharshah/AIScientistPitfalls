import os, pathlib, random, gc, time
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ---------------- I/O -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------- Device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- Load / synth data ---------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for s in ["train", "dev", "test"]:
        out[s] = _load(f"{s}.csv")
    return out


def make_synth_split(n):
    shapes, colors = "ABCD", "1234"
    return {
        "id": list(range(n)),
        "sequence": [
            " ".join(
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 9))
            )
            for _ in range(n)
        ],
        "label": [random.randint(0, 1) for _ in range(n)],
    }


data_root_candidates = [pathlib.Path(p) for p in ["SPR_BENCH", "./data/SPR_BENCH"]]
spr_bench = None
for p in data_root_candidates:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print(f"Loaded data from {p}")
        break
if spr_bench is None:
    print("Dataset not found, creating synthetic toy data.")
    spr_bench = DatasetDict(
        {
            "train": HFDataset.from_dict(make_synth_split(800)),
            "dev": HFDataset.from_dict(make_synth_split(200)),
            "test": HFDataset.from_dict(make_synth_split(200)),
        }
    )

tokenize = lambda s: s.strip().split()

# ------------- Vocabulary -------------
vocab = {"<PAD>": 0}
for split in spr_bench.values():
    for seq in split["sequence"]:
        for tok in tokenize(seq):
            if tok not in vocab:
                vocab[tok] = len(vocab)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ------------ Metrics -----------------
def count_shape_variety(seq):
    return len(set(t[0] for t in tokenize(seq)))


def count_color_variety(seq):
    return len(set(t[1] for t in tokenize(seq) if len(t) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


def scwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


# ------------- Datasets ---------------
class SPRClassifierDataset(Dataset):
    def __init__(self, hf_ds):
        self.seq = hf_ds["sequence"]
        self.lab = hf_ds["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        return tokenize(self.seq[i]), self.lab[i], self.seq[i]


def collate_classifier(batch):
    toks, labels, raw = zip(*batch)
    ids = [torch.tensor([vocab[t] for t in ts], dtype=torch.long) for ts in toks]
    lens = torch.tensor([len(i) for i in ids], dtype=torch.long)
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    return {
        "ids": ids,
        "lens": lens,
        "label": torch.tensor(labels, dtype=torch.long),
        "sequence": raw,
    }


# -------------- Model -----------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)

    def forward(self, ids, lens):
        x = self.emb(ids)
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(encoder.proj.out_features, num_classes)

    def forward(self, ids, lens):
        return self.head(self.enc(ids, lens))


# ---- Experiment tracking -------------
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "SCWA": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ----- Training settings --------------
sup_epochs_total = 6  # keep runtime modest
batch_train = 128
batch_val = 256
n_classes = len(set(spr_bench["train"]["label"]))

train_loader = DataLoader(
    SPRClassifierDataset(spr_bench["train"]),
    batch_size=batch_train,
    shuffle=True,
    collate_fn=collate_classifier,
)
val_loader = DataLoader(
    SPRClassifierDataset(spr_bench["dev"]),
    batch_size=batch_val,
    shuffle=False,
    collate_fn=collate_classifier,
)

encoder = Encoder(vocab_size).to(device)
model = Classifier(encoder, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------- Training loop ----------
for epoch in range(1, sup_epochs_total + 1):
    # ---- Train
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["ids"], batch["lens"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch["ids"].size(0)
    train_loss /= len(train_loader.dataset)

    # ---- Validate
    model.eval()
    val_loss = 0.0
    preds, gts, seqs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["ids"], batch["lens"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * batch["ids"].size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["sequence"])
    val_loss /= len(val_loader.dataset)

    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    scwa_val = scwa(seqs, gts, preds)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["SCWA"].append(scwa_val)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"SWA={swa:.3f} CWA={cwa:.3f} SCWA={scwa_val:.3f}"
    )

# ---------- Plot losses ---------------
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"], label="val")
plt.title("Loss vs Epochs (SPR_BENCH)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve_SPR_BENCH.png"))
plt.close()

# ---------- Save predictions ----------
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All results saved to", working_dir)

# Cleanup
torch.cuda.empty_cache()
gc.collect()
