import os, random, string, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ------------------------------ experiment bookkeeping
experiment_data = {
    "no_contrastive_pretraining": {
        "spr": {
            "losses": {"train": [], "val": []},
            "metrics": {"SWA": [], "CWA": [], "CompWA": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ------------------------------ misc paths / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------ dataset helpers
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def build_synthetic_dataset(n_tr=2000, n_dev=500, n_test=500, max_len=10):
    def _row():
        L = random.randint(4, max_len)
        seq, label = [], 0
        for _ in range(L):
            sh, co = random.choice("ABCDE"), random.choice("01234")
            seq.append(sh + co)
            label ^= (ord(sh) + int(co)) & 1
        return {
            "id": str(random.randint(0, 1e9)),
            "sequence": " ".join(seq),
            "label": label,
        }

    def _many(n):
        return [_row() for _ in range(n)]

    return DatasetDict(
        train=HFDataset.from_list(_many(n_tr)),
        dev=HFDataset.from_list(_many(n_dev)),
        test=HFDataset.from_list(_many(n_test)),
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
spr = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else build_synthetic_dataset()
print({k: len(v) for k, v in spr.items()})

# ------------------------------ vocab / encoding
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in ["train", "dev", "test"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx = vocab[PAD]
MAX_LEN = 40


def encode(seq, max_len=MAX_LEN):
    ids = [vocab.get(t, vocab[UNK]) for t in seq.split()][:max_len]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# ------------------------------ metrics
def count_shape_variety(sequence):
    return len({tok[0] for tok in sequence.split()})


def count_color_variety(sequence):
    return len({tok[1] for tok in sequence.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return (
        sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / sum(w) if sum(w) else 0.0
    )


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return (
        sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / sum(w) if sum(w) else 0.0
    )


def complexity_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return (
        sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / sum(w) if sum(w) else 0.0
    )


# ------------------------------ torch dataset for classification
class ClassificationSPRDataset(TorchDataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        r = self.ds[idx]
        return (
            torch.tensor(encode(r["sequence"]), dtype=torch.long),
            torch.tensor(r["label"], dtype=torch.long),
            r["sequence"],
        )


def collate_classification(batch):
    ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    seqs = [b[2] for b in batch]
    return {"input_ids": ids, "labels": labels, "sequence": seqs}


# ------------------------------ model
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)

    def forward(self, x):
        emb = self.emb(x)
        mask = (x != pad_idx).float().unsqueeze(-1)
        packed, _ = self.gru(emb)
        pooled = (packed * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return pooled  # B, 2*hid


class Classifier(nn.Module):
    def __init__(self, enc, num_cls=2):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(512, num_cls)

    def forward(self, x):
        rep = self.enc(x)
        return self.fc(rep)


# ------------------------------ fine-tune from scratch (no pretraining)
FINE_EPOCHS, BATCH_F = 5, 256
train_loader = DataLoader(
    ClassificationSPRDataset(spr["train"]),
    batch_size=BATCH_F,
    shuffle=True,
    collate_fn=collate_classification,
)
dev_loader = DataLoader(
    ClassificationSPRDataset(spr["dev"]),
    batch_size=BATCH_F,
    shuffle=False,
    collate_fn=collate_classification,
)

encoder = Encoder(len(vocab)).to(device)  # RANDOMLY INITIALISED
model = Classifier(encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("\n--- Supervised training from scratch ---")
for ep in range(1, FINE_EPOCHS + 1):
    # ---- training loop
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        ids, lbl = batch["input_ids"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(ids)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    tr_loss = total_loss / len(train_loader)
    experiment_data["no_contrastive_pretraining"]["spr"]["losses"]["train"].append(
        (ep, tr_loss)
    )

    # ---- validation
    model.eval()
    val_loss, seqs, preds, gts = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            ids, lbl = batch["input_ids"].to(device), batch["labels"].to(device)
            logits = model(ids)
            val_loss += criterion(logits, lbl).item()
            p = logits.argmax(-1).cpu().tolist()
            preds.extend(p)
            gts.extend(batch["labels"].tolist())
            seqs.extend(batch["sequence"])
    val_loss /= len(dev_loader)
    experiment_data["no_contrastive_pretraining"]["spr"]["losses"]["val"].append(
        (ep, val_loss)
    )

    SWA = shape_weighted_accuracy(seqs, gts, preds)
    CWA = color_weighted_accuracy(seqs, gts, preds)
    CompWA = complexity_weighted_accuracy(seqs, gts, preds)

    experiment_data["no_contrastive_pretraining"]["spr"]["metrics"]["SWA"].append(
        (ep, SWA)
    )
    experiment_data["no_contrastive_pretraining"]["spr"]["metrics"]["CWA"].append(
        (ep, CWA)
    )
    experiment_data["no_contrastive_pretraining"]["spr"]["metrics"]["CompWA"].append(
        (ep, CompWA)
    )
    experiment_data["no_contrastive_pretraining"]["spr"]["predictions"].append(
        (ep, preds)
    )
    experiment_data["no_contrastive_pretraining"]["spr"]["ground_truth"].append(
        (ep, gts)
    )

    print(
        f"Epoch {ep}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f} "
        f"SWA={SWA:.4f} CWA={CWA:.4f} CompWA={CompWA:.4f}"
    )

# ------------------------------ save results
out_file = os.path.join(working_dir, "experiment_data.npy")
np.save(out_file, experiment_data, allow_pickle=True)
print("Saved experiment data to", out_file)
