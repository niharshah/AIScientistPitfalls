import os, pathlib, numpy as np, torch, torch.nn as nn, random, math, time
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict


# ------------------- utility --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def find_spr_bench() -> pathlib.Path:
    candidates = [
        os.environ.get("SPR_DATA_PATH"),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in candidates:
        if p and pathlib.Path(p).expanduser().joinpath("train.csv").exists():
            return pathlib.Path(p).expanduser().resolve()
    raise FileNotFoundError("SPR_BENCH not found.")


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ----------------- load dataset -----------------
def load_spr_bench(root) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


spr = load_spr_bench(DATA_PATH)

# ----------------- vocabulary -------------------
all_tokens = set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1


def encode(seq: str):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, num_classes={num_classes}")


# ------------------ dataset ---------------------
class SPRTorchSet(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.encoded = [encode(s) for s in self.seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encoded[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad = lambda t: (
        torch.cat([t, torch.full((maxlen - len(t),), PAD_ID, dtype=torch.long)])
        if len(t) < maxlen
        else t
    )
    return {
        "input_ids": torch.stack([pad(b["input_ids"]) for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


ds_train = SPRTorchSet(spr["train"])
ds_dev = SPRTorchSet(spr["dev"])


# ------------- metrics helpers ------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return (
        sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / sum(w)
        if sum(w) > 0
        else 0.0
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return (
        sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / sum(w)
        if sum(w) > 0
        else 0.0
    )


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if swa + cwa > 0 else 0.0


# ------------------- model ----------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hidden=128, num_cls=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_cls)

    def forward(self, x):
        emb = self.embed(x)
        lengths = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = torch.cat([h_n[-2], h_n[-1]], 1)
        return self.fc(out)


# -------------- hyperparameter sweep ------------
BATCH_SIZES = [64, 128, 256, 512]
EPOCHS = 6
experiment_data = {"batch_size": {"SPR_BENCH": {}}}

for bs in BATCH_SIZES:
    print(f"\n=== Training with batch_size={bs} ===")
    train_loader = DataLoader(
        ds_train, batch_size=bs, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        ds_dev, batch_size=512, shuffle=False, collate_fn=collate_fn
    )

    model = BiLSTMClassifier(vocab_size, num_cls=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batch = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        train_loss = total_loss / n_batch
        exp["losses"]["train"].append((epoch, train_loss))

        model.eval()
        val_loss_tot = 0.0
        nb = 0
        all_preds, all_labels, all_seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                val_loss_tot += loss.item()
                nb += 1
                preds = logits.argmax(-1).cpu().tolist()
                labels = batch["label"].cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_seqs.extend(batch["raw_seq"])
        val_loss = val_loss_tot / nb
        exp["losses"]["val"].append((epoch, val_loss))

        swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
        cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        exp["metrics"]["val"].append((epoch, swa, cwa, hwa))

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )

    exp["predictions"] = all_preds
    exp["ground_truth"] = all_labels
    experiment_data["batch_size"]["SPR_BENCH"][str(bs)] = exp

# ----------------- save results -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
