import os, pathlib, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# ---------- work dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- evaluation helpers (copied from spec) ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def hsca(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 2 * swa * cwa / (swa + cwa + 1e-8)


# ---------- load or create dataset ----------
def load_spr_dataset():
    default_path = pathlib.Path("./SPR_BENCH")
    if default_path.exists():
        print("Loading real SPR_BENCH …")

        def _load(split):
            return load_dataset(
                "csv",
                data_files=str(default_path / f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = DatasetDict(train=_load("train"), dev=_load("dev"), test=_load("test"))
    else:
        print("No SPR_BENCH found – generating tiny synthetic data …")

        def make_split(n):
            seqs, labels = [], []
            for i in range(n):
                length = random.randint(5, 12)
                seq = []
                for _ in range(length):
                    shape = random.choice("ABCD")
                    color = random.choice("WXYZ")
                    seq.append(shape + color)
                seqs.append(" ".join(seqs_tok := seq))
                # simple rule: label 1 if more A than B, else 0
                labels.append(
                    int(
                        sum(t[0] == "A" for t in seqs_tok)
                        > sum(t[0] == "B" for t in seqs_tok)
                    )
                )
            return Dataset.from_dict({"sequence": seqs, "label": labels})

        d = DatasetDict(train=make_split(200), dev=make_split(50), test=make_split(50))
    return d


dset = load_spr_dataset()

# ---------- vocabulary ----------
ALL_TOKENS = set()
for s in dset["train"]["sequence"]:
    ALL_TOKENS.update(s.split())
vocab = {tok: i + 2 for i, tok in enumerate(sorted(ALL_TOKENS))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
inv_vocab = {i: t for t, i in vocab.items()}


def encode(seq, max_len):
    ids = [vocab.get(tok, 1) for tok in seq.split()][:max_len]
    return ids + [0] * (max_len - len(ids))


MAX_LEN = max(len(s.split()) for s in dset["train"]["sequence"])
print("Vocab size:", len(vocab), "Max_len:", MAX_LEN)


# ---------- PyTorch dataset ----------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item["sequence"]
        label = item["label"]
        return {
            "input_ids": torch.tensor(encode(seq, MAX_LEN), dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "sequence": seq,
        }


def collate(batch):
    ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    seqs = [b["sequence"] for b in batch]
    return {"input_ids": ids.to(device), "label": labels.to(device), "sequence": seqs}


train_loader = DataLoader(
    SPRTorchDataset(dset["train"]), batch_size=64, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRTorchDataset(dset["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hid=128, num_classes=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid, batch_first=True)
        self.fc = nn.Linear(hid, num_classes)

    def forward(self, ids):
        x = self.emb(ids)
        _, (h, _) = self.lstm(x)
        logits = self.fc(h[-1])
        return logits


n_classes = len(set(dset["train"]["label"]))
model = SPRClassifier(len(vocab), num_classes=n_classes).to(device)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- experiment data structure ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_HSCA": [], "val_HSCA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------- training loop ----------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    # --- train ---
    model.train()
    train_loss, n = 0.0, 0
    for batch in train_loader:
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item() * len(batch["label"])
        n += len(batch["label"])
    train_loss /= n
    # collect train metric on subset for speed
    model.eval()
    tr_seqs, tr_true, tr_pred = [], [], []
    with torch.no_grad():
        for batch in random.sample(list(train_loader), min(5, len(train_loader))):
            logits = model(batch["input_ids"])
            pred = logits.argmax(-1).cpu().tolist()
            tr_pred.extend(pred)
            tr_true.extend(batch["label"].cpu().tolist())
            tr_seqs.extend(batch["sequence"])
    train_hsca = hsca(tr_seqs, tr_true, tr_pred)

    # --- validation ---
    val_loss, n = 0.0, 0
    val_seqs, val_true, val_pred = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * len(batch["label"])
            n += len(batch["label"])
            preds = logits.argmax(-1).cpu().tolist()
            val_pred.extend(preds)
            val_true.extend(batch["label"].cpu().tolist())
            val_seqs.extend(batch["sequence"])
    val_loss /= n
    swa = shape_weighted_accuracy(val_seqs, val_true, val_pred)
    cwa = color_weighted_accuracy(val_seqs, val_true, val_pred)
    val_hsca = hsca(val_seqs, val_true, val_pred)

    # --- log ---
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  SWA={swa:.3f}  CWA={cwa:.3f}  HSCA={val_hsca:.3f}"
    )
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_HSCA"].append(train_hsca)
    experiment_data["SPR_BENCH"]["metrics"]["val_HSCA"].append(val_hsca)

# save predictions of last epoch
experiment_data["SPR_BENCH"]["predictions"] = val_pred
experiment_data["SPR_BENCH"]["ground_truth"] = val_true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
