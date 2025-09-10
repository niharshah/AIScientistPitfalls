import os, time, random, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# I/O & device set-up ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Reproducibility ---------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# Data helpers ------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.strip().split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ------------------------------------------------------------------
# Load SPR_BENCH ----------------------------------------------------
DATA_PATH = pathlib.Path(
    "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
)  # adapt if needed
spr_raw = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr_raw.items()})

# ------------------------------------------------------------------
# Zero-shot split: withhold a subset of rule labels -----------------
all_labels = sorted(set(spr_raw["train"]["label"]))
# at least 1 unseen label (20% of total rounded up)
n_unseen = max(1, int(0.2 * len(all_labels)))
unseen_labels = set(random.sample(all_labels, n_unseen))
print("Unseen labels (held out from training):", unseen_labels)


def filter_by_labels(dataset, allowed_labels):
    idx = [i for i, l in enumerate(dataset["label"]) if l in allowed_labels]
    return dataset.select(idx)


seen_labels = [l for l in all_labels if l not in unseen_labels]

spr = DatasetDict()
spr["train"] = filter_by_labels(spr_raw["train"], seen_labels)
spr["dev"] = filter_by_labels(spr_raw["dev"], seen_labels)
spr["test"] = spr_raw["test"]  # keep full test set

print(
    {k: len(v) for k, v in spr.items()},
    "| #seen:",
    len(seen_labels),
    "#unseen:",
    len(unseen_labels),
)


# ------------------------------------------------------------------
# Vocabulary & label mapping ---------------------------------------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
id2tok = {i: t for t, i in vocab.items()}

label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print("Vocab size:", len(vocab), "| #labels:", num_labels)


# ------------------------------------------------------------------
# PyTorch Dataset ---------------------------------------------------
def encode_seq(seq):
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, training=True):
        self.training = training
        self.seqs = hf_split["sequence"]
        self.seq_enc = [encode_seq(s) for s in self.seqs]
        if training:
            self.labels = [label2id[l] for l in hf_split["label"]]
        else:
            self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seq_enc)

    def __getitem__(self, idx):
        seq_ids = torch.tensor(self.seq_enc[idx], dtype=torch.long)
        sym_feat = torch.tensor(
            [
                count_shape_variety(self.seqs[idx]),
                count_color_variety(self.seqs[idx]),
                len(self.seq_enc[idx]),
            ],
            dtype=torch.float,
        )
        item = {
            "input": seq_ids,
            "sym": sym_feat,
            "length": torch.tensor(len(seq_ids), dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }
        if self.training:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            item["label_str"] = self.labels[idx]
        return item


def collate(batch):
    xs = [b["input"] for b in batch]
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    sym = torch.stack([b["sym"] for b in batch])
    out = {
        "input": xs_pad,
        "length": lens,
        "sym": sym,
        "raw_seq": [b["raw_seq"] for b in batch],
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    else:
        out["label_str"] = [b["label_str"] for b in batch]
    return out


train_ds = SPRTorchDataset(spr["train"], True)
dev_ds = SPRTorchDataset(spr["dev"], True)
test_ds = SPRTorchDataset(spr["test"], False)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ------------------------------------------------------------------
# Neuro-Symbolic model ---------------------------------------------
class NeuroSymbolicSPR(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.sym_mlp = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
        )
        self.out = nn.Linear(hid_dim * 2 + 16, num_labels)

    def forward(self, x, lens, sym):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        h_cat = torch.cat([h[0], h[1]], dim=-1)
        sym_feat = self.sym_mlp(sym)
        z = torch.cat([h_cat, sym_feat], dim=-1)
        return self.out(z)


model = NeuroSymbolicSPR(len(vocab), 64, 128, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------------------------------------------
# Train / eval utilities -------------------------------------------
def batch_to_device(batch):
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_ok = tot = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch_to_device(batch)
            logits = model(batch["input"], batch["length"], batch["sym"])
            if train:
                loss = criterion(logits, batch["label"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss = criterion(logits, batch["label"])
            tot_loss += loss.item() * batch["input"].size(0)
            preds = logits.argmax(1)
            tot_ok += (preds == batch["label"]).sum().item()
            tot += batch["input"].size(0)
    return tot_loss / tot, tot_ok / tot


@torch.no_grad()
def evaluate(loader):
    model.eval()
    preds, labels, seqs = [], [], []
    for batch in loader:
        batch_d = batch_to_device(batch)
        logits = model(batch_d["input"], batch_d["length"], batch_d["sym"])
        p_ids = logits.argmax(1).cpu().tolist()
        preds.extend([id2label[i] for i in p_ids])
        if "label_str" in batch:
            labels.extend(batch["label_str"])
        else:
            labels.extend([id2label[i.item()] for i in batch["label"]])
        seqs.extend(batch["raw_seq"])
    acc = np.mean([p == t for p, t in zip(preds, labels)])
    cwa = color_weighted_accuracy(seqs, labels, preds)
    unseen_idx = [i for i, l in enumerate(labels) if l in unseen_labels]
    ura = (
        np.mean([preds[i] == labels[i] for i in unseen_idx])
        if unseen_idx
        else float("nan")
    )
    return acc, cwa, ura


# ------------------------------------------------------------------
# Experiment container ---------------------------------------------
experiment_data = {
    "ZeroShotSPR": {
        "metrics": {
            "train_acc": [],
            "val_acc": [],
            "val_CWA": [],
            "URA": [],
            "val_loss": [],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------------------------------------------------------------
# Training loop -----------------------------------------------------
EPOCHS = 6
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(dev_loader, train=False)
    _, val_cwa, val_ura = evaluate(dev_loader)

    experiment_data["ZeroShotSPR"]["metrics"]["train_acc"].append(tr_acc)
    experiment_data["ZeroShotSPR"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["ZeroShotSPR"]["metrics"]["val_CWA"].append(val_cwa)
    experiment_data["ZeroShotSPR"]["metrics"]["URA"].append(val_ura)
    experiment_data["ZeroShotSPR"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["ZeroShotSPR"]["losses"]["train"].append(tr_loss)
    experiment_data["ZeroShotSPR"]["losses"]["val"].append(val_loss)
    experiment_data["ZeroShotSPR"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | CWA={val_cwa:.4f} | URA={val_ura:.4f}"
    )

# ------------------------------------------------------------------
# Final test evaluation --------------------------------------------
test_acc, test_cwa, test_ura = evaluate(test_loader)
print(f"\nTEST  Acc={test_acc:.4f}  CWA={test_cwa:.4f}  URA={test_ura:.4f}")

# store final predictions / ground truth
with torch.no_grad():
    preds, labels = [], []
    for batch in test_loader:
        batch_d = batch_to_device(batch)
        logits = model(batch_d["input"], batch_d["length"], batch_d["sym"])
        p = logits.argmax(1).cpu().tolist()
        preds.extend([id2label[idx] for idx in p])
        labels.extend(batch["label_str"])
experiment_data["ZeroShotSPR"]["predictions"] = preds
experiment_data["ZeroShotSPR"]["ground_truth"] = labels

# ------------------------------------------------------------------
# Save artefacts ----------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to {working_dir}/experiment_data.npy")
