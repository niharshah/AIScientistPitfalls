import os, pathlib, time, math, itertools, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- house-keeping ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment log ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
log = experiment_data["SPR_BENCH"]


# ---------- metric helpers ----------
def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


# ---------- data loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    _ld = lambda fn: load_dataset(
        "csv", data_files=str(root / fn), split="train", cache_dir=".cache_dsets"
    )
    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- vocab ----------
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(dataset):
    tokset = set(itertools.chain.from_iterable(s.strip().split() for s in dataset))
    vocab = {PAD: 0, UNK: 1}
    for tok in sorted(tokset):
        vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
V = len(vocab)

labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}


def encode(seq):
    return [vocab.get(tok, 1) for tok in seq.split()]


# ---------- dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.lab = [lab2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s = self.seq[idx]
        return {
            "input_ids": torch.tensor(encode(s), dtype=torch.long),
            "label": torch.tensor(self.lab[idx], dtype=torch.long),
            "sym_feats": torch.tensor(
                [count_shape_variety(s), count_color_variety(s), len(s.split())],
                dtype=torch.float,
            ),
            "raw_seq": s,
        }


def collate(batch):
    ids = [b["input_ids"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    feats = torch.stack([b["sym_feats"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {"input_ids": padded, "sym_feats": feats, "labels": labels, "raw_seq": raw}


train_loader = DataLoader(SPRTorchDataset(spr["train"]), 128, True, collate_fn=collate)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"]), 256, False, collate_fn=collate)
test_loader = DataLoader(SPRTorchDataset(spr["test"]), 256, False, collate_fn=collate)


# ---------- model ----------
class HybridClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, emb=64, hid=128, feat_dim=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.seq_fc = nn.Sequential(
            nn.LayerNorm(hid), nn.Dropout(0.3), nn.ReLU(), nn.Linear(hid, hid)
        )
        self.feat_fc = nn.Sequential(nn.Linear(feat_dim, hid), nn.ReLU())
        self.out = nn.Linear(hid, num_labels)

    def forward(self, input_ids, sym_feats):
        x = self.embedding(input_ids)
        _, h = self.gru(x)  # (1,B,H)
        h = h.squeeze(0)
        seq_repr = self.seq_fc(h)
        feat_repr = self.feat_fc(sym_feats)
        joint = seq_repr + feat_repr  # fusion by addition
        return self.out(joint)


model = HybridClassifier(V, len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)


# ---------- evaluation ----------
def eval_loader(loader):
    model.eval()
    loss_tot, n, y_true, y_pred, raws = 0.0, 0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["sym_feats"])
            loss = criterion(logits, batch["labels"])
            loss_tot += loss.item() * len(batch["labels"])
            pred = logits.argmax(1)
            n += len(pred)
            y_true += batch["labels"].cpu().tolist()
            y_pred += pred.cpu().tolist()
            raws += batch["raw_seq"]
    swa = shape_weighted_accuracy(raws, y_true, y_pred)
    return loss_tot / n, swa, y_true, y_pred, raws


# ---------- training ----------
best_swa, best_state, patience, max_epochs = 0.0, None, 4, 25
for epoch in range(1, max_epochs + 1):
    model.train()
    tr_loss = 0.0
    m = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["sym_feats"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(batch["labels"])
        m += len(batch["labels"])
    scheduler.step()
    tr_loss /= m
    val_loss, val_swa, *_ = eval_loader(dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA = {val_swa:.3f}")
    log["losses"]["train"].append(tr_loss)
    log["losses"]["val"].append(val_loss)
    log["metrics"]["train"].append({"epoch": epoch})
    log["metrics"]["val"].append({"epoch": epoch, "swa": val_swa})
    log["timestamps"].append(time.time())

    if val_swa > best_swa + 1e-4:
        best_swa, best_state, wait = val_swa, model.state_dict(), 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# restore best model
if best_state:
    model.load_state_dict(best_state)

# ---------- final test ----------
test_loss, test_swa, y_t, y_p, raws = eval_loader(test_loader)
log["predictions"], log["ground_truth"] = y_p, y_t
log["metrics"]["test"] = {"loss": test_loss, "swa": test_swa}
print(f"\nTEST  | loss={test_loss:.4f}  SWA={test_swa:.3f}")

# ---------- save artefacts ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")

# bar plot
plt.figure(figsize=(4, 4))
plt.bar(["SWA"], [test_swa], color="steelblue")
plt.ylim(0, 1)
plt.title("Test SWA")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "swa_bar.png"))
print("Plot saved")
