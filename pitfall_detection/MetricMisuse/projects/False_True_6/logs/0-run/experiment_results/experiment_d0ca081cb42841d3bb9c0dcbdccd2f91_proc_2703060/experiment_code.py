import os, pathlib, time, math, random, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ---------- folder & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment log ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_swa": [], "val_swa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
log = experiment_data["SPR_BENCH"]


# ---------- utility functions ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


# ---------- dataset & vocab ----------
DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1}
    for tok in sorted(set(itertools.chain.from_iterable(s.split() for s in seqs))):
        vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


label_set = sorted(set(spr["train"]["label"]))
lab2idx = {l: i for i, l in enumerate(label_set)}
idx2lab = {i: l for l, i in lab2idx.items()}
print("Labels:", label_set, " Vocab:", len(vocab))


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.lab = [lab2idx[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s = self.seq[idx]
        return {
            "input_ids": torch.tensor(encode(s), dtype=torch.long),
            "label": torch.tensor(self.lab[idx], dtype=torch.long),
            "sym_feat_raw": (
                count_shape_variety(s),
                count_color_variety(s),
                len(s.split()),
            ),
            "raw_seq": s,
        }


def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    feats = torch.tensor([b["sym_feat_raw"] for b in batch], dtype=torch.float32)
    pad_ids = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=vocab[PAD])
    # simple normalisation
    feats[:, 0] /= 10.0
    feats[:, 1] /= 10.0
    feats[:, 2] /= 20.0
    return {"input_ids": pad_ids, "sym_feats": feats, "labels": labels, "raw_seq": raw}


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr["train"]),
    SPRTorchDataset(spr["dev"]),
    SPRTorchDataset(spr["test"]),
)
train_loader = DataLoader(train_ds, 128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, 256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, 256, shuffle=False, collate_fn=collate)


# ---------- hybrid model ----------
class HybridClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, emb=32, hid=64, feat_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.feat_mlp = nn.Sequential(nn.Linear(3, feat_dim), nn.ReLU())
        self.fc = nn.Linear(hid + feat_dim, num_labels)

    def forward(self, input_ids, sym_feats):
        x = self.embedding(input_ids)
        _, h = self.gru(x)  # [1,B,hid]
        h = h.squeeze(0)  # [B,hid]
        f = self.feat_mlp(sym_feats)  # [B,feat_dim]
        logits = self.fc(torch.cat([h, f], dim=-1))
        return logits


model = HybridClassifier(len(vocab), len(label_set)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- evaluation ----------
def evaluate(loader):
    model.eval()
    total_loss, all_seq, all_true, all_pred = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            feats = batch["sym_feats"].to(device)
            lab = batch["labels"].to(device)
            logits = model(inp, feats)
            loss = criterion(logits, lab)
            total_loss += loss.item() * lab.size(0)
            pred = logits.argmax(-1)
            all_seq.extend(batch["raw_seq"])
            all_true.extend(lab.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    return total_loss / len(all_seq), swa, all_pred, all_true, all_seq


# ---------- training ----------
MAX_EPOCHS, PATIENCE = 20, 3
best_swa, best_state, no_imp = -1, None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    run_loss = 0.0
    for batch in train_loader:
        inp = batch["input_ids"].to(device)
        feats = batch["sym_feats"].to(device)
        lab = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(inp, feats)
        loss = criterion(logits, lab)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * lab.size(0)
    train_loss = run_loss / len(train_ds)
    val_loss, val_swa, _, _, _ = evaluate(dev_loader)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_SWA={val_swa:.3f}"
    )
    log["losses"]["train"].append(train_loss)
    log["losses"]["val"].append(val_loss)
    log["metrics"]["train_swa"].append(None)  # placeholder
    log["metrics"]["val_swa"].append(val_swa)
    log["timestamps"].append(time.time())
    # early stopping on swa
    if val_swa > best_swa + 1e-4:
        best_swa = val_swa
        best_state = model.state_dict()
        no_imp = 0
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print("Early stopping.")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# ---------- test ----------
test_loss, test_swa, preds, trues, seqs = evaluate(test_loader)
print(f"\nTEST: loss={test_loss:.4f}  SWA={test_swa:.3f}")

log["predictions"] = preds
log["ground_truth"] = trues
log["metrics"]["test"] = {"loss": test_loss, "swa": test_swa}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")

# quick bar plot
plt.figure(figsize=(4, 3))
plt.bar(["SWA"], [test_swa], color="cornflowerblue")
plt.ylim(0, 1)
plt.title("Test SWA")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "swa_bar.png"))
print("Plot saved.")
