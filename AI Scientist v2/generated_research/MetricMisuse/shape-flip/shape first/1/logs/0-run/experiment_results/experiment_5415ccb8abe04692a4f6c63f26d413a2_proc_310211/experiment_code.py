import os, pathlib, time, random, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict
from typing import List, Dict

# ----------------------------------------------------------------------
# house-keeping / caching
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "val_loss": [], "ZSRTA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ----------------------------------------------------------------------
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------------------------------------------------
# util from prompt ------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ----------------------------------------------------------------------
# load data -------------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ----------------------------------------------------------------------
# vocab + label mapping -------------------------------------------------
def build_vocab(dataset) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print(f"Vocab size: {len(vocab)}")


def encode_seq(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]


# label ids
train_labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(train_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print(f"#seen rule labels: {num_labels}")


# ----------------------------------------------------------------------
# Torch Dataset ---------------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab, label2id, train_mode=True):
        self.seq_enc = [encode_seq(s, vocab) for s in split["sequence"]]
        self.labels = split["label"]
        self.train_mode = train_mode
        self.label2id = label2id

    def __len__(self):
        return len(self.seq_enc)

    def __getitem__(self, idx):
        x = torch.tensor(self.seq_enc[idx], dtype=torch.long)
        if self.train_mode:
            y = torch.tensor(self.label2id[self.labels[idx]], dtype=torch.long)
            return {"input": x, "label": y}
        else:
            return {"input": x, "label_str": self.labels[idx]}


def collate(batch):
    xs = [b["input"] for b in batch]
    lens = [len(x) for x in xs]
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    out = {"input": xs_pad, "lengths": torch.tensor(lens, dtype=torch.long)}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    else:
        out["label_str"] = [b["label_str"] for b in batch]
    return out


train_ds = SPRTorchDataset(spr["train"], vocab, label2id, True)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id, True)
test_ds = SPRTorchDataset(spr["test"], vocab, label2id, False)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ----------------------------------------------------------------------
# Model ----------------------------------------------------------------
class SimpleSPRModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x, lengths):
        e = self.emb(x)  # (B,L,D)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h: (2, B, H)
        h_cat = torch.cat([h[0], h[1]], dim=-1)  # (B, 2H)
        logits = self.lin(h_cat)
        return logits


model = SimpleSPRModel(len(vocab), 64, 128, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------------------------------------------------------------
# helpers ---------------------------------------------------------------
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, total_ok, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            inp = batch["input"].to(device)
            lens = batch["lengths"].to(device)
            lbl = batch["label"].to(device)
            logits = model(inp, lens)
            loss = criterion(logits, lbl)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * inp.size(0)
            preds = logits.argmax(1)
            total_ok += (preds == lbl).sum().item()
            total += inp.size(0)
    return total_loss / total, total_ok / total


# ----------------------------------------------------------------------
# training loop ---------------------------------------------------------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    val_loss, val_acc = run_epoch(dev_loader, False)

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(tr_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}"
    )

# ----------------------------------------------------------------------
# evaluation on test ----------------------------------------------------
model.eval()
all_preds, all_labels, all_seqs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        inp = batch["input"].to(device)
        lens = batch["lengths"].to(device)
        logits = model(inp, lens)
        preds = logits.argmax(1).cpu().tolist()
        label_strs = batch["label_str"]
        all_preds.extend([id2label.get(p, "UNK") for p in preds])
        all_labels.extend(label_strs)
        all_seqs.extend(
            [
                " ".join([list(vocab.keys())[tok] for tok in seq.tolist() if tok != 0])
                for seq in batch["input"]
            ]
        )

# compute metrics -------------------------------------------------------
overall_acc = np.mean([p == t for p, t in zip(all_preds, all_labels)])
swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)

# ZSRTA
seen_rules = set(train_labels)
zs_indices = [i for i, lbl in enumerate(all_labels) if lbl not in seen_rules]
if zs_indices:
    zs_acc = np.mean([all_preds[i] == all_labels[i] for i in zs_indices])
else:
    zs_acc = float("nan")

experiment_data["SPR_BENCH"]["metrics"]["ZSRTA"].append(zs_acc)
experiment_data["SPR_BENCH"]["predictions"] = all_preds
experiment_data["SPR_BENCH"]["ground_truth"] = all_labels

print(
    f"\nTEST Acc: {overall_acc:.4f} | SWA: {swa:.4f} | CWA: {cwa:.4f} | ZSRTA: {zs_acc:.4f}"
)

# ----------------------------------------------------------------------
# save artefacts --------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {working_dir}")
