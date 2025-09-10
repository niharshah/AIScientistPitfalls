import os, pathlib, time, math, random, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ---------- boilerplate paths / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data scaffold ------------
experiment_data = {
    "SPR_HYBRID": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
exp = experiment_data["SPR_HYBRID"]


# ---------- helper functions --------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_t, y_p)]
    return sum(correct) / max(sum(weights), 1)


# ---------- load SPR_BENCH ----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):  # treat csv as single split
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
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


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- vocab building ----------------------
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(dataset):
    vocab = {PAD: 0, UNK: 1}
    tokens = set(itertools.chain.from_iterable(seq.split() for seq in dataset))
    for tok in sorted(tokens):
        vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])


def encode(seq: str):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}


# ---------- Torch Dataset -----------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.raw_seq = hf_split["sequence"]
        self.label = [label2idx[l] for l in hf_split["label"]]
        self.shape_v = [count_shape_variety(s) for s in self.raw_seq]
        self.color_v = [count_color_variety(s) for s in self.raw_seq]

    def __len__(self):
        return len(self.raw_seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.raw_seq[idx]), dtype=torch.long),
            "label": torch.tensor(self.label[idx], dtype=torch.long),
            "shape_v": torch.tensor(self.shape_v[idx], dtype=torch.float32),
            "color_v": torch.tensor(self.color_v[idx], dtype=torch.float32),
            "raw": self.raw_seq[idx],
        }


def collate(batch):
    seqs = [b["input_ids"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=vocab[PAD])
    return {
        "input_ids": padded,
        "labels": torch.stack([b["label"] for b in batch]),
        "shape_v": torch.stack([b["shape_v"] for b in batch]),
        "color_v": torch.stack([b["color_v"] for b in batch]),
        "raw": [b["raw"] for b in batch],
    }


train_ds, dev_ds, test_ds = map(SPRDataset, (spr["train"], spr["dev"], spr["test"]))
train_loader = DataLoader(train_ds, 128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, 256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, 256, shuffle=False, collate_fn=collate)


# ---------- Hybrid Model ------------------------
class HybridClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim=32, hid=64, n_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True)
        self.symb_proj = nn.Linear(2, 16)
        self.classifier = nn.Linear(hid + 16, n_labels)

    def forward(self, input_ids, symb_feats):
        x = self.embedding(input_ids)
        _, h = self.gru(x)
        s = self.symb_proj(symb_feats)  # (B,16)
        concat = torch.cat([h.squeeze(0), s], dim=-1)
        return self.classifier(concat)


model = HybridClassifier(len(vocab), 32, 64, len(labels)).to(device)

# loss weighted by shape_variety to align with SWA
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- train / eval loops ------------------
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, w_sum = 0.0, 0.0
    for batch in loader:
        inp = batch["input_ids"].to(device)
        lab = batch["labels"].to(device)
        shape = batch["shape_v"].unsqueeze(1).to(device)
        symb = torch.cat([shape, batch["color_v"].unsqueeze(1).to(device)], dim=1)
        logits = model(inp, symb)
        sample_loss = criterion(logits, lab) * shape.squeeze(1)  # weight
        loss = sample_loss.sum() / shape.sum()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * shape.sum().item()
        w_sum += shape.sum().item()
    return total_loss / w_sum


def evaluate(loader):
    model.eval()
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            shape = batch["shape_v"].unsqueeze(1).to(device)
            symb = torch.cat([shape, batch["color_v"].unsqueeze(1).to(device)], dim=1)
            logits = model(inp, symb)
            pred = logits.argmax(-1)
            all_seq.extend(batch["raw"])
            all_true.extend(lab.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    return swa, all_pred, all_true, all_seq


# ---------- training loop with early stop -------
MAX_EPOCHS, PATIENCE = 20, 3
best_swa, best_state = -1.0, None
no_improve = 0
for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss = run_epoch(train_loader, train=True)
    val_swa, _, _, _ = evaluate(dev_loader)
    print(f"Epoch {epoch}: val_SWA={val_swa:.4f}")
    exp["losses"]["train"].append(tr_loss)
    exp["metrics"]["val"].append({"epoch": epoch, "SWA": val_swa})
    exp["timestamps"].append(time.time())
    if val_swa > best_swa + 1e-4:
        best_swa, val_swa_best = val_swa, val_swa
        best_state = model.state_dict()
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("Early stopping.")
            break
if best_state:
    model.load_state_dict(best_state)

# ---------- final evaluation --------------------
test_swa, preds, trues, seqs = evaluate(test_loader)
print(f"\nTEST Shape-Weighted Accuracy (SWA): {test_swa:.4f}")
exp["metrics"]["test"] = {"SWA": test_swa}
exp["predictions"], exp["ground_truth"] = preds, trues

# ---------- save artefacts ----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")

plt.figure(figsize=(4, 3))
plt.bar(["SWA"], [test_swa], color="coral")
plt.ylim(0, 1)
plt.title("Test SWA")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "test_swa.png"))
print("Saved plot.")
