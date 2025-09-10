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

import os, random, string, time, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------
# Working directory and device setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------
# Utility: metrics from proposal
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1e-6)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1e-6)


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 2 * swa * cwa / max(swa + cwa, 1e-6)


# -----------------------------------------------------------
# Data loading (real or synthetic fallback)
def load_spr_bench(root_path: str) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root_path, csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _ld("train.csv")
    d["dev"] = _ld("dev.csv")
    d["test"] = _ld("test.csv")
    return d


def make_synthetic_dataset(path: str, n_train=2000, n_dev=500, n_test=500):
    shapes = list("STCH")
    colors = list("RGBY")

    def rand_seq():
        L = random.randint(3, 10)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
        return " ".join(toks)

    def rule(seq):
        return int(count_shape_variety(seq) > count_color_variety(seq))

    def mk(n, fname):
        rows = ["id,sequence,label"]
        for i in range(n):
            s = rand_seq()
            rows.append(f"{i},{s},{rule(s)}")
        with open(os.path.join(path, fname), "w") as f:
            f.write("\n".join(rows))

    os.makedirs(path, exist_ok=True)
    mk(n_train, "train.csv")
    mk(n_dev, "dev.csv")
    mk(n_test, "test.csv")


root = os.getenv("SPR_PATH", "SPR_BENCH")
if not (
    os.path.exists(root)
    and all(
        os.path.exists(os.path.join(root, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )
):
    print("SPR_BENCH not found, generating synthetic data …")
    make_synthetic_dataset(root)

spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# -----------------------------------------------------------
# Tokenisation / vocab
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
num_labels = len(set(spr["train"]["label"]))
max_len = 20


def encode(seq, vocab=vocab, max_len=max_len):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in seq.split()][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


# -----------------------------------------------------------
# Torch dataset
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seq = hf_ds["sequence"]
        self.y = hf_ds["label"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.seq[idx]), dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return {"input": x, "label": y, "raw": self.seq[idx]}


batch_size = 64
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=batch_size)


# -----------------------------------------------------------
# Model
class BaselineGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.gru(emb)
        pooled = out.mean(dim=1)
        return self.fc(pooled)


model = BaselineGRU(len(vocab), num_classes=num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------------------------------------
# Experiment logger
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -----------------------------------------------------------
# Training loop
epochs = 5


def run_epoch(dl, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()
    tot_loss, y_true, y_pred, seqs = 0.0, [], [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"])
        loss = criterion(logits, batch["label"])
        if train_mode:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += loss.item() * len(batch["label"])
        preds = logits.argmax(dim=-1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch["label"].detach().cpu().tolist())
        seqs.extend(batch["raw"])
    avg_loss = tot_loss / len(dl.dataset)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, swa, cwa, hwa, y_true, y_pred


for epoch in range(1, epochs + 1):
    tr_loss, tr_swa, tr_cwa, tr_hwa, _, _ = run_epoch(train_dl, True)
    val_loss, val_swa, val_cwa, val_hwa, _, _ = run_epoch(dev_dl, False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append((epoch, tr_hwa))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, val_hwa))
    print(
        f"Epoch {epoch}: val_loss = {val_loss:.4f}, HWA = {val_hwa:.4f} (SWA {val_swa:.4f}, CWA {val_cwa:.4f})"
    )

# -----------------------------------------------------------
# Final evaluation on test
_, _, _, test_hwa, test_y, test_pred = run_epoch(test_dl, False)
print(f"Test HWA = {test_hwa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = test_y

# -----------------------------------------------------------
# Save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
