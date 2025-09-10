# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, numpy as np, torch
from typing import List
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ----------------- house-keeping -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------- data utilities -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.strip().split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(v if t == p else 0 for v, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(v if t == p else 0 for v, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa, cwa = shape_weighted_accuracy(seqs, y_true, y_pred), color_weighted_accuracy(
        seqs, y_true, y_pred
    )
    return 0 if swa + cwa == 0 else 2 * swa * cwa / (swa + cwa)


# ----------------- vocab -----------------
class Vocab:
    def __init__(self, tokens: List[str]):
        self.itos = ["<pad>"] + sorted(set(tokens))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, toks: List[str]):
        return [self.stoi[t] for t in toks]


# ----------------- model -----------------
class BagClassifier(nn.Module):
    def __init__(self, vocab_sz: int, embed_dim: int, n_cls: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_sz, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, n_cls)

    def forward(self, text, offsets):
        return self.fc(self.embedding(text, offsets))


# ----------------- dataset path -----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"SPR_BENCH not found at {DATA_PATH}")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# build vocab / label maps
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


# collate fn
def collate(batch):
    tok_ids, offs, lab_ids = [], [0], []
    for ex in batch:
        ids = vocab(ex["sequence"].split())
        tok_ids.extend(ids)
        offs.append(offs[-1] + len(ids))
        lab_ids.append(label2id[ex["label"]])
    text = torch.tensor(tok_ids, dtype=torch.long)
    offs = torch.tensor(offs[:-1], dtype=torch.long)
    labs = torch.tensor(lab_ids, dtype=torch.long)
    return text.to(device), offs.to(device), labs.to(device)


batch_size = 128
train_loader = DataLoader(
    spr["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    spr["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    spr["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)

# ----------------- experiment container -----------------
experiment_data = {"epochs": {}}

# ----------------- helper: evaluation -----------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    y_true, y_pred, seqs, loss_sum = [], [], [], 0.0
    with torch.no_grad():
        for b_idx, (txt, off, labs) in enumerate(loader):
            out = model(txt, off)
            loss_sum += criterion(out, labs).item() * labs.size(0)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[i] for i in labs.cpu().tolist()])
            start = b_idx * batch_size
            seqs.extend(loader.dataset["sequence"][start : start + labs.size(0)])
    avg_loss = loss_sum / len(y_true)
    swa, cwa = shape_weighted_accuracy(seqs, y_true, y_pred), color_weighted_accuracy(
        seqs, y_true, y_pred
    )
    hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, swa, cwa, hwa, y_true, y_pred


# ----------------- hyperparameter sweep -----------------
epoch_options = [5, 15, 25, 40]
embed_dim = 64
lr = 1e-3

for n_epochs in epoch_options:
    key = f"{n_epochs}_epochs"
    print(f"\n--- Training model for {n_epochs} epochs ---")
    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = BagClassifier(len(vocab), embed_dim, len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for ep in range(1, n_epochs + 1):
        model.train()
        run_loss = 0.0
        for txt, off, labs in train_loader:
            optimizer.zero_grad()
            out = model(txt, off)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labs.size(0)
        tr_loss = run_loss / len(spr["train"])
        val_loss, swa, cwa, hwa, _, _ = evaluate(model, dev_loader)

        exp_rec["losses"]["train"].append(tr_loss)
        exp_rec["losses"]["val"].append(val_loss)
        exp_rec["metrics"]["train"].append(None)
        exp_rec["metrics"]["val"].append({"SWA": swa, "CWA": cwa, "HWA": hwa})

        print(
            f"Epoch {ep}/{n_epochs} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} "
            f"| SWA={swa:.4f} | CWA={cwa:.4f} | HWA={hwa:.4f}"
        )

    # test evaluation
    test_loss, swa_t, cwa_t, hwa_t, y_t, y_p = evaluate(model, test_loader)
    print(
        f"Test @ {n_epochs} epochs | loss={test_loss:.4f} | SWA={swa_t:.4f} | "
        f"CWA={cwa_t:.4f} | HWA={hwa_t:.4f}"
    )

    exp_rec["predictions"], exp_rec["ground_truth"] = y_p, y_t
    exp_rec["test_metrics"] = {
        "loss": test_loss,
        "SWA": swa_t,
        "CWA": cwa_t,
        "HWA": hwa_t,
    }
    experiment_data["epochs"][key] = exp_rec

    torch.cuda.empty_cache()  # free GPU mem between runs

# ----------------- save all -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
