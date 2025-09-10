import os, pathlib, math, random, time
import numpy as np
from typing import List
from collections import Counter

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ------------------- experiment bookkeeping -------------------
experiment_data = {
    "num_epochs": {  # <-- hyper-parameter we tune
        "SPR_BENCH": []  # each list element = one run
    }
}
save_file = "experiment_data.npy"

# ------------------- device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# ------------------- helper fns for metrics -------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    cor = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    cor = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) else 0.0


def dwhs(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# ------------------- load dataset -------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


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


spr = load_spr_bench(DATA_PATH)
print("split sizes:", {k: len(v) for k, v in spr.items()})


# ------------------- glyph clustering -------------------
def token_feature(tok: str) -> List[float]:
    chars = [ord(c) for c in tok]
    return [chars[0], sum(chars[1:]) / len(chars[1:]) if len(chars) > 1 else 0.0]


all_tokens = set()
for seq in spr["train"]["sequence"]:
    all_tokens.update(seq.split())
all_tokens = sorted(all_tokens)
X = np.array([token_feature(t) for t in all_tokens])
k = max(8, int(math.sqrt(len(all_tokens))))
print(f"Clustering {len(all_tokens)} glyphs into {k} clusters…")
clusters = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(X)
glyph2cluster = {tok: int(c) for tok, c in zip(all_tokens, clusters)}
print("clustering done.")


# ------------------- dataset -------------------
class SPRClusteredDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [glyph2cluster.get(t, 0) + 1 for t in self.seqs[idx].split()]
        return {
            "input": torch.tensor(ids),
            "label": torch.tensor(int(self.labels[idx])),
            "raw": self.seqs[idx],
        }


def collate_fn(batch):
    lens = [len(b["input"]) for b in batch]
    maxlen = max(lens)
    pad = torch.zeros(maxlen, dtype=torch.long)
    inputs = []
    for b in batch:
        pad_len = maxlen - len(b["input"])
        inputs.append(torch.cat([b["input"], pad[:pad_len]]))
    return {
        "input": torch.stack(inputs).to(device),
        "len": torch.tensor(lens).to(device),
        "label": torch.stack([b["label"] for b in batch]).to(device),
        "raw": [b["raw"] for b in batch],
    }


train_ds, dev_ds, test_ds = (
    SPRClusteredDataset(spr[s]) for s in ["train", "dev", "test"]
)
train_loader = lambda bs: DataLoader(
    train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

num_labels = len(set(spr["train"]["label"]))
vocab_size = k + 2  # clusters + PAD(0) + OOV


# ------------------- model -------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb, hidden, out):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out)

    def forward(self, x, l):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, l.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(h.squeeze(0))


# ------------------- hyperparam search -------------------
max_epoch_choices = [5, 10, 15, 20]  # candidate maximum epochs
patience = 3  # early stopping patience

for max_epochs in max_epoch_choices:
    print(f"\n=== Run with max_epochs={max_epochs} ===")
    model = GRUClassifier(vocab_size, 32, 64, num_labels).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    run_record = {
        "hyperparam_value": max_epochs,
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": None,
        "ground_truth": None,
        "best_epoch": 0,
        "best_val_dwhs": 0.0,
    }
    best_state = None
    no_improve = 0

    # training loop
    tl = train_loader(256)
    for epoch in range(1, max_epochs + 1):
        # -- train
        model.train()
        total_loss = 0.0
        for batch in tl:
            optim.zero_grad()
            logit = model(batch["input"], batch["len"])
            loss = crit(logit, batch["label"])
            loss.backward()
            optim.step()
            total_loss += loss.item() * batch["label"].size(0)
        train_loss = total_loss / len(train_ds)
        run_record["losses"]["train"].append((epoch, train_loss))

        # -- validation
        model.eval()
        v_loss, preds, labels, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                logit = model(batch["input"], batch["len"])
                loss = crit(logit, batch["label"])
                v_loss += loss.item() * batch["label"].size(0)
                p = logit.argmax(-1).cpu().tolist()
                l = batch["label"].cpu().tolist()
                seqs.extend(batch["raw"])
                preds.extend(p)
                labels.extend(l)
        v_loss /= len(dev_ds)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        swa = shape_weighted_accuracy(seqs, labels, preds)
        dw = dwhs(cwa, swa)
        run_record["losses"]["val"].append((epoch, v_loss))
        run_record["metrics"]["val"].append((epoch, cwa, swa, dw))
        print(f"Epoch {epoch}/{max_epochs} | val DWHS={dw:.3f}")

        # early stopping bookkeeping
        if dw > run_record["best_val_dwhs"]:
            run_record["best_val_dwhs"] = dw
            run_record["best_epoch"] = epoch
            best_state = {
                k: v.clone().detach().cpu() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  early stopping triggered.")
                break

    # ---------------- test evaluation with best checkpoint ----------------
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logit = model(batch["input"], batch["len"])
            p = logit.argmax(-1).cpu().tolist()
            l = batch["label"].cpu().tolist()
            preds.extend(p)
            labels.extend(l)
            seqs.extend(batch["raw"])
    cwa = color_weighted_accuracy(seqs, labels, preds)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    dw = dwhs(cwa, swa)
    print(f"TEST  | best_epoch={run_record['best_epoch']}  DWHS={dw:.3f}")

    run_record["predictions"] = preds
    run_record["ground_truth"] = labels
    run_record["test_metrics"] = (cwa, swa, dw)
    experiment_data["num_epochs"]["SPR_BENCH"].append(run_record)

# ------------------- save -------------------
np.save(save_file, experiment_data)
print(f"Saved all results to {save_file}")
