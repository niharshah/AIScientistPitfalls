import os, pathlib, random, time, math, json
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------- experiment store ----------
experiment_data = {
    "no_aug_contrast": {  # ablation name
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- reproducibility -----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- locate SPR_BENCH ----------
def find_spr_bench() -> pathlib.Path:
    cands = [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for c in cands:
        if not c:
            continue
        p = pathlib.Path(c).expanduser().resolve()
        if (p / "train.csv").exists() and (p / "dev.csv").exists():
            print("Found SPR_BENCH at", p)
            return p
    raise FileNotFoundError("SPR_BENCH not found")


DATA_PATH = find_spr_bench()


# ---------- load dataset --------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):
        return load_dataset(
            "csv",
            data_files=str(root / fname),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- metrics helpers ----------
def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def ccwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------- vocab / labels ------------
def build_vocab(dataset):
    v = {"<pad>": 0, "<unk>": 1}
    for ex in dataset:
        for tok in ex["sequence"].split():
            if tok not in v:
                v[tok] = len(v)
    return v


def build_label(dataset):
    labs = sorted({ex["label"] for ex in dataset})
    return {l: i for i, l in enumerate(labs)}


vocab = build_vocab(spr["train"])
label2id = build_label(spr["train"])
id2label = {i: l for l, i in label2id.items()}
pad_id = vocab["<pad>"]
num_labels = len(label2id)
print("vocab size", len(vocab), "labels", num_labels)


# ---------- dataset (NO AUG) ----------
class SPRNoAugDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, vocab, label2id):
        self.ds = hf_ds
        self.vocab = vocab
        self.label2id = label2id

    def encode(self, seq):
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in seq.split()]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        enc = torch.tensor(self.encode(ex["sequence"]), dtype=torch.long)
        lab = torch.tensor(self.label2id[ex["label"]], dtype=torch.long)
        return {
            "orig": enc,
            "a1": enc.clone(),  # identical view 1
            "a2": enc.clone(),  # identical view 2
            "label": lab,
            "sequence": ex["sequence"],
        }


def collate_joint(batch):
    def pad(seqs):
        m = max(len(s) for s in seqs)
        out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
        return out

    return {
        "orig": pad([b["orig"] for b in batch]),
        "a1": pad([b["a1"] for b in batch]),
        "a2": pad([b["a2"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
        "sequences": [b["sequence"] for b in batch],
    }


# ---------- model --------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hid, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(), nn.Linear(hid, hid)
        )

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        mask = (x != pad_id).unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj(mean)


class JointModel(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(encoder.proj[-1].out_features, num_labels)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.cls(z)


# ---------- losses -------------------
def nt_xent(z, temp=0.5):
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp
    B = z.size(0) // 2
    labels = torch.arange(0, 2 * B, device=z.device)
    labels = (labels + B) % (2 * B)
    sim.fill_diagonal_(-9e15)
    return nn.functional.cross_entropy(sim, labels)


# ---------- training loop ------------
def train(
    model, train_ds, dev_ds, alpha=0.5, epochs=20, batch=128, lr=1e-3, patience=4
):
    loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, collate_fn=collate_joint
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=256, shuffle=False, collate_fn=collate_joint
    )
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    best_ccwa = -1
    best_state = None
    no_imp = 0
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tot = 0
        for b in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in b.items()
            }
            optim.zero_grad()
            z1, _ = model(bt["a1"])
            z2, _ = model(bt["a2"])
            contrast = nt_xent(torch.cat([z1, z2], 0))
            _, logits = model(bt["orig"])
            ce = ce_loss(logits, bt["labels"])
            loss = ce + alpha * contrast
            loss.backward()
            optim.step()
            tot += loss.item() * bt["labels"].size(0)
        train_loss = tot / len(train_ds)
        experiment_data["no_aug_contrast"]["SPR_BENCH"]["losses"]["train"].append(
            train_loss
        )
        experiment_data["no_aug_contrast"]["SPR_BENCH"]["metrics"]["train"].append(
            {"epoch": epoch, "loss": train_loss}
        )
        # ---- eval ----
        model.eval()
        dev_tot = 0
        preds = []
        trues = []
        seqs = []
        with torch.no_grad():
            for b in dev_loader:
                bt = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in b.items()
                }
                _, logits = model(bt["orig"])
                loss = ce_loss(logits, bt["labels"])
                dev_tot += loss.item() * bt["labels"].size(0)
                preds.extend(logits.argmax(-1).cpu().tolist())
                trues.extend(bt["labels"].cpu().tolist())
                seqs.extend(b["sequences"])
        dev_loss = dev_tot / len(dev_ds)
        s = swa(seqs, trues, preds)
        c = cwa(seqs, trues, preds)
        cc = ccwa(seqs, trues, preds)
        experiment_data["no_aug_contrast"]["SPR_BENCH"]["losses"]["val"].append(
            dev_loss
        )
        experiment_data["no_aug_contrast"]["SPR_BENCH"]["metrics"]["val"].append(
            {"epoch": epoch, "swa": s, "cwa": c, "ccwa": cc, "loss": dev_loss}
        )
        print(
            f"Epoch {epoch}: val_loss={dev_loss:.4f} SWA={s:.4f} CWA={c:.4f} CCWA={cc:.4f}"
        )
        if cc > best_ccwa + 1e-5:
            best_ccwa = cc
            best_state = model.state_dict()
            no_imp = 0
            experiment_data["no_aug_contrast"]["SPR_BENCH"]["predictions"] = preds
            experiment_data["no_aug_contrast"]["SPR_BENCH"]["ground_truth"] = trues
        else:
            no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break
    if best_state:
        model.load_state_dict(best_state)
    print("Best dev CCWA:", best_ccwa)


# ---------- build datasets -----------
train_ds = SPRNoAugDataset(spr["train"], vocab, label2id)
dev_ds = SPRNoAugDataset(spr["dev"], vocab, label2id)

# ---------- run experiment -----------
enc = Encoder(len(vocab))
model = JointModel(enc, num_labels)
train(model, train_ds, dev_ds)

# ---------- save ---------------------
os.makedirs("working", exist_ok=True)
np.save("working/experiment_data.npy", experiment_data)
print("Saved experiment data -> working/experiment_data.npy")
