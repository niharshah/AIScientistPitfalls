# Bag-of-Words Encoder (Remove LSTM) – single-file experiment
import os, pathlib, random, time, math, json
from typing import List, Dict
import numpy as np
import torch, datasets
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- experiment store -----------------
experiment_data = {
    "bag_of_words_encoder": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}
exp_key = experiment_data["bag_of_words_encoder"]["SPR_BENCH"]

# ---------------- reproducibility ------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

# ---------------- device ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- locate SPR_BENCH -----------------
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


# ---------------- dataset loading ------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fn: str):
        return load_dataset(
            "csv",
            data_files=str(root / fn),
            split="train",
            cache_dir=str(pathlib.Path.cwd() / "working" / ".cache_dsets"),
        )

    dd = DatasetDict()
    for split in ["train", "dev", "test"]:
        dd[split] = _load(f"{split}.csv")
    return dd


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------------- metrics helpers ------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def swa(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def cwa(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def ccwa(seqs, y_t, y_p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


# ---------------- vocab / labels -------------------
def build_vocab(ds):
    v = {"<pad>": 0, "<unk>": 1}
    for ex in ds:
        for tok in ex["sequence"].split():
            if tok not in v:
                v[tok] = len(v)
    return v


def build_label(ds):
    labs = sorted({ex["label"] for ex in ds})
    return {l: i for i, l in enumerate(labs)}


vocab = build_vocab(spr["train"])
pad_id = vocab["<pad>"]
label2id = build_label(spr["train"])
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print("vocab size", len(vocab), "num_labels", num_labels)


# ---------------- data augmentation ---------------
def augment(seq: str) -> str:
    toks = seq.split()
    new = []
    for tok in toks:
        r = random.random()
        if r < 0.15:
            continue  # delete
        elif r < 0.30:
            new.append("<unk>")  # mask
        else:
            new.append(tok)
    if len(new) > 1 and random.random() < 0.3:
        i = random.randint(0, len(new) - 2)
        new[i], new[i + 1] = new[i + 1], new[i]
    if not new:
        new = ["<unk>"]
    return " ".join(new)


# ---------------- torch datasets ------------------
class SPRJointDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, vocab, label2id):
        self.ds, self.vocab, self.label2id = hf_ds, vocab, label2id

    def enc(self, s):
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in s.split()]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        return {
            "orig": torch.tensor(self.enc(ex["sequence"]), dtype=torch.long),
            "a1": torch.tensor(self.enc(augment(ex["sequence"])), dtype=torch.long),
            "a2": torch.tensor(self.enc(augment(ex["sequence"])), dtype=torch.long),
            "label": torch.tensor(self.label2id[ex["label"]], dtype=torch.long),
            "sequence": ex["sequence"],
        }


def collate(batch):
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


# ---------------- Bag-of-Words Encoder ------------
class BOWEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hid), nn.ReLU(), nn.Linear(hid, hid)
        )

    def forward(self, x):
        e = self.emb(x)  # (B,L,E)
        mask = (x != pad_id).unsqueeze(-1).type_as(e)  # (B,L,1)
        mean = (e * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B,E)
        return self.proj(mean)  # (B,hid)


class JointModel(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(encoder.proj[-1].out_features, num_labels)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.cls(z)
        return z, logits


# ---------------- losses --------------------------
def nt_xent(z, temp=0.5):
    z = nn.functional.normalize(z, dim=1)
    sim = z @ z.T / temp
    B = z.size(0) // 2
    labels = torch.arange(0, 2 * B, device=z.device)
    labels = (labels + B) % (2 * B)
    sim.fill_diagonal_(-9e15)
    return nn.functional.cross_entropy(sim, labels)


# ---------------- training ------------------------
def train(
    model, train_ds, dev_ds, epochs=20, batch=128, alpha=0.5, patience=4, lr=1e-3
):
    tr_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate)
    dv_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    best_ccwa = -1
    no_imp = 0
    best_state = None
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tot = 0
        for b in tr_loader:
            bt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            opt.zero_grad()
            z1, _ = model(bt["a1"])
            z2, _ = model(bt["a2"])
            contrast = nt_xent(torch.cat([z1, z2], 0))
            _, log = model(bt["orig"])
            ce_loss = ce(log, bt["labels"])
            loss = ce_loss + alpha * contrast
            loss.backward()
            opt.step()
            tot += loss.item() * bt["labels"].size(0)
        train_loss = tot / len(train_ds)
        exp_key["losses"]["train"].append(train_loss)
        exp_key["metrics"]["train"].append({"epoch": ep, "loss": train_loss})
        # ---- eval -----
        model.eval()
        dev_loss = 0
        preds = []
        trues = []
        seqs = []
        with torch.no_grad():
            for b in dv_loader:
                bt = {
                    k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()
                }
                _, log = model(bt["orig"])
                l = ce(log, bt["labels"])
                dev_loss += l.item() * bt["labels"].size(0)
                preds.extend(log.argmax(-1).cpu().tolist())
                trues.extend(bt["labels"].cpu().tolist())
                seqs.extend(b["sequences"])
        dev_loss /= len(dev_ds)
        s, c, cc = (
            swa(seqs, trues, preds),
            cwa(seqs, trues, preds),
            ccwa(seqs, trues, preds),
        )
        exp_key["losses"]["val"].append(dev_loss)
        exp_key["metrics"]["val"].append(
            {"epoch": ep, "swa": s, "cwa": c, "ccwa": cc, "loss": dev_loss}
        )
        print(
            f"Epoch {ep}: val_loss={dev_loss:.4f} SWA={s:.4f} CWA={c:.4f} CCWA={cc:.4f}"
        )
        if cc > best_ccwa + 1e-5:
            best_ccwa = cc
            best_state = model.state_dict()
            no_imp = 0
            exp_key["predictions"] = preds
            exp_key["ground_truth"] = trues
        else:
            no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break
    if best_state:
        model.load_state_dict(best_state)
    print("Best dev CCWA:", best_ccwa)


# ---------------- build datasets ------------------
working_dir = str(pathlib.Path.cwd() / "working")
os.makedirs(working_dir, exist_ok=True)
train_ds = SPRJointDataset(spr["train"], vocab, label2id)
dev_ds = SPRJointDataset(spr["dev"], vocab, label2id)

# ---------------- run experiment ------------------
encoder = BOWEncoder(len(vocab))
model = JointModel(encoder, num_labels)
train(model, train_ds, dev_ds)

# ---------------- save results --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data ->", os.path.join(working_dir, "experiment_data.npy"))
