# No-Projection-Head (Identity Representation) ablation study
import os, pathlib, random, time, math, json
from typing import List, Dict
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------- reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------- file system --------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "no_projection_head": {
        "spr_bench": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- locate SPR_BENCH ----------
def find_spr_bench() -> pathlib.Path:
    cand = [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for c in cand:
        if not c:
            continue
        p = pathlib.Path(c).expanduser().resolve()
        if (p / "train.csv").exists() and (p / "dev.csv").exists():
            print("Found SPR_BENCH at", p)
            return p
    raise FileNotFoundError("SPR_BENCH not found")


DATA_PATH = find_spr_bench()


# ---------- dataset loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):
        return load_dataset(
            "csv",
            data_files=str(root / fname),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- metrics helpers ----------
def count_shape_variety(s):
    return len(set(tok[0] for tok in s.split() if tok))


def count_color_variety(s):
    return len(set(tok[1] for tok in s.split() if len(tok) > 1))


def swa(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]


def cwa(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]


def ccwa(seqs, y, p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]


def metric(fn, seqs, y, p):
    w = [fn(s) for s in seqs]


# rewrote correctly
def swa(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == q else 0 for wi, t, q in zip(w, y, p)) / max(sum(w), 1)


def cwa(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == q else 0 for wi, t, q in zip(w, y, p)) / max(sum(w), 1)


def ccwa(seqs, y, p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi if t == q else 0 for wi, t, q in zip(w, y, p)) / max(sum(w), 1)


# ---------- vocab / label maps ----------
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
print("vocab", len(vocab), "labels", num_labels)


# ---------- augmentations -------------
def augment(seq: str) -> str:
    toks = seq.split()
    new = []
    for tok in toks:
        r = random.random()
        if r < 0.15:
            continue
        elif r < 0.30:
            new.append("<unk>")
        else:
            new.append(tok)
    if len(new) > 1 and random.random() < 0.3:
        i = random.randint(0, len(new) - 2)
        new[i], new[i + 1] = new[i + 1], new[i]
    if not new:
        new = ["<unk>"]
    return " ".join(new)


# ---------- torch dataset ------------
class SPRJointDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, vocab, lab2id):
        self.ds = hf_ds
        self.vocab = vocab
        self.lab2id = lab2id

    def encode(self, seq):
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in seq.split()]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        orig = self.encode(ex["sequence"])
        a1 = self.encode(augment(ex["sequence"]))
        a2 = self.encode(augment(ex["sequence"]))
        lab = self.lab2id[ex["label"]]
        return {
            "orig": torch.tensor(orig),
            "a1": torch.tensor(a1),
            "a2": torch.tensor(a2),
            "label": torch.tensor(lab),
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


# ---------- model (No Projection) ----
class EncoderNoProj(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hid, bidirectional=True, batch_first=True)
        self.out_dim = hid * 2

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.lstm(e)
        mask = (x != pad_id).unsqueeze(-1)
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return mean  # identity representation


class JointModel(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(encoder.out_dim, num_labels)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.cls(z)
        return z, logits


# ---------- losses -------------------
def nt_xent(z, temp=0.5):
    z = nn.functional.normalize(z, dim=1)
    sim = (z @ z.T) / temp
    B = z.size(0) // 2
    labels = torch.arange(0, 2 * B, device=z.device)
    labels = (labels + B) % (2 * B)
    sim.fill_diagonal_(-9e15)
    return nn.functional.cross_entropy(sim, labels)


# ---------- training loop ------------
def train(
    model, train_ds, dev_ds, epochs=20, batch=128, alpha=0.5, patience=4, lr=1e-3
):
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    best_cc = -1
    best_state = None
    no_imp = 0
    for ep in range(1, epochs + 1):
        # train
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
        experiment_data["no_projection_head"]["spr_bench"]["losses"]["train"].append(
            train_loss
        )
        experiment_data["no_projection_head"]["spr_bench"]["metrics"]["train"].append(
            {"epoch": ep, "loss": train_loss}
        )

        # eval
        model.eval()
        dev_loss = 0
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
                dev_loss += loss.item() * bt["labels"].size(0)
                preds.extend(logits.argmax(-1).cpu().tolist())
                trues.extend(bt["labels"].cpu().tolist())
                seqs.extend(b["sequences"])
        dev_loss /= len(dev_ds)
        s = swa(seqs, trues, preds)
        c = cwa(seqs, trues, preds)
        cc = ccwa(seqs, trues, preds)
        experiment_data["no_projection_head"]["spr_bench"]["losses"]["val"].append(
            dev_loss
        )
        experiment_data["no_projection_head"]["spr_bench"]["metrics"]["val"].append(
            {"epoch": ep, "swa": s, "cwa": c, "ccwa": cc, "loss": dev_loss}
        )
        print(
            f"Epoch {ep}: val_loss={dev_loss:.4f} SWA={s:.4f} CWA={c:.4f} CCWA={cc:.4f}"
        )

        if cc > best_cc + 1e-5:
            best_cc = cc
            best_state = model.state_dict()
            no_imp = 0
            experiment_data["no_projection_head"]["spr_bench"]["predictions"] = preds
            experiment_data["no_projection_head"]["spr_bench"]["ground_truth"] = trues
        else:
            no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break
    if best_state:
        model.load_state_dict(best_state)
    print("Best dev CCWA:", best_cc)


# ---------- build datasets & run -------
train_ds = SPRJointDataset(spr["train"], vocab, label2id)
dev_ds = SPRJointDataset(spr["dev"], vocab, label2id)
enc = EncoderNoProj(len(vocab))
model = JointModel(enc, num_labels)
train(model, train_ds, dev_ds)

# ---------- save results ---------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data ->", os.path.join(working_dir, "experiment_data.npy"))
