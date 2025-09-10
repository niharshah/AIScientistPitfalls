import os, pathlib, random, time, math, json, copy
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- experiment store -----------------
experiment_data = {
    "dual_encoder_no_share": {
        "spr_bench": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------------- reproducibility ------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- device ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- SPR_BENCH locator ----------------
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


# ---------------- dataset load ---------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):
        return load_dataset(
            "csv",
            data_files=str(root / fname),
            split="train",
            cache_dir="./working/.cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(f"{s}.csv")
    return d


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------------- metric helpers -------------------
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


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


# ---------------- vocab / label --------------------
def build_vocab(dataset) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for ex in dataset:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_label(dataset) -> Dict[str, int]:
    labs = sorted({ex["label"] for ex in dataset})
    return {l: i for i, l in enumerate(labs)}


vocab = build_vocab(spr["train"])
label2id = build_label(spr["train"])
id2label = {i: l for l, i in label2id.items()}
pad_id = vocab["<pad>"]
num_labels = len(label2id)
print("vocab", len(vocab), "labels", num_labels)


# ---------------- augmentations --------------------
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
        idx = random.randint(0, len(new) - 2)
        new[idx], new[idx + 1] = new[idx + 1], new[idx]
    if not new:
        new = ["<unk>"]
    return " ".join(new)


# ---------------- torch datasets ------------------
class SPRJointDataset(torch.utils.data.Dataset):
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
        orig = self.encode(ex["sequence"])
        a1 = self.encode(augment(ex["sequence"]))
        a2 = self.encode(augment(ex["sequence"]))
        lab = self.label2id[ex["label"]]
        return {
            "orig": torch.tensor(orig, dtype=torch.long),
            "a1": torch.tensor(a1, dtype=torch.long),
            "a2": torch.tensor(a2, dtype=torch.long),
            "label": torch.tensor(lab, dtype=torch.long),
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


# ---------------- encoders ------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            emb_dim, hid, num_layers=1, bidirectional=True, batch_first=True
        )
        self.proj = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(), nn.Linear(hid, hid)
        )

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        mask = (x != pad_id).unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        z = self.proj(mean)
        return z


class DualEncoderModel(nn.Module):
    def __init__(self, encoder_a: Encoder, encoder_b: Encoder, num_labels: int):
        super().__init__()
        self.enc_a = encoder_a
        self.enc_b = encoder_b
        hid = encoder_a.proj[-1].out_features
        self.cls = nn.Linear(hid, num_labels)

    def encode_a(self, x):
        return self.enc_a(x)

    def encode_b(self, x):
        return self.enc_b(x)

    def classify(self, x):
        z = self.encode_a(x)
        return z, self.cls(z)


# ---------------- loss ---------------------------
def nt_xent(z, temp=0.5):
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp
    B = z.size(0) // 2
    labels = torch.arange(0, 2 * B, device=z.device)
    labels = (labels + B) % (2 * B)
    sim.fill_diagonal_(-9e15)
    return nn.functional.cross_entropy(sim, labels)


# ---------------- training -----------------------
def train_dual(
    model,
    train_ds,
    dev_ds,
    epochs=20,
    batch=128,
    alpha=0.5,
    patience=4,
    lr=1e-3,
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
    best_ccwa, no_imp, best_state = -1, 0, None

    for epoch in range(1, epochs + 1):
        # --------- train -----------
        model.train()
        tot_loss = 0
        for b in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in b.items()
            }
            optim.zero_grad()
            z1 = model.encode_a(bt["a1"])
            z2 = model.encode_b(bt["a2"])
            contrast = nt_xent(torch.cat([z1, z2], 0))
            logits = model.cls(z1)
            ce = ce_loss(logits, bt["labels"])
            loss = ce + alpha * contrast
            loss.backward()
            optim.step()
            tot_loss += loss.item() * bt["labels"].size(0)

        train_loss = tot_loss / len(train_ds)
        experiment_data["dual_encoder_no_share"]["spr_bench"]["losses"]["train"].append(
            train_loss
        )
        experiment_data["dual_encoder_no_share"]["spr_bench"]["metrics"][
            "train"
        ].append({"epoch": epoch, "loss": train_loss})

        # --------- eval ------------
        model.eval()
        dev_loss, preds, trues, seqs = 0, [], [], []
        with torch.no_grad():
            for b in dev_loader:
                bt = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in b.items()
                }
                _, logits = model.classify(bt["a1"])
                loss = ce_loss(logits, bt["labels"])
                dev_loss += loss.item() * bt["labels"].size(0)
                preds.extend(logits.argmax(-1).cpu().tolist())
                trues.extend(bt["labels"].cpu().tolist())
                seqs.extend(b["sequences"])
        dev_loss /= len(dev_ds)
        s, c, cc = (
            swa(seqs, trues, preds),
            cwa(seqs, trues, preds),
            ccwa(seqs, trues, preds),
        )
        experiment_data["dual_encoder_no_share"]["spr_bench"]["losses"]["val"].append(
            dev_loss
        )
        experiment_data["dual_encoder_no_share"]["spr_bench"]["metrics"]["val"].append(
            {"epoch": epoch, "swa": s, "cwa": c, "ccwa": cc, "loss": dev_loss}
        )
        print(
            f"Epoch {epoch}: val_loss={dev_loss:.4f} SWA={s:.4f} CWA={c:.4f} CCWA={cc:.4f}"
        )

        if cc > best_ccwa + 1e-5:
            best_ccwa = cc
            best_state = copy.deepcopy(model.state_dict())
            experiment_data["dual_encoder_no_share"]["spr_bench"]["predictions"] = preds
            experiment_data["dual_encoder_no_share"]["spr_bench"][
                "ground_truth"
            ] = trues
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print("Best dev CCWA:", best_ccwa)


# ---------------- build datasets ------------------
train_ds = SPRJointDataset(spr["train"], vocab, label2id)
dev_ds = SPRJointDataset(spr["dev"], vocab, label2id)

# ---------------- run experiment ------------------
enc_a = Encoder(len(vocab))
enc_b = copy.deepcopy(enc_a)
model = DualEncoderModel(enc_a, enc_b, num_labels)
train_dual(model, train_ds, dev_ds)

# ---------------- save ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data ->", os.path.join(working_dir, "experiment_data.npy"))
