import os, random, math, time, pathlib, csv
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# Working dir, device & experiment dict
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {"batch_size": {}}  # every entry will be "SPR_BENCH_bs{size}": {...}


# ------------------------------------------------------------
# Accuracy helpers
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1)


def harmonic_sca(swa, cwa, eps=1e-8):
    return 2 * swa * cwa / (swa + cwa + eps)


# ------------------------------------------------------------
# Build / load data (synthetic fallback)
DATA_PATH = pathlib.Path(os.environ.get("SPR_PATH", "./SPR_BENCH"))


def _generate_synthetic(path: pathlib.Path):
    shapes, colors = ["A", "B", "C", "D"], ["1", "2", "3"]

    def gen_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(5, 10))
        )

    def gen_csv(name, n):
        with open(path / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(n):
                seq = gen_seq()
                label = int(count_shape_variety(seq) % 2 == 0)
                w.writerow([i, seq, label])

    gen_csv("train.csv", 2000)
    gen_csv("dev.csv", 500)
    gen_csv("test.csv", 500)


def _load_csv(path: pathlib.Path) -> Dict[str, List[Dict]]:
    d = {}
    for split in ["train", "dev", "test"]:
        with open(path / f"{split}.csv") as f:
            rdr = csv.DictReader(f)
            d[split] = [r for r in rdr]
            for r in d[split]:
                r["label"] = int(r["label"])
    return d


if not DATA_PATH.exists():
    os.makedirs(DATA_PATH, exist_ok=True)
    _generate_synthetic(DATA_PATH)
datasets = _load_csv(DATA_PATH)
print({k: len(v) for k, v in datasets.items()})

# ------------------------------------------------------------
# Vocabulary / encoding
PAD, MASK = "<PAD>", "<MASK>"


def build_vocab(samples):
    vocab = {PAD: 0, MASK: 1}
    idx = 2
    for s in samples:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab([r["sequence"] for r in datasets["train"]])
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq: str) -> List[int]:
    return [vocab[t] for t in seq.split()]


# ------------------------------------------------------------
# Torch Datasets / collate
class SPRDataset(Dataset):
    def __init__(self, rows, supervised=True):
        self.rows, self.sup = rows, supervised

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        ids = encode(r["sequence"])
        out = {"input": ids, "seq": r["sequence"]}
        if self.sup:
            out["label"] = r["label"]
        return out


def collate(batch):
    maxlen = max(len(b["input"]) for b in batch)
    inp, lab, seqs = [], [], []
    for b in batch:
        seqs.append(b["seq"])
        pad = b["input"] + [0] * (maxlen - len(b["input"]))
        inp.append(pad)
        if "label" in b:
            lab.append(b["label"])
    out = {"input": torch.tensor(inp, dtype=torch.long), "seq": seqs}
    if lab:
        out["label"] = torch.tensor(lab, dtype=torch.long)
    return out


# ------------------------------------------------------------
# Augmentation for contrastive views
def augment(ids: List[int]) -> List[int]:
    new = []
    for tok in ids:
        r = random.random()
        if r < 0.1:
            continue
        if r < 0.2:
            new.append(1)  # MASK id
        else:
            new.append(tok)
    return new or ids


# ------------------------------------------------------------
# Models
class Encoder(nn.Module):
    def __init__(self, vocab_sz, dim=128, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, dim, padding_idx=0)
        self.rnn = nn.GRU(dim, hidden, batch_first=True)

    def forward(self, x):
        emb = self.emb(x)
        lengths = (x != 0).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        return h[-1]


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, in_dim, n_cls=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_cls)

    def forward(self, x):
        return self.fc(x)


# ------------------------------------------------------------
# NT-Xent
def nt_xent(z, t=0.1):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / t
    B = z.size(0) // 2
    idx = torch.arange(B, device=z.device)
    loss = 0.0
    for i in range(B):
        pos = sim[i, i + B]
        denom = torch.cat([sim[i, :i], sim[i, i + 1 :]])
        loss += -torch.log(torch.exp(pos) / (torch.exp(denom).sum() + 1e-8))
        j = i + B
        pos = sim[j, i]
        denom = torch.cat([sim[j, :j], sim[j, j + 1 :]])
        loss += -torch.log(torch.exp(pos) / (torch.exp(denom).sum() + 1e-8))
    return loss / (2 * B)


# ------------------------------------------------------------
# Evaluation helpers
def evaluate(enc, clf, loader, return_preds=False):
    enc.eval(), clf.eval()
    y_t, y_p, seqs = [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["input"].to(device)
            logits = clf(enc(x))
            pred = logits.argmax(1).cpu().tolist()
            y_p.extend(pred)
            y_t.extend(b["label"].tolist())
            seqs.extend(b["seq"])
    swa = shape_weighted_accuracy(seqs, y_t, y_p)
    cwa = color_weighted_accuracy(seqs, y_t, y_p)
    hsca = harmonic_sca(swa, cwa)
    if return_preds:
        return swa, cwa, hsca, y_p, y_t
    return swa, cwa, hsca


# ------------------------------------------------------------
# Full experiment over different batch sizes
batch_sizes = [64, 128, 256, 512]
epochs_pre, epochs_sup = 3, 5

for bs in batch_sizes:
    tag = f"SPR_BENCH_bs{bs}"
    print(f"\n=== Running experiment: batch_size={bs} ===")
    # ---------------------- DataLoaders
    contrast_loader = DataLoader(
        SPRDataset(datasets["train"], supervised=False),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate,
    )
    train_loader = DataLoader(
        SPRDataset(datasets["train"], supervised=True),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        SPRDataset(datasets["dev"], supervised=True),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        SPRDataset(datasets["test"], supervised=True),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate,
    )

    # ---------------------- Models & optimizers
    encoder = Encoder(vocab_size).to(device)
    proj = ProjectionHead().to(device)
    optim_enc = torch.optim.Adam(
        list(encoder.parameters()) + list(proj.parameters()), lr=1e-3
    )

    # ---------- Stage 1: contrastive
    for ep in range(1, epochs_pre + 1):
        encoder.train(), proj.train()
        tot, cnt = 0, 0
        for batch in contrast_loader:
            ids = batch["input"]
            v1 = [augment(s.tolist()) for s in ids]
            v2 = [augment(s.tolist()) for s in ids]

            def to_tensor(lst):
                ml = max(len(s) for s in lst)
                return torch.tensor(
                    [s + [0] * (ml - len(s)) for s in lst], dtype=torch.long
                ).to(device)

            z1 = proj(encoder(to_tensor(v1)))
            z2 = proj(encoder(to_tensor(v2)))
            loss = nt_xent(torch.cat([z1, z2], 0))
            optim_enc.zero_grad()
            loss.backward()
            optim_enc.step()
            tot += loss.item()
            cnt += 1
        print(f"[Contrastive] epoch {ep} loss={tot/cnt:.4f}")

    # ---------- Stage 2: supervised fine-tuning
    clf = Classifier(128, 2).to(device)
    optim_all = torch.optim.Adam(
        list(encoder.parameters()) + list(clf.parameters()), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    train_losses, val_metrics = [], []
    for ep in range(1, epochs_sup + 1):
        encoder.train(), clf.train()
        tloss, tcnt = 0, 0
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["label"].to(device)
            logits = clf(encoder(x))
            loss = criterion(logits, y)
            optim_all.zero_grad()
            loss.backward()
            optim_all.step()
            tloss += loss.item()
            tcnt += 1
        swa, cwa, hsca = evaluate(encoder, clf, dev_loader)
        train_losses.append(tloss / tcnt)
        val_metrics.append(hsca)
        print(f"[Supervised] epoch {ep} loss={tloss/tcnt:.4f} | dev HSCA={hsca:.4f}")

    # ---------- Final test
    swa, cwa, hsca, preds, gts = evaluate(encoder, clf, test_loader, True)
    print(f"TEST  -> SWA {swa:.4f} | CWA {cwa:.4f} | HSCA {hsca:.4f}")

    # ---------- Store
    experiment_data["batch_size"][tag] = {
        "metrics": {"train": val_metrics, "val": [hsca]},
        "losses": {"train": train_losses, "val": []},
        "predictions": preds,
        "ground_truth": gts,
    }

# ------------------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
