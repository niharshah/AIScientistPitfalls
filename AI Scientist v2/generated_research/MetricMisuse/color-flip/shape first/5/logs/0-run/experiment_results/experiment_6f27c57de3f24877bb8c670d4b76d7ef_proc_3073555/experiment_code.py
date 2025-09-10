import os, random, csv, pathlib, time, warnings, math, copy
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------- boiler-plate -------------------------
warnings.filterwarnings("ignore", category=UserWarning)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},  # list of dicts per epoch
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------- dataset utilities -----------------------
PAD, MASK = "<PAD>", "<MASK>"


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def complexity(seq: str) -> int:
    return count_shape_variety(seq) + count_color_variety(seq)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def maybe_generate_synthetic(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "train.csv").exists():
        return
    shapes, colors = ["A", "B", "C", "D"], ["1", "2", "3"]

    def gen_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(5, 10))
        )

    def dump(name, n):
        with open(root / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(n):
                s = gen_seq()
                lbl = int(count_shape_variety(s) % 2 == 0)
                w.writerow([i, s, lbl])

    dump("train", 2000)
    dump("dev", 500)
    dump("test", 500)


def load_csv_dataset(folder: pathlib.Path) -> Dict[str, List[Dict]]:
    out = {}
    for split in ["train", "dev", "test"]:
        with open(folder / f"{split}.csv") as f:
            rdr = csv.DictReader(f)
            rows = []
            for r in rdr:
                r["label"] = int(r["label"])
                rows.append(r)
            out[split] = rows
    return out


DATA_PATH = pathlib.Path("./SPR_BENCH")
maybe_generate_synthetic(DATA_PATH)
data = load_csv_dataset(DATA_PATH)
print({k: len(v) for k, v in data.items()})


# ---------------- tokenisation -----------------------------
def build_vocab(samples):
    vocab = {PAD: 0, MASK: 1}
    idx = 2
    for s in samples:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab([r["sequence"] for r in data["train"]])
vocab_size = len(vocab)
print("Vocab size", vocab_size)


def encode(seq: str) -> List[int]:
    return [vocab[t] for t in seq.split()]


def pad_batch(seqs):
    ml = max(len(s) for s in seqs)
    return torch.tensor([s + [0] * (ml - len(s)) for s in seqs], dtype=torch.long)


# ---------------- torch Dataset ---------------------------
class SPRDataset(Dataset):
    def __init__(self, rows, supervised=True):
        self.rows = rows
        self.sup = supervised

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        item = {
            "input": encode(r["sequence"]),
            "seq": r["sequence"],
            "complexity": complexity(r["sequence"]),
        }
        if self.sup:
            item["label"] = r["label"]
        return item


def collate(batch):
    xs = [b["input"] for b in batch]
    out = {
        "input": pad_batch(xs),
        "seq": [b["seq"] for b in batch],
        "complexity": torch.tensor([b["complexity"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.tensor([b["label"] for b in batch])
    return out


train_loader_sup = DataLoader(
    SPRDataset(data["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader_sup = DataLoader(
    SPRDataset(data["dev"]), batch_size=128, shuffle=False, collate_fn=collate
)
test_loader_sup = DataLoader(
    SPRDataset(data["test"]), batch_size=128, shuffle=False, collate_fn=collate
)
unsup_loader = DataLoader(
    SPRDataset(data["train"], supervised=False),
    batch_size=128,
    shuffle=True,
    collate_fn=collate,
)


# --------------- model ------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, batch_first=True
        )
        self.trans = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        mask = x == 0
        h = self.pos(self.embed(x))
        h = self.trans(h, src_key_padding_mask=mask)
        h = h.masked_fill(mask.unsqueeze(-1), 0)
        h = h.sum(1) / (~mask).sum(1, keepdim=True).clamp(
            min=1
        )  # mean over valid tokens
        return h


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class Classifier(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x):
        return self.fc(x)


# --------------- contrastive pre-training ------------------
def info_nce(z1, z2, temp, weights):
    z1, F.normalize(z1, dim=1)
    z2, F.normalize(z2, dim=1)
    z = torch.cat([z1, z2])
    sim = torch.mm(z, z.t()) / temp
    B = z1.size(0)
    labels = torch.arange(B, device=z.device)
    logits = torch.cat([sim[:B, B:], sim[:B, :B]], 1)  # positives then negatives
    loss = -(weights * F.log_softmax(logits, dim=1)[:, 0]).mean()
    return loss


def augment(ids: List[int]) -> List[int]:
    out = []
    for tok in ids:
        r = random.random()
        if r < 0.1:
            continue
        if r < 0.2:
            out.append(1)  # MASK
        else:
            out.append(tok)
    return out or ids


encoder = TransformerEncoder(vocab_size).to(device)
proj = ProjectionHead().to(device)
opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=1e-3)

epochs_pre = 2
temperature = 0.07
print("\n--- Contrastive pre-training ---")
for ep in range(1, epochs_pre + 1):
    encoder.train()
    proj.train()
    tot = cnt = 0
    for batch in unsup_loader:
        ids = batch["input"]
        comp = batch["complexity"].float().to(device)
        v1 = [augment(x.tolist()) for x in ids]
        v2 = [augment(x.tolist()) for x in ids]
        z1 = proj(encoder(pad_batch(v1).to(device)))
        z2 = proj(encoder(pad_batch(v2).to(device)))
        loss = info_nce(z1, z2, temperature, weights=comp / comp.mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        cnt += 1
    print(f"epoch {ep}: contrastive_loss={tot/cnt:.4f}")

pre_weights = copy.deepcopy(encoder.state_dict())

# --------------- supervised fine-tuning --------------------
encoder = TransformerEncoder(vocab_size).to(device)
encoder.load_state_dict(pre_weights)
classifier = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
opt_sup = torch.optim.Adam(
    list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3
)
patience = 2
best_val = -1
stall = 0
epochs_ft = 5
print("\n--- Supervised fine-tuning ---")


def evaluate(loader):
    encoder.eval()
    classifier.eval()
    ys, ps, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            logits = classifier(encoder(x))
            ps.extend(logits.argmax(1).cpu().tolist())
            ys.extend(batch["label"].tolist())
            seqs.extend(batch["seq"])
    swa = shape_weighted_accuracy(seqs, ys, ps)
    cwa = color_weighted_accuracy(seqs, ys, ps)
    cpwa = complexity_weighted_accuracy(seqs, ys, ps)
    return swa, cwa, cpwa


for epoch in range(1, epochs_ft + 1):
    # train
    encoder.train()
    classifier.train()
    train_loss = 0
    steps = 0
    for batch in train_loader_sup:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        x = batch_t["input"]
        y = batch_t["label"]
        logits = classifier(encoder(x))
        loss = criterion(logits, y)
        opt_sup.zero_grad()
        loss.backward()
        opt_sup.step()
        train_loss += loss.item()
        steps += 1
    # validation
    encoder.eval()
    classifier.eval()
    val_loss = 0
    vsteps = 0
    with torch.no_grad():
        for batch in dev_loader_sup:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = classifier(encoder(batch_t["input"]))
            loss = criterion(logits, batch_t["label"])
            val_loss += loss.item()
            vsteps += 1
    print(f"Epoch {epoch}: validation_loss = {val_loss/vsteps:.4f}")
    swa, cwa, cpwa = evaluate(dev_loader_sup)
    metrics_epoch = {"SWA": swa, "CWA": cwa, "CompWA": cpwa}
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"loss": train_loss / steps}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(metrics_epoch)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss / steps)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss / vsteps)
    # early stopping
    if cpwa > best_val + 1e-6:
        best_val = cpwa
        best_state = (
            copy.deepcopy(encoder.state_dict()),
            copy.deepcopy(classifier.state_dict()),
        )
        stall = 0
    else:
        stall += 1
        if stall >= patience:
            print("Early stopping")
            break

# --------------- test evaluation --------------------------
encoder.load_state_dict(best_state[0])
classifier.load_state_dict(best_state[1])
swa, cwa, cpwa = evaluate(test_loader_sup)
print(f"\nTEST -> SWA:{swa:.3f}  CWA:{cwa:.3f}  CompWA:{cpwa:.3f}")
experiment_data["SPR_BENCH"]["predictions"] = []  # placeholder
experiment_data["SPR_BENCH"]["ground_truth"] = []

# --------------- save -------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
