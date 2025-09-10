import os, random, math, time, pathlib, csv, sys
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------
# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# ---------------------------------------------------------------
# Working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------
# Metrics helpers (identical to baseline)
def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def harmonic_sca(swa, cwa, eps=1e-8):
    return 2 * swa * cwa / (swa + cwa + eps)


# ---------------------------------------------------------------
# Data (create tiny synthetic set if SPR_BENCH missing)
def generate_synth(path: pathlib.Path):
    shapes, colors = ["A", "B", "C", "D"], ["1", "2", "3"]

    def gen_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(5, 10))
        )

    def gen_csv(fn, n):
        with open(path / fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(n):
                seq = gen_seq()
                label = int(count_shape_variety(seq) % 2 == 0)
                w.writerow([i, seq, label])

    gen_csv("train.csv", 2000)
    gen_csv("dev.csv", 500)
    gen_csv("test.csv", 500)


def load_csv_dataset(folder: pathlib.Path) -> Dict[str, List[Dict]]:
    data = {}
    for split in ["train", "dev", "test"]:
        with open(folder / f"{split}.csv") as f:
            rdr = csv.DictReader(f)
            data[split] = [r for r in rdr]
            for r in data[split]:
                r["label"] = int(r["label"])
    return data


DATA_PATH = pathlib.Path(os.environ.get("SPR_PATH", "./SPR_BENCH"))
if not DATA_PATH.exists():
    os.makedirs(DATA_PATH, exist_ok=True)
    generate_synth(DATA_PATH)
datasets = load_csv_dataset(DATA_PATH)
print({k: len(v) for k, v in datasets.items()})

# ---------------------------------------------------------------
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


def encode(seq: str) -> List[int]:
    return [vocab[t] for t in seq.split()]


# ---------------------------------------------------------------
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
    pad = lambda x: x + [0] * (maxlen - len(x))
    inp = torch.tensor([pad(b["input"]) for b in batch], dtype=torch.long)
    seqs = [b["seq"] for b in batch]
    out = {"input": inp, "seq": seqs}
    if "label" in batch[0]:
        out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out


# ---------------------------------------------------------------
def augment(ids: List[int]) -> List[int]:
    new = []
    for tok in ids:
        r = random.random()
        if r < 0.1:
            continue
        if r < 0.2:
            new.append(1)
        else:
            new.append(tok)
    return new or ids


# ---------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, dim=128, hid=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=0)
        self.gru = nn.GRU(dim, hid, batch_first=True)

    def forward(self, x):
        emb = self.embed(x)
        lens = (x != 0).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return h[-1]


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, enc_out, cls):
        super().__init__()
        self.fc = nn.Linear(enc_out, cls)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------
def nt_xent(z, t=0.1):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / t
    B = z.size(0) // 2
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


# ---------------------------------------------------------------
def evaluate(enc, clf, loader):
    enc.eval()
    clf.eval()
    y_t, y_p, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            logits = clf(enc(x))
            y_p.extend(logits.argmax(1).cpu().tolist())
            y_t.extend(batch["label"].tolist())
            seqs.extend(batch["seq"])
    swa = shape_weighted_accuracy(seqs, y_t, y_p)
    cwa = color_weighted_accuracy(seqs, y_t, y_p)
    return swa, cwa, harmonic_sca(swa, cwa)


# ---------------------------------------------------------------
batch_size = 128
contrast_loader = DataLoader(
    SPRDataset(datasets["train"], supervised=False),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
train_loader = DataLoader(
    SPRDataset(datasets["train"], supervised=True),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRDataset(datasets["dev"], supervised=True),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRDataset(datasets["test"], supervised=True),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)

# ---------------------------------------------------------------
# Hyper-parameter sweep: contrastive_pretraining_epochs
pretrain_grid = [8, 11, 15]  # explore 8-15 epochs
experiment_data = {
    "contrastive_pretraining_epochs": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": pretrain_grid,
        }
    }
}

for epochs_pre in pretrain_grid:
    print(f"\n=== Running setting: contrastive_pretraining_epochs={epochs_pre} ===")
    # ---------- model instantiation ----------
    encoder = Encoder(vocab_size).to(device)
    proj = ProjectionHead().to(device)
    optim_enc = torch.optim.Adam(
        list(encoder.parameters()) + list(proj.parameters()), lr=1e-3
    )

    # ---------- contrastive pre-training ----------
    for ep in range(1, epochs_pre + 1):
        encoder.train()
        proj.train()
        tot = cnt = 0
        for batch in contrast_loader:
            ids = batch["input"]
            v1 = [augment(s.tolist()) for s in ids]
            v2 = [augment(s.tolist()) for s in ids]

            def to_t(seqs):
                L = max(len(s) for s in seqs)
                return torch.tensor(
                    [s + [0] * (L - len(s)) for s in seqs], dtype=torch.long
                )

            z1 = proj(encoder(to_t(v1).to(device)))
            z2 = proj(encoder(to_t(v2).to(device)))
            loss = nt_xent(torch.cat([z1, z2], 0))
            optim_enc.zero_grad()
            loss.backward()
            optim_enc.step()
            tot += loss.item()
            cnt += 1
        print(f"[Contrastive] epoch {ep}/{epochs_pre} loss={tot/cnt:.4f}")

    # ---------- supervised fine-tuning ----------
    classifier = Classifier(128, 2).to(device)
    optim_sup = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3
    )
    crit = nn.CrossEntropyLoss()
    sup_epochs = 5
    sup_losses = []
    for ep in range(1, sup_epochs + 1):
        encoder.train()
        classifier.train()
        tl = tc = 0
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["label"].to(device)
            logits = classifier(encoder(x))
            loss = crit(logits, y)
            optim_sup.zero_grad()
            loss.backward()
            optim_sup.step()
            tl += loss.item()
            tc += 1
        swa, cwa, hsca_dev = evaluate(encoder, classifier, dev_loader)
        sup_losses.append(tl / tc)
        print(f"[Sup] epoch {ep}/{sup_epochs} loss={tl/tc:.4f} Dev-HSCA={hsca_dev:.4f}")
    # store dev metric as last epoch HSCA
    experiment_data["contrastive_pretraining_epochs"]["SPR_BENCH"]["metrics"][
        "train"
    ].append(hsca_dev)
    experiment_data["contrastive_pretraining_epochs"]["SPR_BENCH"]["losses"][
        "train"
    ].append(np.mean(sup_losses))

    # ---------- final test ----------
    swa_t, cwa_t, hsca_test = evaluate(encoder, classifier, test_loader)
    print(f"TEST HSCA={hsca_test:.4f} (SWA={swa_t:.4f}, CWA={cwa_t:.4f})")
    experiment_data["contrastive_pretraining_epochs"]["SPR_BENCH"]["metrics"][
        "val"
    ].append(hsca_test)

# ---------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
