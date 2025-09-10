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

import os, random, math, time, pathlib, csv, copy, warnings
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
# Working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Experiment-data dict (required format)
experiment_data = {
    "supervised_finetuning_epochs": {
        "SPR_BENCH": {
            "epochs_grid": [5, 10, 15, 20],
            "metrics": {"train": [], "val": []},  # list-of-lists per sweep
            "losses": {"train": [], "val": []},
            "test_hsca": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# ------------------------------------------------------------------
# Metric helpers
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum((wi if yt == yp else 0) for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum((wi if yt == yp else 0) for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def harmonic_sca(swa, cwa, eps=1e-8):
    return 2 * swa * cwa / (swa + cwa + eps)


# ------------------------------------------------------------------
# Data – load or create synthetic
def generate_synthetic(path: pathlib.Path):
    shapes = ["A", "B", "C", "D"]
    colors = ["1", "2", "3"]

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


def load_csv_dataset(folder: pathlib.Path) -> Dict[str, List[Dict]]:
    out = {}
    for split in ["train", "dev", "test"]:
        with open(folder / f"{split}.csv") as f:
            rdr = csv.DictReader(f)
            lst = [row for row in rdr]
            for r in lst:
                r["label"] = int(r["label"])
            out[split] = lst
    return out


DATA_PATH = pathlib.Path(os.environ.get("SPR_PATH", "./SPR_BENCH"))
if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    generate_synthetic(DATA_PATH)

datasets = load_csv_dataset(DATA_PATH)
print({k: len(v) for k, v in datasets.items()})

# ------------------------------------------------------------------
# Vocab
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
print(f"Vocab size: {vocab_size}")


def encode(seq: str) -> List[int]:
    return [vocab[t] for t in seq.split()]


# ------------------------------------------------------------------
# Dataset & collate
class SPRDataset(Dataset):
    def __init__(self, rows, supervised=True):
        self.rows = rows
        self.sup = supervised

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        ids = encode(r["sequence"])
        item = {"input": ids, "seq": r["sequence"]}
        if self.sup:
            item["label"] = r["label"]
        return item


def collate(batch):
    maxlen = max(len(b["input"]) for b in batch)
    pads = [b["input"] + [0] * (maxlen - len(b["input"])) for b in batch]
    inp = torch.tensor(pads, dtype=torch.long)
    out = {"input": inp, "seq": [b["seq"] for b in batch]}
    if "label" in batch[0]:
        out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out


# ------------------------------------------------------------------
# Augmentation for contrastive
def augment(ids: List[int]) -> List[int]:
    new = []
    for tok in ids:
        r = random.random()
        if r < 0.1:
            continue
        if r < 0.2:
            new.append(1)  # MASK token id
        else:
            new.append(tok)
    return new if new else ids


# ------------------------------------------------------------------
# Model components
class Encoder(nn.Module):
    def __init__(self, vocab_sz, dim=128, hid=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, dim, padding_idx=0)
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


# ------------------------------------------------------------------
# Losses / evaluation
def nt_xent(z, temp=0.1):
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temp
    B = z.size(0) // 2
    loss = 0.0
    for i in range(B):
        pos = sim[i, i + B]
        denom = torch.cat([sim[i, :i], sim[i, i + 1 :]])
        loss += -torch.log(torch.exp(pos) / torch.exp(denom).sum())
        j = i + B
        pos2 = sim[j, i]
        denom2 = torch.cat([sim[j, :j], sim[j, j + 1 :]])
        loss += -torch.log(torch.exp(pos2) / torch.exp(denom2).sum())
    return loss / (2 * B)


def evaluate(enc, clf, loader):
    enc.eval()
    clf.eval()
    ys, ps, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            logits = clf(enc(x))
            ps.extend(logits.argmax(1).cpu().tolist())
            ys.extend(batch["label"].tolist())
            seqs.extend(batch["seq"])
    swa = shape_weighted_accuracy(seqs, ys, ps)
    cwa = color_weighted_accuracy(seqs, ys, ps)
    return swa, cwa, harmonic_sca(swa, cwa)


# ------------------------------------------------------------------
# DataLoaders
batch_size = 128
contrast_loader = DataLoader(
    SPRDataset(datasets["train"], supervised=False),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
train_loader = DataLoader(
    SPRDataset(datasets["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRDataset(datasets["dev"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRDataset(datasets["test"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)

# ------------------------------------------------------------------
# Stage-1: Contrastive pre-training
encoder = Encoder(vocab_size).to(device)
proj = ProjectionHead().to(device)
optim_enc = torch.optim.Adam(
    list(encoder.parameters()) + list(proj.parameters()), lr=1e-3
)

epochs_pre = 3
for ep in range(1, epochs_pre + 1):
    encoder.train()
    proj.train()
    tot = cnt = 0
    for batch in contrast_loader:
        ids = batch["input"]
        v1 = [augment(seq.tolist()) for seq in ids]
        v2 = [augment(seq.tolist()) for seq in ids]

        def pad(seqs):
            ml = max(len(s) for s in seqs)
            return torch.tensor(
                [s + [0] * (ml - len(s)) for s in seqs], dtype=torch.long
            )

        z1 = proj(encoder(pad(v1).to(device)))
        z2 = proj(encoder(pad(v2).to(device)))
        loss = nt_xent(torch.cat([z1, z2], 0))
        optim_enc.zero_grad()
        loss.backward()
        optim_enc.step()
        tot += loss.item()
        cnt += 1
    print(f"[Pre-train] epoch {ep}/{epochs_pre}  loss={tot/cnt:.4f}")

pretrained_weights = copy.deepcopy(encoder.state_dict())  # save snapshot

# ------------------------------------------------------------------
# Stage-2 sweep: fine-tuning epochs grid
epochs_grid = experiment_data["supervised_finetuning_epochs"]["SPR_BENCH"][
    "epochs_grid"
]
criterion = nn.CrossEntropyLoss()
patience = 3  # early stopping patience

for max_epochs in epochs_grid:
    print(
        f"\n=== Fine-tuning for up-to {max_epochs} epochs (early-stopping patience={patience}) ==="
    )
    enc = Encoder(vocab_size).to(device)
    enc.load_state_dict(pretrained_weights)
    clf = Classifier(128, 2).to(device)
    optim_all = torch.optim.Adam(
        list(enc.parameters()) + list(clf.parameters()), lr=1e-3
    )

    train_hsca_history, val_hsca_history = [], []
    train_loss_history, val_loss_dummy = [], []

    best_hsca, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        enc.train()
        clf.train()
        t_loss, steps = 0, 0
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["label"].to(device)
            logits = clf(enc(x))
            loss = criterion(logits, y)
            optim_all.zero_grad()
            loss.backward()
            optim_all.step()
            t_loss += loss.item()
            steps += 1

        swa, cwa, hsca = evaluate(enc, clf, train_loader)
        train_hsca_history.append(hsca)
        train_loss_history.append(t_loss / steps)

        val_swa, val_cwa, val_hsca = evaluate(enc, clf, dev_loader)
        val_hsca_history.append(val_hsca)
        val_loss_dummy.append(0.0)  # placeholder if you want val loss later

        print(f"  Epoch {epoch:02d}: train HSCA={hsca:.4f} | dev HSCA={val_hsca:.4f}")

        # early stopping
        if val_hsca > best_hsca + 1e-6:
            best_hsca = val_hsca
            best_state = (
                copy.deepcopy(enc.state_dict()),
                copy.deepcopy(clf.state_dict()),
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  Early stopping!")
                break

    # restore best & evaluate on test
    enc.load_state_dict(best_state[0])
    clf.load_state_dict(best_state[1])
    test_swa, test_cwa, test_hsca = evaluate(enc, clf, test_loader)
    print(f"  >>> TEST HSCA = {test_hsca:.4f}")

    experiment_data["supervised_finetuning_epochs"]["SPR_BENCH"]["metrics"][
        "train"
    ].append(train_hsca_history)
    experiment_data["supervised_finetuning_epochs"]["SPR_BENCH"]["metrics"][
        "val"
    ].append(val_hsca_history)
    experiment_data["supervised_finetuning_epochs"]["SPR_BENCH"]["losses"][
        "train"
    ].append(train_loss_history)
    experiment_data["supervised_finetuning_epochs"]["SPR_BENCH"]["losses"][
        "val"
    ].append(val_loss_dummy)
    experiment_data["supervised_finetuning_epochs"]["SPR_BENCH"]["test_hsca"].append(
        test_hsca
    )
    experiment_data["supervised_finetuning_epochs"]["SPR_BENCH"]["predictions"].append(
        []
    )  # placeholder
    experiment_data["supervised_finetuning_epochs"]["SPR_BENCH"]["ground_truth"].append(
        []
    )  # placeholder

# ------------------------------------------------------------------
# Save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
