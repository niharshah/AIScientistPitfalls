import os, pathlib, random, math, time, json
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from datasets import load_dataset, DatasetDict

# -------------------------------------------------
# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------
# -------- SPR helper functions -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(f"{s}.csv")
    return d


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# -------------------------------------------------
# -------- Dataset preparation --------------------
data_root_candidates = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr_bench = None
for p in data_root_candidates:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print(f"Loaded data from {p}")
        break
if spr_bench is None:
    print("SPR_BENCH not found, generating tiny synthetic data.")

    def synth(n):
        seqs, labels = [], []
        shapes, colors = "ABCD", "1234"
        for _ in range(n):
            L = random.randint(4, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(L)
            )
            seqs.append(seq)
            labels.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr_bench = DatasetDict()
    for split, n in [("train", 500), ("dev", 100), ("test", 100)]:
        spr_bench[split] = load_dataset(
            "json", data_files={"train": None}, split="train", data=synth(n)
        )


# Build vocabulary
def tokenize(seq):
    return seq.strip().split()


vocab = {"<PAD>": 0, "<MASK>": 1}
for split in spr_bench.values():
    for seq in split["sequence"]:
        for tok in tokenize(seq):
            if tok not in vocab:
                vocab[tok] = len(vocab)
vocab_size, mask_id = len(vocab), vocab["<MASK>"]
print(f"Vocab size = {vocab_size}")


# -------------------------------------------------
# Torch datasets
class SPRContrastiveDataset(Dataset):
    def __init__(self, ds):
        self.seqs = ds["sequence"]

    def __len__(self):
        return len(self.seqs)

    def _augment(self, toks, p=0.15):
        out = [t for t in toks if random.random() >= p]
        if not out:
            out = [random.choice(toks)]
        return out

    def __getitem__(self, idx):
        toks = tokenize(self.seqs[idx])
        return self._augment(toks), self._augment(toks)


class SPRClassifierDataset(Dataset):
    def __init__(self, ds):
        self.seqs, self.labels = ds["sequence"], ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def collate_contrastive(batch):
    v1, v2 = zip(*batch)

    def encode(lst):
        ids = [torch.tensor([vocab[t] for t in toks]) for toks in lst]
        lens = [len(i) for i in ids]
        return pad_sequence(ids, batch_first=True), torch.tensor(lens)

    ids1, len1 = encode(v1)
    ids2, len2 = encode(v2)
    return {
        "ids1": ids1.to(device),
        "len1": len1.to(device),
        "ids2": ids2.to(device),
        "len2": len2.to(device),
    }


def collate_classifier(batch):
    toks, labels, raw = zip(*batch)
    ids = [torch.tensor([vocab[t] for t in t]) for t in toks]
    lens = [len(i) for i in ids]
    return {
        "ids": pad_sequence(ids, batch_first=True).to(device),
        "len": torch.tensor(lens).to(device),
        "label": torch.tensor(labels).to(device),
        "sequence": raw,
    }


pretrain_loader = DataLoader(
    SPRContrastiveDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_contrastive,
)
train_loader = DataLoader(
    SPRClassifierDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_classifier,
)
dev_loader = DataLoader(
    SPRClassifierDataset(spr_bench["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_classifier,
)


# -------------------------------------------------
# Model
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)

    def forward(self, ids, lens):
        x = self.emb(ids)
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, enc, num_cls):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.proj.out_features, num_cls)

    def forward(self, ids, lens):
        return self.head(self.enc(ids, lens))


# -------------------------------------------------
# Losses
def nt_xent_loss(z1, z2, temp=0.5):
    z1, z2 = nn.functional.normalize(z1, 1), nn.functional.normalize(z2, 1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(N, 2 * N, device=z.device)
    logits_12, logits_21 = sim[:N, N:], sim[N:, :N]
    denom_12 = torch.logsumexp(sim[:N], 1)
    denom_21 = torch.logsumexp(sim[N:], 1)
    return (
        (-logits_12.diag() + denom_12) + (-logits_21.diag() + denom_21)
    ).mean() * 0.5


# -------------------------------------------------
# Experiment dictionary
experiment_data = {"weight_decay": {}}


# -------------------------------------------------
# Training routine
def run_experiment(weight_decay_value):
    tag = str(weight_decay_value)
    experiment_data["weight_decay"][tag] = {
        "losses": {"pretrain": [], "train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "SCHM": []},
        "predictions": [],
        "ground_truth": [],
    }
    # build models
    encoder = Encoder(vocab_size).to(device)
    clf = Classifier(encoder, num_cls=len(set(spr_bench["train"]["label"]))).to(device)
    # pretrain
    opt_pt = torch.optim.Adam(
        encoder.parameters(), lr=1e-3, weight_decay=weight_decay_value
    )
    for ep in range(2):
        encoder.train()
        running = 0.0
        for b in pretrain_loader:
            opt_pt.zero_grad()
            z1 = encoder(b["ids1"], b["len1"])
            z2 = encoder(b["ids2"], b["len2"])
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            opt_pt.step()
            running += loss.item() * b["ids1"].size(0)
        epoch_loss = running / len(pretrain_loader.dataset)
        experiment_data["weight_decay"][tag]["losses"]["pretrain"].append(epoch_loss)
        print(f"[WD={tag}] Pretrain epoch {ep+1}: {epoch_loss:.4f}")
    # fine-tune
    opt_ft = torch.optim.Adam(
        clf.parameters(), lr=1e-3, weight_decay=weight_decay_value
    )
    crit = nn.CrossEntropyLoss()
    for ep in range(3):
        clf.train()
        running = 0.0
        for b in train_loader:
            opt_ft.zero_grad()
            logits = clf(b["ids"], b["len"])
            loss = crit(logits, b["label"])
            loss.backward()
            opt_ft.step()
            running += loss.item() * b["ids"].size(0)
        tr_loss = running / len(train_loader.dataset)
        experiment_data["weight_decay"][tag]["losses"]["train"].append(tr_loss)
        # eval
        clf.eval()
        val_loss = 0.0
        preds, truth, seqs = [], [], []
        with torch.no_grad():
            for b in dev_loader:
                logits = clf(b["ids"], b["len"])
                loss = crit(logits, b["label"])
                val_loss += loss.item() * b["ids"].size(0)
                p = logits.argmax(1).cpu().tolist()
                preds.extend(p)
                truth.extend(b["label"].cpu().tolist())
                seqs.extend(b["sequence"])
        val_loss /= len(dev_loader.dataset)
        experiment_data["weight_decay"][tag]["losses"]["val"].append(val_loss)
        swa = shape_weighted_accuracy(seqs, truth, preds)
        cwa = color_weighted_accuracy(seqs, truth, preds)
        schm = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
        experiment_data["weight_decay"][tag]["metrics"]["SWA"].append(swa)
        experiment_data["weight_decay"][tag]["metrics"]["CWA"].append(cwa)
        experiment_data["weight_decay"][tag]["metrics"]["SCHM"].append(schm)
        print(
            f"[WD={tag}] FT epoch {ep+1}: train={tr_loss:.4f} val={val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} SCHM={schm:.3f}"
        )
    experiment_data["weight_decay"][tag]["predictions"] = preds
    experiment_data["weight_decay"][tag]["ground_truth"] = truth
    # Plot curve
    plt.figure()
    plt.plot(experiment_data["weight_decay"][tag]["losses"]["train"], label="train")
    plt.plot(experiment_data["weight_decay"][tag]["losses"]["val"], label="val")
    plt.title(f"Loss curve (weight_decay={tag})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_wd_{tag}.png"))
    plt.close()
    del encoder, clf
    torch.cuda.empty_cache()


# -------------------------------------------------
# Hyper-parameter sweep
for wd in [0.0, 1e-5, 1e-4, 1e-3]:
    run_experiment(wd)

# -------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data and plots in ./working/")
