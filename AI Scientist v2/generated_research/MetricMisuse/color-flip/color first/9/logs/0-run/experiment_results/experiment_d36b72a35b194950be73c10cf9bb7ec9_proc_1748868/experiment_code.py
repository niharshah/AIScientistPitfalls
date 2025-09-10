import os, math, random, pathlib, numpy as np, torch, torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# -----------------------  I/O & DEVICE  ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------  EXPERIMENT DATA  ------------------------
experiment_data = {"single_dataset": {}, "union_all": {}}

# ------------------  SYNTHETIC DATA GENERATION  -------------------
rules = {
    "majority_shape_A": lambda toks: int(
        sum(t[0] == "A" for t in toks) > len(toks) / 2
    ),
    "majority_color_1": lambda toks: int(
        sum(t[1:] == "1" for t in toks) > len(toks) / 2
    ),
    "even_sequence_len": lambda toks: int(len(toks) % 2 == 0),
    "xor_first_last_shape": lambda toks: int(toks[0][0] != toks[-1][0]),
}


def make_seq(rng):
    shapes, colors = "ABCD", "1234"
    return [rng.choice(shapes) + rng.choice(colors) for _ in range(rng.randint(4, 8))]


def build_dataset(root: pathlib.Path, rule_fn):
    rng = random.Random(0)
    nrows = {"train": 500, "dev": 100, "test": 100}
    root.mkdir(parents=True, exist_ok=True)
    for split, n in nrows.items():
        with open(root / f"{split}.csv", "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                toks = make_seq(rng)
                f.write(f"{i},{' '.join(toks)},{rule_fn(toks)}\n")


multi_root = pathlib.Path(working_dir) / "MULTI_SPR_DS"
for rname, rfun in rules.items():
    build_dataset(multi_root / rname, rfun)
print("Synthetic datasets built at", multi_root)


# ------------------------  LOAD WITH HF  --------------------------
def load_csv_folder(folder: pathlib.Path) -> DatasetDict:
    def _load(csv_file):
        return load_dataset(
            "csv", data_files=str(csv_file), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load(folder / "train.csv"),
            "dev": _load(folder / "dev.csv"),
            "test": _load(folder / "test.csv"),
        }
    )


datasets = {r: load_csv_folder(multi_root / r) for r in rules}

# ----------------------  GLYPH CLUSTERING  ------------------------
all_glyphs = sorted(
    {
        tok
        for r in rules
        for seq in datasets[r]["train"]["sequence"]
        for tok in seq.split()
    }
)
k_clusters = min(16, len(all_glyphs)) or 1
features = np.stack([[ord(t[0]), ord(t[1])] for t in all_glyphs])
labels = KMeans(n_clusters=k_clusters, random_state=0, n_init="auto").fit_predict(
    features
)
glyph2cluster = {g: int(c) + 1 for g, c in zip(all_glyphs, labels)}  # 0 = PAD
vocab_size = k_clusters + 1
print(f"Clustered {len(all_glyphs)} glyphs into {k_clusters} clusters.")


# -------------------------  METRICS  ------------------------------
def count_color_variety(seq):
    return len({t[1:] for t in seq.split()})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split()})


def color_weighted_acc(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]
    return sum(w_i if yt == pt else 0 for w_i, yt, pt in zip(w, y, p)) / max(sum(w), 1)


def shape_weighted_acc(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(w_i if yt == pt else 0 for w_i, yt, pt in zip(w, y, p)) / max(sum(w), 1)


def balanced_weighted_acc(cwa, swa):
    return (cwa + swa) / 2


# ----------------------  DATASET / DATALOADER  --------------------
class SPR(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.lab = [int(x) for x in hf_split["label"]]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        ids = [glyph2cluster[t] for t in self.seq[i].split()]
        return {
            "input": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.lab[i], dtype=torch.long),
            "raw": self.seq[i],
        }


def collate(batch):
    lens = [len(b["input"]) for b in batch]
    maxl = max(lens)
    pad = lambda x: torch.cat([x, torch.zeros(maxl - len(x), dtype=torch.long)])
    return {
        "input": torch.stack([pad(b["input"]) for b in batch]),
        "len": torch.tensor(lens),
        "label": torch.stack([b["label"] for b in batch]),
        "raw": [b["raw"] for b in batch],
    }


# ---------------------------  MODEL  ------------------------------
class TransCLS(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 64
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(64, 1, d_model))
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 4, 2 * d_model, batch_first=False), 2
        )
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        src = self.emb(x).transpose(0, 1)  # T,B,D
        src = src + self.pos[: src.size(0)]
        mask = x == 0
        h = self.enc(src, src_key_padding_mask=mask)
        h = h.masked_fill(mask.transpose(0, 1).unsqueeze(-1), 0)
        pooled = h.sum(0) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(pooled)


# ----------------------  TRAIN / EVALUATE  ------------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs = [], [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input"])
            loss = criterion(logits, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
            preds += logits.argmax(-1).cpu().tolist()
            gts += batch["label"].cpu().tolist()
            seqs += batch["raw"]
    cwa = color_weighted_acc(seqs, gts, preds)
    swa = shape_weighted_acc(seqs, gts, preds)
    hwa = 2 * cwa * swa / (cwa + swa) if (cwa + swa) > 0 else 0
    return total_loss / max(n, 1), cwa, swa, hwa, preds, gts


def train_model(train_loader, dev_loader, rule_name, epochs=3, lr=2e-3):
    model = TransCLS().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    experiment_data["single_dataset"][rule_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    }

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            opt.zero_grad()
            loss = criterion(model(batch["input"]), batch["label"])
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
        train_loss = total_loss / max(n, 1)
        # validation
        val_loss, cwa, swa, hwa, _, _ = evaluate(model, dev_loader)
        bwa = balanced_weighted_acc(cwa, swa)

        experiment_data["single_dataset"][rule_name]["losses"]["train"].append(
            train_loss
        )
        experiment_data["single_dataset"][rule_name]["losses"]["val"].append(val_loss)
        experiment_data["single_dataset"][rule_name]["metrics"]["val"].append(
            (cwa, swa, hwa, bwa)
        )

        print(
            f"Rule {rule_name} | Epoch {ep} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | CWA={cwa:.3f} | SWA={swa:.3f} | BWA={bwa:.3f}"
        )

    return model


# ----------------------  BUILD LOADERS  ---------------------------
loaders = {}
for r in rules:
    loaders[r] = {
        split: DataLoader(
            SPR(datasets[r][split]),
            batch_size=128,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "dev", "test"]
    }

# ---------------- 1)  SINGLE-DATASET EXPERIMENTS ------------------
for r in rules:
    print(f"\nTraining single-rule model on {r}")
    model = train_model(loaders[r]["train"], loaders[r]["dev"], r)
    test_loss, cwa, swa, hwa, preds, gts = evaluate(model, loaders[r]["test"])
    bwa = balanced_weighted_acc(cwa, swa)
    experiment_data["single_dataset"][r]["metrics"]["test"] = (cwa, swa, hwa, bwa)
    experiment_data["single_dataset"][r]["predictions"] = preds
    experiment_data["single_dataset"][r]["ground_truth"] = gts
    print(f"TEST {r} | CWA={cwa:.3f} | SWA={swa:.3f} | BWA={bwa:.3f} | HWA={hwa:.3f}")

# ---------------- 2)  UNION-ALL ABLATION --------------------------
print("\nTraining UNION model on all rules")
union_train = ConcatDataset([SPR(datasets[r]["train"]) for r in rules])
union_dev = ConcatDataset([SPR(datasets[r]["dev"]) for r in rules])
ut_loader = DataLoader(union_train, batch_size=128, shuffle=True, collate_fn=collate)
ud_loader = DataLoader(union_dev, batch_size=256, shuffle=False, collate_fn=collate)


def train_union(train_loader, dev_loader, epochs=3, lr=2e-3):
    model = TransCLS().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            opt.zero_grad()
            loss = criterion(model(batch["input"]), batch["label"])
            loss.backward()
            opt.step()
            tot += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
        val_loss, cwa, swa, hwa, _, _ = evaluate(model, dev_loader)
        bwa = balanced_weighted_acc(cwa, swa)
        print(
            f"UNION Epoch {ep} | train_loss={tot/max(n,1):.4f} | "
            f"val_loss={val_loss:.4f} | CWA={cwa:.3f} | SWA={swa:.3f} | BWA={bwa:.3f}"
        )
    return model


model_union = train_union(ut_loader, ud_loader)

for r in rules:
    print(f"Evaluating UNION model on {r}")
    t_loss, cwa, swa, hwa, preds, gts = evaluate(model_union, loaders[r]["test"])
    bwa = balanced_weighted_acc(cwa, swa)
    experiment_data["union_all"][r] = {
        "metrics": {"test": (cwa, swa, hwa, bwa)},
        "predictions": preds,
        "ground_truth": gts,
    }
    print(f"  TEST {r} | CWA={cwa:.3f} | SWA={swa:.3f} | BWA={bwa:.3f} | HWA={hwa:.3f}")

# -----------------------  SAVE RESULTS  ---------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
