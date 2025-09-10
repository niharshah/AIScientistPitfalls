import os, pathlib, random, copy, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# --------------------------------------------------------------------------- #
# House-keeping                                                               #
# --------------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Locate SPR_BENCH automatically                                              #
# --------------------------------------------------------------------------- #
def find_spr_bench_root() -> pathlib.Path:
    env = os.getenv("SPR_BENCH_ROOT")
    if env and pathlib.Path(env).exists():
        return pathlib.Path(env)
    cwd = pathlib.Path.cwd()
    for p in [cwd, *cwd.parents]:
        cand = p / "SPR_BENCH"
        if (cand / "train.csv").exists():
            return cand
    raise FileNotFoundError("SPR_BENCH folder not found.")


DATA_PATH = find_spr_bench_root()
print("SPR_BENCH:", DATA_PATH)


# --------------------------------------------------------------------------- #
# Benchmark helpers                                                           #
# --------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({sp: _load(sp) for sp in ["train", "dev", "test"]})


def count_color_variety(seq):
    return len(set(t[1] for t in seq.strip().split() if len(t) > 1))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.strip().split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0


# --------------------------------------------------------------------------- #
# Load dataset                                                                #
# --------------------------------------------------------------------------- #
spr = load_spr_bench(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))
train_sequences = spr["train"]["sequence"]


# --------------------------------------------------------------------------- #
# Glyph clustering                                                            #
# --------------------------------------------------------------------------- #
def glyph_vec(g):
    return [ord(g[0]) - 65, int(g[1]) if len(g) > 1 and g[1].isdigit() else 0]


glyph_set = sorted({tok for seq in train_sequences for tok in seq.strip().split()})
vecs = np.array([glyph_vec(g) for g in glyph_set])
k_clusters = 12
glyph_clusters = KMeans(n_clusters=k_clusters, random_state=0, n_init=10).fit_predict(
    vecs
)
g2c = {g: c for g, c in zip(glyph_set, glyph_clusters)}

# Pre-compute train glyph-cluster pairs for SNWA novelty
train_pairs = {(g, g2c[g]) for g in glyph_set}


# --------------------------------------------------------------------------- #
# Feature extraction                                                          #
# --------------------------------------------------------------------------- #
def seq_to_feat(seq: str):
    tokens = seq.strip().split()
    hist = np.zeros(k_clusters, dtype=np.float32)
    for t in tokens:
        hist[g2c.get(t, 0)] += 1.0
    if tokens:
        hist /= len(tokens)
    cvar = count_color_variety(seq) / max(1, len(tokens))
    svar = count_shape_variety(seq) / max(1, len(tokens))
    return np.concatenate([hist, [cvar, svar]]).astype(np.float32)


def seq_snwa_weight(seq):
    tokens = seq.strip().split()
    if not tokens:
        return 1.0
    novel = sum((t, g2c.get(t, 0)) not in train_pairs for t in tokens)
    novelty_ratio = novel / len(tokens)
    return 1.0 + novelty_ratio


# --------------------------------------------------------------------------- #
# Torch dataset                                                               #
# --------------------------------------------------------------------------- #
class SPRFeatDS(Dataset):
    def __init__(self, sequences, labels):
        self.x = np.stack([seq_to_feat(s) for s in sequences])
        self.y = np.array(labels, dtype=np.int64)
        self.seqs = sequences

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.tensor(self.y[idx]),
            "seq": self.seqs[idx],
        }


train_ds = SPRFeatDS(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRFeatDS(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRFeatDS(spr["test"]["sequence"], spr["test"]["label"])

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
model = nn.Sequential(
    nn.Linear(k_clusters + 2, 128), nn.ReLU(), nn.Linear(128, num_classes)
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-3)


# --------------------------------------------------------------------------- #
# Metric utils                                                                #
# --------------------------------------------------------------------------- #
def evaluate(model, loader):
    model.eval()
    preds, gts, seqs = [], [], []
    tot_loss, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            tot_loss += loss.item() * batch["y"].size(0)
            n += batch["y"].size(0)
            pred = out.argmax(1)
            preds.extend(pred.cpu().tolist())
            gts.extend(batch["y"].cpu().tolist())
            seqs.extend(batch["seq"])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    # SNWA
    weights = [seq_snwa_weight(s) for s in seqs]
    corr = [w if a == b else 0 for w, a, b in zip(weights, gts, preds)]
    snwa = sum(corr) / sum(weights) if sum(weights) else 0
    return {
        "loss": tot_loss / n,
        "CWA": cwa,
        "SWA": swa,
        "SNWA": snwa,
        "preds": preds,
        "gts": gts,
    }


# --------------------------------------------------------------------------- #
# Training loop with early stopping on dev SNWA                               #
# --------------------------------------------------------------------------- #
max_epochs, patience = 40, 5
best_snwa, best_state, wait = -1, None, 0

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {},
        "ground_truth": {},
    }
}

for epoch in range(1, max_epochs + 1):
    # Train
    model.train()
    tot_loss, n = 0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["x"])
        loss = criterion(out, batch["y"])
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch["y"].size(0)
        n += batch["y"].size(0)
    train_loss = tot_loss / n

    # Evaluate
    val_stats = evaluate(model, dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_stats["loss"]))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (epoch, val_stats["CWA"], val_stats["SWA"], val_stats["SNWA"])
    )
    print(
        f'Epoch {epoch}: val_loss={val_stats["loss"]:.4f} CWA={val_stats["CWA"]:.3f} '
        f'SWA={val_stats["SWA"]:.3f} SNWA={val_stats["SNWA"]:.3f}'
    )

    # Early stopping on SNWA
    if val_stats["SNWA"] > best_snwa + 1e-6:
        best_snwa = val_stats["SNWA"]
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping")
        break

# --------------------------------------------------------------------------- #
# Final evaluation                                                            #
# --------------------------------------------------------------------------- #
model.load_state_dict(best_state)
dev_final = evaluate(model, dev_loader)
test_final = evaluate(model, test_loader)

print(f'\nBest Dev SNWA={dev_final["SNWA"]:.3f}  |  Test SNWA={test_final["SNWA"]:.3f}')
print(f'Dev CWA={dev_final["CWA"]:.3f}  SWA={dev_final["SWA"]:.3f}')
print(f'Test CWA={test_final["CWA"]:.3f}  SWA={test_final["SWA"]:.3f}')

experiment_data["SPR_BENCH"]["predictions"]["dev"] = dev_final["preds"]
experiment_data["SPR_BENCH"]["ground_truth"]["dev"] = dev_final["gts"]
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_final["preds"]
experiment_data["SPR_BENCH"]["ground_truth"]["test"] = test_final["gts"]

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
