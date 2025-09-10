import os, pathlib, random, copy, numpy as np, torch, torch.nn as nn, torch.optim as optim
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------ #
# 0. House-keeping                                                   #
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ------------------------------------------------------------------ #
# 1. Locate SPR_BENCH                                                #
# ------------------------------------------------------------------ #
def find_spr_bench_root() -> pathlib.Path:
    env = os.getenv("SPR_BENCH_ROOT")
    if env:
        p = pathlib.Path(env)
        if (p / "train.csv").exists():
            return p
    cwd = pathlib.Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        cand = parent / "SPR_BENCH"
        if (cand / "train.csv").exists():
            return cand
    raise FileNotFoundError("SPR_BENCH not found; set $SPR_BENCH_ROOT")


DATA_PATH = find_spr_bench_root()
print(f"Loading SPR_BENCH from {DATA_PATH}")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _ld(s) for s in ["train", "dev", "test"]})


spr = load_spr_bench(DATA_PATH)


# ------------------------------------------------------------------ #
# 2. Helper functions & metrics                                      #
# ------------------------------------------------------------------ #
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    s = sum(w)
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / s if s else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    s = sum(w)
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / s if s else 0.0


def harmonic_csa(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ------------------------------------------------------------------ #
# 3. Glyph clustering                                                #
# ------------------------------------------------------------------ #
def glyph_vec(g: str):
    # simple 2-d ASCII embedding: uppercase A-Z, digits 0-9
    if len(g) == 1:
        return [ord(g[0]) - 65, -1]  # no color
    return [ord(g[0]) - 65, ord(g[1]) - 48]


all_glyphs = sorted({tok for seq in spr["train"]["sequence"] for tok in seq.split()})
vecs = np.array([glyph_vec(g) for g in all_glyphs], dtype=np.float32)

k_clusters = 16
kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
glyph_clusters = kmeans.fit_predict(vecs)
glyph_to_cluster = {g: c for g, c in zip(all_glyphs, glyph_clusters)}

# pairs seen in training for SNWA
seen_pairs = {(g, glyph_to_cluster[g]) for g in all_glyphs}


# ------------------------------------------------------------------ #
# 4. Sequence vectorisation                                          #
# ------------------------------------------------------------------ #
def seq_to_vector(seq: str) -> np.ndarray:
    tokens = seq.strip().split()
    hist = np.zeros(k_clusters, dtype=np.float32)
    for tok in tokens:
        hist[glyph_to_cluster.get(tok, 0)] += 1.0
    if tokens:
        hist /= len(tokens)
    extras = np.array(
        [count_color_variety(seq) / 10.0, count_shape_variety(seq) / 10.0],
        dtype=np.float32,
    )
    return np.concatenate([hist, extras])


def sequence_novelty_weighted_accuracy(seqs, y_true, y_pred):
    total_w, correct_w = 0.0, 0.0
    for s, t, p in zip(seqs, y_true, y_pred):
        toks = s.strip().split()
        novel = sum(
            1 for tok in toks if (tok, glyph_to_cluster.get(tok, 0)) not in seen_pairs
        )
        total = len(toks)
        novelty_ratio = novel / total if total else 0.0
        w = 1.0 + novelty_ratio
        total_w += w
        if t == p:
            correct_w += w
    return correct_w / total_w if total_w else 0.0


# ------------------------------------------------------------------ #
# 5. Torch Dataset                                                   #
# ------------------------------------------------------------------ #
class SPRDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int]):
        self.x = np.stack([seq_to_vector(s) for s in sequences])
        self.y = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.from_numpy(self.x[idx]), "y": torch.tensor(self.y[idx])}


train_ds = SPRDataset(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRDataset(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRDataset(spr["test"]["sequence"], spr["test"]["label"])

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

num_classes = len(set(spr["train"]["label"]))
input_dim = k_clusters + 2

# ------------------------------------------------------------------ #
# 6. Model                                                           #
# ------------------------------------------------------------------ #
model = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------------------------------------------ #
# 7. Evaluation helper                                               #
# ------------------------------------------------------------------ #
def evaluate(model: nn.Module, loader, sequences):
    model.eval()
    preds, gts, losses = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            losses += loss.item() * batch["y"].size(0)
            pred = logits.argmax(1)
            preds.extend(pred.cpu().tolist())
            gts.extend(batch["y"].cpu().tolist())
    n = len(gts)
    avg_loss = losses / n
    cwa = color_weighted_accuracy(sequences, gts, preds)
    swa = shape_weighted_accuracy(sequences, gts, preds)
    hcs = harmonic_csa(cwa, swa)
    snwa = sequence_novelty_weighted_accuracy(sequences, gts, preds)
    return {
        "loss": avg_loss,
        "CWA": cwa,
        "SWA": swa,
        "HCSA": hcs,
        "SNWA": snwa,
        "preds": preds,
        "gts": gts,
    }


# ------------------------------------------------------------------ #
# 8. Training loop with early stopping                               #
# ------------------------------------------------------------------ #
max_epochs, patience = 30, 5
best_snwa, best_state, epochs_since = -1.0, None, 0

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
    }
}

for epoch in range(1, max_epochs + 1):
    model.train()
    tot_loss, cnt = 0.0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch["y"].size(0)
        cnt += batch["y"].size(0)
    train_loss = tot_loss / cnt
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))

    # validation
    val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_stats["loss"]))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (
            epoch,
            val_stats["CWA"],
            val_stats["SWA"],
            val_stats["HCSA"],
            val_stats["SNWA"],
        )
    )
    print(
        f'Epoch {epoch}: validation_loss = {val_stats["loss"]:.4f} '
        f'SNWA = {val_stats["SNWA"]:.3f}'
    )

    # early stopping on SNWA
    if val_stats["SNWA"] > best_snwa + 1e-6:
        best_snwa = val_stats["SNWA"]
        best_state = copy.deepcopy(model.state_dict())
        epochs_since = 0
    else:
        epochs_since += 1
    if epochs_since >= patience:
        print("Early stopping triggered.")
        break

# restore best model
if best_state:
    model.load_state_dict(best_state)

# ------------------------------------------------------------------ #
# 9. Final evaluation & save                                         #
# ------------------------------------------------------------------ #
dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
test_final = evaluate(model, test_loader, spr["test"]["sequence"])

experiment_data["SPR_BENCH"]["predictions"]["dev"] = dev_final["preds"]
experiment_data["SPR_BENCH"]["ground_truth"]["dev"] = dev_final["gts"]
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_final["preds"]
experiment_data["SPR_BENCH"]["ground_truth"]["test"] = test_final["gts"]

print(
    f"DEV  -> CWA {dev_final['CWA']:.3f} | SWA {dev_final['SWA']:.3f} | "
    f"HCSA {dev_final['HCSA']:.3f} | SNWA {dev_final['SNWA']:.3f}"
)
print(
    f"TEST -> CWA {test_final['CWA']:.3f} | SWA {test_final['SWA']:.3f} | "
    f"HCSA {test_final['HCSA']:.3f} | SNWA {test_final['SNWA']:.3f}"
)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"All metrics saved to {working_dir}/experiment_data.npy")
