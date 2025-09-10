# hidden_dim_tuning.py – single-file script
import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import load_dataset, DatasetDict

# --------------------------------------------------------------------------- #
# 0. House-keeping & GPU / working_dir                                         #
# --------------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------------- #
# 1. Locate SPR_BENCH dataset                                                  #
# --------------------------------------------------------------------------- #
def find_spr_bench_root() -> pathlib.Path:
    env_path = os.getenv("SPR_BENCH_ROOT")
    candidates = []
    if env_path:
        candidates.append(pathlib.Path(env_path))
    cwd = pathlib.Path.cwd()
    candidates.extend(
        [
            cwd / "SPR_BENCH",
            cwd.parent / "SPR_BENCH",
            cwd.parent.parent / "SPR_BENCH",
            pathlib.Path("/workspace/SPR_BENCH"),
            pathlib.Path("/data/SPR_BENCH"),
            pathlib.Path.home() / "SPR_BENCH",
            pathlib.Path.home() / "AI-Scientist-v2" / "SPR_BENCH",
        ]
    )
    for parent in cwd.parents:
        candidates.append(parent / "SPR_BENCH")
    for path in candidates:
        if (
            (path / "train.csv").exists()
            and (path / "dev.csv").exists()
            and (path / "test.csv").exists()
        ):
            print(f"Found SPR_BENCH dataset at: {path}")
            return path.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


DATA_PATH = find_spr_bench_root()


# --------------------------------------------------------------------------- #
# 2. Dataset helpers                                                          #
# --------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for sp in ["train", "dev", "test"]:
        dset[sp] = _load(f"{sp}.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [cw if t == p else 0 for cw, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [sw if t == p else 0 for sw, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def harmonic_csa(cwa: float, swa: float) -> float:
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# --------------------------------------------------------------------------- #
# 3. Reproducibility                                                           #
# --------------------------------------------------------------------------- #
def set_all_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_all_seeds(0)

# --------------------------------------------------------------------------- #
# 4. Load dataset                                                             #
# --------------------------------------------------------------------------- #
spr = load_spr_bench(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))


# --------------------------------------------------------------------------- #
# 5. Glyph clustering → histogram featuriser                                  #
# --------------------------------------------------------------------------- #
def glyph_vector(g: str):
    if len(g) >= 2:
        return [ord(g[0]) - 65, ord(g[1]) - 48]
    elif len(g) == 1:
        return [ord(g[0]) - 65, 0]
    return [0, 0]


all_glyphs = set(tok for seq in spr["train"]["sequence"] for tok in seq.strip().split())
vecs = np.array([glyph_vector(g) for g in all_glyphs])
k_clusters = 8
kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
cluster_labels = kmeans.fit_predict(vecs)
glyph_to_cluster = {g: c for g, c in zip(all_glyphs, cluster_labels)}


def seq_to_hist(seq: str) -> np.ndarray:
    h = np.zeros(k_clusters, dtype=np.float32)
    tokens = seq.strip().split()
    for tok in tokens:
        h[glyph_to_cluster.get(tok, 0)] += 1.0
    if len(tokens) > 0:
        h /= len(tokens)
    return h


# --------------------------------------------------------------------------- #
# 6. Torch Dataset                                                            #
# --------------------------------------------------------------------------- #
class SPRHistDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int]):
        self.x = np.stack([seq_to_hist(s) for s in sequences])
        self.y = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.from_numpy(self.x[idx]), "y": torch.tensor(self.y[idx])}


train_ds = SPRHistDataset(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRHistDataset(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRHistDataset(spr["test"]["sequence"], spr["test"]["label"])

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# --------------------------------------------------------------------------- #
# 7. Evaluation helper                                                        #
# --------------------------------------------------------------------------- #
def evaluate(model: nn.Module, loader, sequences) -> Dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    preds, gts = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            bx = batch["x"].to(device)
            by = batch["y"].to(device)
            logits = model(bx)
            loss = criterion(logits, by)
            total_loss += loss.item() * by.size(0)
            n += by.size(0)
            pred = logits.argmax(1)
            preds.extend(pred.cpu().tolist())
            gts.extend(by.cpu().tolist())
    avg_loss = total_loss / n
    cwa = color_weighted_accuracy(sequences, gts, preds)
    swa = shape_weighted_accuracy(sequences, gts, preds)
    hcs = harmonic_csa(cwa, swa)
    return {
        "loss": avg_loss,
        "CWA": cwa,
        "SWA": swa,
        "HCSA": hcs,
        "preds": preds,
        "gts": gts,
    }


# --------------------------------------------------------------------------- #
# 8. Hyperparameter sweep                                                     #
# --------------------------------------------------------------------------- #
hidden_dims = [32, 64, 128, 256, 512]
epochs = 10

experiment_data = {"hidden_dim": {"SPR_BENCH": {}}}

for hd in hidden_dims:
    print(f"\n===== Training with hidden_dim={hd} =====")
    set_all_seeds(0)  # start from same seed for fair comparison

    model = nn.Sequential(
        nn.Linear(k_clusters, hd), nn.ReLU(), nn.Linear(hd, num_classes)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    edict = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
    }
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, n_seen = 0.0, 0
        for batch in train_loader:
            bx = batch["x"].to(device)
            by = batch["y"].to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * by.size(0)
            n_seen += by.size(0)
        train_loss = tot_loss / n_seen
        val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])

        edict["losses"]["train"].append((epoch, train_loss))
        edict["losses"]["val"].append((epoch, val_stats["loss"]))
        edict["metrics"]["val"].append(
            (epoch, val_stats["CWA"], val_stats["SWA"], val_stats["HCSA"])
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"CWA={val_stats['CWA']:.3f} SWA={val_stats['SWA']:.3f} "
            f"HCSA={val_stats['HCSA']:.3f}"
        )

    # final evaluation
    dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
    test_final = evaluate(model, test_loader, spr["test"]["sequence"])
    for split, res in zip(["dev", "test"], [dev_final, test_final]):
        edict["predictions"][split] = res["preds"]
        edict["ground_truth"][split] = res["gts"]
        edict["metrics"][split if split == "dev" else "test"] = (
            res["CWA"],
            res["SWA"],
            res["HCSA"],
        )

    print(
        f"Dev  -> CWA {dev_final['CWA']:.3f} SWA {dev_final['SWA']:.3f} HCSA {dev_final['HCSA']:.3f}"
    )
    print(
        f"Test -> CWA {test_final['CWA']:.3f} SWA {test_final['SWA']:.3f} HCSA {test_final['HCSA']:.3f}"
    )

    experiment_data["hidden_dim"]["SPR_BENCH"][hd] = edict

# --------------------------------------------------------------------------- #
# 9. Persist results                                                          #
# --------------------------------------------------------------------------- #
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"\nSaved experiment data to {working_dir}/experiment_data.npy")
