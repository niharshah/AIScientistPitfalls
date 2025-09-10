import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import load_dataset, DatasetDict

# --------------------------------------------------------------------------- #
# 0. House-keeping                                                            #
# --------------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------------- #
# 1. Locate SPR_BENCH                                                         #
# --------------------------------------------------------------------------- #
def find_spr_bench_root() -> pathlib.Path:
    env = os.getenv("SPR_BENCH_ROOT")
    guesses = []
    if env:
        guesses.append(pathlib.Path(env))
    cwd = pathlib.Path.cwd()
    guesses.extend(
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
    for p in cwd.parents:
        guesses.append(p / "SPR_BENCH")
    for p in guesses:
        if (
            (p / "train.csv").exists()
            and (p / "dev.csv").exists()
            and (p / "test.csv").exists()
        ):
            print(f"Found SPR_BENCH dataset at: {p}")
            return p.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


DATA_PATH = find_spr_bench_root()


# --------------------------------------------------------------------------- #
# 2. Utilities                                                                #
# --------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(c if t == p else 0 for c, t, p in zip(w, y_true, y_pred)) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(s if t == p else 0 for s, t, p in zip(w, y_true, y_pred)) / (sum(w) or 1)


def harmonic_csa(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# --------------------------------------------------------------------------- #
# 3. Global experiment container                                              #
# --------------------------------------------------------------------------- #
experiment_data = {"batch_size": {}}  # will populate per-batch-size

# --------------------------------------------------------------------------- #
# 4. Prepare static dataset artefacts                                         #
# --------------------------------------------------------------------------- #
spr = load_spr_bench(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))


# glyph clustering (once – reused for every run)
def glyph_vector(g: str):
    return (
        [ord(g[0]) - 65, ord(g[1]) - 48]
        if len(g) >= 2
        else ([ord(g[0]) - 65, 0] if len(g) == 1 else [0, 0])
    )


all_glyphs = set(tok for seq in spr["train"]["sequence"] for tok in seq.strip().split())
vecs = np.array([glyph_vector(g) for g in all_glyphs])
k_clusters = 8
kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
glyph_to_cluster = {g: c for g, c in zip(all_glyphs, kmeans.fit_predict(vecs))}


def seq_to_hist(seq: str) -> np.ndarray:
    hist = np.zeros(k_clusters, dtype=np.float32)
    toks = seq.strip().split()
    for t in toks:
        hist[glyph_to_cluster.get(t, 0)] += 1.0
    return hist / len(toks) if toks else hist


class SPRHistDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int]):
        self.x = np.stack([seq_to_hist(s) for s in sequences])
        self.y = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.from_numpy(self.x[idx]), "y": torch.tensor(self.y[idx])}


# --------------------------------------------------------------------------- #
# 5. Evaluation helper                                                        #
# --------------------------------------------------------------------------- #
def evaluate(model: nn.Module, loader, sequences) -> Dict[str, float]:
    model.eval()
    total_loss, seen = 0.0, 0
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            total_loss += loss.item() * batch["y"].size(0)
            seen += batch["y"].size(0)
            p = logits.argmax(1)
            preds.extend(p.cpu().tolist())
            gts.extend(batch["y"].cpu().tolist())
    avg_loss = total_loss / seen
    cwa, swa = color_weighted_accuracy(sequences, gts, preds), shape_weighted_accuracy(
        sequences, gts, preds
    )
    return {
        "loss": avg_loss,
        "CWA": cwa,
        "SWA": swa,
        "HCSA": harmonic_csa(cwa, swa),
        "preds": preds,
        "gts": gts,
    }


# --------------------------------------------------------------------------- #
# 6. Hyper-parameter sweep                                                    #
# --------------------------------------------------------------------------- #
batch_sizes = [32, 64, 128, 256]
epochs = 10
for bs in batch_sizes:
    print(f"\n=== Training with batch_size={bs} ===")
    # ----------------------------------------------------------------------- #
    # DataLoaders                                                             #
    # ----------------------------------------------------------------------- #
    train_ds = SPRHistDataset(spr["train"]["sequence"], spr["train"]["label"])
    dev_ds = SPRHistDataset(spr["dev"]["sequence"], spr["dev"]["label"])
    test_ds = SPRHistDataset(spr["test"]["sequence"], spr["test"]["label"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    # ----------------------------------------------------------------------- #
    # Model, criterion, optimiser                                             #
    # ----------------------------------------------------------------------- #
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model = nn.Sequential(
        nn.Linear(k_clusters, 128), nn.ReLU(), nn.Linear(128, num_classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # container for this setting
    exp_entry = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
    }

    # ----------------------------------------------------------------------- #
    # Training loop                                                           #
    # ----------------------------------------------------------------------- #
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch["y"].size(0)
            seen += batch["y"].size(0)
        train_loss = tot_loss / seen
        exp_entry["losses"]["train"].append((ep, train_loss))

        val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])
        exp_entry["losses"]["val"].append((ep, val_stats["loss"]))
        exp_entry["metrics"]["val"].append(
            (ep, val_stats["CWA"], val_stats["SWA"], val_stats["HCSA"])
        )
        print(
            f"bs={bs} | Epoch {ep}: "
            f"train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} "
            f"HCSA={val_stats['HCSA']:.3f}"
        )

    # ----------------------------------------------------------------------- #
    # Final dev/test evaluation                                               #
    # ----------------------------------------------------------------------- #
    dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
    test_final = evaluate(model, test_loader, spr["test"]["sequence"])
    for split, res in [("dev", dev_final), ("test", test_final)]:
        exp_entry["predictions"][split] = res["preds"]
        exp_entry["ground_truth"][split] = res["gts"]
    print(
        f"bs={bs} | Dev HCSA={dev_final['HCSA']:.3f} | "
        f"Test HCSA={test_final['HCSA']:.3f}"
    )

    # store
    experiment_data["batch_size"][f"bs_{bs}"] = exp_entry

# --------------------------------------------------------------------------- #
# 7. Save experiment                                                          #
# --------------------------------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to {working_dir}/experiment_data.npy")
