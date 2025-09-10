import os, pathlib, random, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import load_dataset, DatasetDict

# ------------------------ 0. House-keeping & GPU --------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------ 1. Locate SPR_BENCH ------------------------------ #
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

    for p in candidates:
        if (
            (p / "train.csv").exists()
            and (p / "dev.csv").exists()
            and (p / "test.csv").exists()
        ):
            print(f"Found SPR_BENCH dataset at: {p}")
            return p.resolve()

    raise FileNotFoundError("SPR_BENCH dataset not found")


DATA_PATH = find_spr_bench_root()


# ------------------------ 2. Data loading helpers -------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [cw if t == p else 0 for cw, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [sw if t == p else 0 for sw, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def harmonic_csa(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ------------------------ 3. Reproducibility ------------------------------- #
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ------------------------ 4. Load dataset ---------------------------------- #
spr = load_spr_bench(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))


# ------------------------ 5. Glyph clustering ------------------------------ #
def glyph_vector(g: str):
    if len(g) >= 2:
        return [ord(g[0]) - 65, ord(g[1]) - 48]
    if len(g) == 1:
        return [ord(g[0]) - 65, 0]
    return [0, 0]


all_glyphs = set(tok for seq in spr["train"]["sequence"] for tok in seq.strip().split())
vecs = np.array([glyph_vector(g) for g in all_glyphs])
k_clusters = 8
kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
glyph_to_cluster = {g: c for g, c in zip(all_glyphs, kmeans.fit_predict(vecs))}


def seq_to_hist(seq: str) -> np.ndarray:
    h = np.zeros(k_clusters, dtype=np.float32)
    tokens = seq.strip().split()
    for tok in tokens:
        h[glyph_to_cluster.get(tok, 0)] += 1.0
    if tokens:
        h /= len(tokens)
    return h


# ------------------------ 6. Torch Dataset --------------------------------- #
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

# ------------------------ 7. DataLoaders ----------------------------------- #
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# ------------------------ 8. Evaluation routine ---------------------------- #
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader, sequences) -> Dict[str, float]:
    model.eval()
    total_loss, seen = 0.0, 0
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            total_loss += loss.item() * batch["y"].size(0)
            seen += batch["y"].size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch["y"].cpu().tolist())
    avg_loss = total_loss / seen
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


# ------------------------ 9. Experiment storage ---------------------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
    }
}

# ------------------------ 10. Hyper-parameter sweep ------------------------ #
lr_grid = [5e-4, 1e-3, 3e-3]
epochs = 10
patience = 3
best_overall = {"HCSA": -1}

for lr in lr_grid:
    lr_key = f"{lr:.0e}"
    print(f"\n=== Training with learning rate {lr_key} ===")
    # fresh seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    model = nn.Sequential(
        nn.Linear(k_clusters, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_dev_hcsa = -1
    best_state = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        # -------------------- Training -------------------- #
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch["y"].size(0)
            seen += batch["y"].size(0)
        train_loss = total_loss / seen

        # -------------------- Validation ------------------ #
        val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])
        print(
            f"Epoch {epoch:02d} | lr={lr_key} | "
            f"train_loss={train_loss:.4f} | val_loss={val_stats['loss']:.4f} "
            f"| HCSA={val_stats['HCSA']:.3f}"
        )

        # Save metrics
        experiment_data["SPR_BENCH"]["losses"]["train"].append(
            (lr_key, epoch, train_loss)
        )
        experiment_data["SPR_BENCH"]["losses"]["val"].append(
            (lr_key, epoch, val_stats["loss"])
        )
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(
            (lr_key, epoch, None, None, None)
        )
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            (lr_key, epoch, val_stats["CWA"], val_stats["SWA"], val_stats["HCSA"])
        )

        # ---------------- Early stopping ------------------ #
        if val_stats["HCSA"] > best_dev_hcsa:
            best_dev_hcsa = val_stats["HCSA"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # -------------------- Reload best model -------------- #
    model.load_state_dict(best_state)

    dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
    test_final = evaluate(model, test_loader, spr["test"]["sequence"])

    # -------------------- Store predictions -------------- #
    experiment_data["SPR_BENCH"]["predictions"]["dev"].append(dev_final["preds"])
    experiment_data["SPR_BENCH"]["predictions"]["test"].append(test_final["preds"])
    experiment_data["SPR_BENCH"]["ground_truth"]["dev"].append(dev_final["gts"])
    experiment_data["SPR_BENCH"]["ground_truth"]["test"].append(test_final["gts"])

    print(f"Best dev HCSA={dev_final['HCSA']:.3f} | Test HCSA={test_final['HCSA']:.3f}")

    if dev_final["HCSA"] > best_overall.get("HCSA", -1):
        best_overall = {
            "lr": lr_key,
            "HCSA": dev_final["HCSA"],
            "test_HCSA": test_final["HCSA"],
        }

print(
    f"\nBest LR according to dev HCSA: {best_overall['lr']} "
    f"(dev HCSA={best_overall['HCSA']:.3f}, test HCSA={best_overall['test_HCSA']:.3f})"
)

# ------------------------ 11. Persist results ------------------------------ #
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Saved experiment data to {working_dir}/experiment_data.npy")
