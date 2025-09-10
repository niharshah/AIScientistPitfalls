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
# 1. Locate SPR_BENCH                                                          #
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


# --------------------------------------------------------------------------- #
# 2. Benchmark utilities                                                      #
# --------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


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


# --------------------------------------------------------------------------- #
# 3. Seeds                                                                    #
# --------------------------------------------------------------------------- #
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# --------------------------------------------------------------------------- #
# 4. Load dataset                                                             #
# --------------------------------------------------------------------------- #
spr = load_spr_bench(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))


# --------------------------------------------------------------------------- #
# 5. Glyph clustering                                                         #
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
kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10).fit(vecs)
glyph_to_cluster = {g: c for g, c in zip(all_glyphs, kmeans.labels_)}


def seq_to_hist(seq: str) -> np.ndarray:
    h = np.zeros(k_clusters, dtype=np.float32)
    toks = seq.strip().split()
    for tok in toks:
        h[glyph_to_cluster.get(tok, 0)] += 1.0
    if toks:
        h /= len(toks)
    return h


# --------------------------------------------------------------------------- #
# 6. Torch Dataset / DataLoader                                               #
# --------------------------------------------------------------------------- #
class SPRHistDataset(Dataset):
    def __init__(self, seqs: List[str], labels: List[int]):
        self.x = np.stack([seq_to_hist(s) for s in seqs])
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
# 7. Helper: evaluation                                                       #
# --------------------------------------------------------------------------- #
def evaluate(model, loader, sequences) -> Dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    preds, gts = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            total_loss += loss.item() * batch["y"].size(0)
            n += batch["y"].size(0)
            p = logits.argmax(1)
            preds.extend(p.cpu().tolist())
            gts.extend(batch["y"].cpu().tolist())
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
# 8. Hyperparameter tuning: learning_rate                                     #
# --------------------------------------------------------------------------- #
lrs = [3e-4, 1e-3, 3e-3]
epochs = 10

experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {
            "metrics": {"train": {}, "val": {}},
            "losses": {"train": {}, "val": {}},
            "predictions": {"dev": {}, "test": {}},
            "ground_truth": {"dev": {}, "test": {}},
            "best_lr": None,
        }
    }
}

best_hcs, best_lr = -1.0, None

for lr in lrs:
    print(f"\n=== Training with learning rate: {lr} ===")
    model = nn.Sequential(
        nn.Linear(k_clusters, 128), nn.ReLU(), nn.Linear(128, num_classes)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["train"][lr] = []
    experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["val"][lr] = []
    experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["val"][lr] = []

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, nseen = 0.0, 0
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
            nseen += batch["y"].size(0)
        train_loss = tot_loss / nseen
        val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])

        experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["train"][lr].append(
            (epoch, train_loss)
        )
        experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["val"][lr].append(
            (epoch, val_stats["loss"])
        )
        experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["val"][lr].append(
            (epoch, val_stats["CWA"], val_stats["SWA"], val_stats["HCSA"])
        )

        print(
            f"Epoch {epoch:2d}: train_loss={train_loss:.4f} | val_loss={val_stats['loss']:.4f} | "
            f"CWA={val_stats['CWA']:.3f} SWA={val_stats['SWA']:.3f} HCSA={val_stats['HCSA']:.3f}"
        )

    # final dev/test
    dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
    test_final = evaluate(model, test_loader, spr["test"]["sequence"])

    experiment_data["learning_rate"]["SPR_BENCH"]["predictions"]["dev"][lr] = dev_final[
        "preds"
    ]
    experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"]["dev"][lr] = (
        dev_final["gts"]
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["predictions"]["test"][lr] = (
        test_final["preds"]
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"]["test"][lr] = (
        test_final["gts"]
    )

    print(
        f"DEV  -> CWA: {dev_final['CWA']:.3f}, SWA: {dev_final['SWA']:.3f}, HCSA: {dev_final['HCSA']:.3f}"
    )
    print(
        f"TEST -> CWA: {test_final['CWA']:.3f}, SWA: {test_final['SWA']:.3f}, HCSA: {test_final['HCSA']:.3f}"
    )

    if dev_final["HCSA"] > best_hcs:
        best_hcs, best_lr = dev_final["HCSA"], lr

    # free cuda mem
    del model
    torch.cuda.empty_cache()

experiment_data["learning_rate"]["SPR_BENCH"]["best_lr"] = best_lr
print(f"\nBest learning rate by dev HCSA: {best_lr}")

# --------------------------------------------------------------------------- #
# 9. Save experiment data                                                     #
# --------------------------------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {working_dir}/experiment_data.npy")
