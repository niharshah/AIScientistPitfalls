import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import load_dataset, DatasetDict

# --------------------------------------------------------------------------- #
# 0. House-keeping & working directory                                         #
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
    for path in candidates:
        if (
            (path / "train.csv").exists()
            and (path / "dev.csv").exists()
            and (path / "test.csv").exists()
        ):
            print(f"Found SPR_BENCH dataset at: {path}")
            return path.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found")


DATA_PATH = find_spr_bench_root()


# --------------------------------------------------------------------------- #
# 2. Dataset utilities                                                         #
# --------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_file):
        return load_dataset(
            "csv",
            data_files=str(root / csv_file),
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
    return sum(cw for cw, t, p in zip(w, y_true, y_pred) if t == p) / (sum(w) + 1e-8)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(sw for sw, t, p in zip(w, y_true, y_pred) if t == p) / (sum(w) + 1e-8)


def harmonic_csa(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# --------------------------------------------------------------------------- #
# 3. Prepare data                                                              #
# --------------------------------------------------------------------------- #
spr = load_spr_bench(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))


def glyph_vector(g):
    if len(g) >= 2:
        return [ord(g[0]) - 65, ord(g[1]) - 48]
    if len(g) == 1:
        return [ord(g[0]) - 65, 0]
    return [0, 0]


all_glyphs = set(tok for seq in spr["train"]["sequence"] for tok in seq.strip().split())
vecs = np.array([glyph_vector(g) for g in all_glyphs])
k_clusters = 8
kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10).fit(vecs)
glyph_to_cluster = {g: c for g, c in zip(all_glyphs, kmeans.labels_)}


def seq_to_hist(seq):
    h = np.zeros(k_clusters, dtype=np.float32)
    toks = seq.strip().split()
    for t in toks:
        h[glyph_to_cluster.get(t, 0)] += 1.0
    if toks:
        h /= len(toks)
    return h


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


# --------------------------------------------------------------------------- #
# 4. Evaluation helper                                                         #
# --------------------------------------------------------------------------- #
def evaluate(model, loader, sequences):
    model.eval()
    total_loss, n = 0.0, 0
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            pred = logits.argmax(1).cpu().tolist()
            preds.extend(pred)
            gts.extend(y.cpu().tolist())
    avg_loss = total_loss / n
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
# 5. Hyper-parameter sweep                                                     #
# --------------------------------------------------------------------------- #
batch_sizes = [32, 64, 128, 256, 512]
epochs = 10
experiment_data = {"batch_size": {"SPR_BENCH": {}}}

for bs in batch_sizes:
    print(f"\n=== Training with batch size {bs} ===")
    # reproducibility per run
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    model = nn.Sequential(
        nn.Linear(k_clusters, 128), nn.ReLU(), nn.Linear(128, num_classes)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    run_log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
    }
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, n = 0.0, 0
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = tot_loss / n
        val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])

        run_log["losses"]["train"].append((epoch, train_loss))
        run_log["losses"]["val"].append((epoch, val_stats["loss"]))
        run_log["metrics"]["val"].append(
            (epoch, val_stats["CWA"], val_stats["SWA"], val_stats["HCSA"])
        )

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | HCSA={val_stats['HCSA']:.3f}"
        )

    # final dev/test evaluation
    dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
    test_final = evaluate(model, test_loader, spr["test"]["sequence"])
    run_log["predictions"]["dev"] = dev_final["preds"]
    run_log["ground_truth"]["dev"] = dev_final["gts"]
    run_log["predictions"]["test"] = test_final["preds"]
    run_log["ground_truth"]["test"] = test_final["gts"]
    run_log["final_dev_metrics"] = (
        dev_final["CWA"],
        dev_final["SWA"],
        dev_final["HCSA"],
    )
    run_log["final_test_metrics"] = (
        test_final["CWA"],
        test_final["SWA"],
        test_final["HCSA"],
    )

    experiment_data["batch_size"]["SPR_BENCH"][bs] = run_log
    print(
        f"Completed batch size {bs}: Dev HCSA={dev_final['HCSA']:.3f} | Test HCSA={test_final['HCSA']:.3f}"
    )

# --------------------------------------------------------------------------- #
# 6. Save experiment data                                                     #
# --------------------------------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved all results to {working_dir}/experiment_data.npy")
