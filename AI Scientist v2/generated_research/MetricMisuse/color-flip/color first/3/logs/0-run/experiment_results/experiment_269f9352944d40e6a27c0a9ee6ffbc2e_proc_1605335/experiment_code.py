import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import load_dataset, DatasetDict

# --------------------------------------------------------------------------- #
# 0. House-keeping & GPU / working_dir                                        #
# --------------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------------- #
# 1. Locate SPR_BENCH                                                         #
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
            print(f"Found SPR_BENCH at {p}")
            return p.resolve()
    raise FileNotFoundError(
        "SPR_BENCH not found. Set $SPR_BENCH_ROOT or place csvs accordingly."
    )


DATA_PATH = find_spr_bench_root()


# --------------------------------------------------------------------------- #
# 2. Utils                                                                    #
# --------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):  # always "train" split in HuggingFace loader
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
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
    return sum(c if t == p else 0 for c, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(s if t == p else 0 for s, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def harmonic_csa(cwa: float, swa: float) -> float:
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# --------------------------------------------------------------------------- #
# 3. Data                                                                     #
# --------------------------------------------------------------------------- #
spr = load_spr_bench(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))


# glyph clustering to histograms
def glyph_vector(g: str):
    return [ord(g[0]) - 65 if g else 0, ord(g[1]) - 48 if len(g) >= 2 else 0]


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


class SPRHistDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int]):
        self.x = np.stack([seq_to_hist(s) for s in sequences])
        self.y = np.asarray(labels, dtype=np.int64)

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
# 4. Evaluation helper                                                        #
# --------------------------------------------------------------------------- #
def evaluate(model, loader, sequences):
    model.eval()
    tot_loss, n = 0.0, 0
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            tot_loss += loss.item() * y.size(0)
            n += y.size(0)
            p = logits.argmax(1)
            preds.extend(p.cpu().tolist())
            gts.extend(y.cpu().tolist())
    avg_loss = tot_loss / n
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
# 5. Dropout sweep                                                            #
# --------------------------------------------------------------------------- #
dropout_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
epochs = 10
experiment_data = {"dropout_prob": {}}

for p_drop in dropout_grid:
    print(f"\n=== Training with dropout_prob={p_drop} ===")
    # reproducibility per run
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(k_clusters, 128),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
        nn.Linear(128, num_classes),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    key = f"{p_drop:.2f}"
    experiment_data["dropout_prob"][key] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": {"dev": [], "test": []},
            "ground_truth": {"dev": [], "test": []},
        }
    }
    for epoch in range(1, epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            tot += loss.item() * y.size(0)
            seen += y.size(0)
        train_loss = tot / seen
        experiment_data["dropout_prob"][key]["SPR_BENCH"]["losses"]["train"].append(
            (epoch, train_loss)
        )
        # validation
        val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])
        experiment_data["dropout_prob"][key]["SPR_BENCH"]["losses"]["val"].append(
            (epoch, val_stats["loss"])
        )
        experiment_data["dropout_prob"][key]["SPR_BENCH"]["metrics"]["val"].append(
            (epoch, val_stats["CWA"], val_stats["SWA"], val_stats["HCSA"])
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_stats['loss']:.4f} "
            f"CWA={val_stats['CWA']:.3f} SWA={val_stats['SWA']:.3f} HCSA={val_stats['HCSA']:.3f}"
        )
    # final dev/test eval
    dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
    test_final = evaluate(model, test_loader, spr["test"]["sequence"])
    exp = experiment_data["dropout_prob"][key]["SPR_BENCH"]
    exp["predictions"]["dev"], exp["ground_truth"]["dev"] = (
        dev_final["preds"],
        dev_final["gts"],
    )
    exp["predictions"]["test"], exp["ground_truth"]["test"] = (
        test_final["preds"],
        test_final["gts"],
    )
    print(f"DEV  HCSA={dev_final['HCSA']:.3f} | TEST HCSA={test_final['HCSA']:.3f}")

# --------------------------------------------------------------------------- #
# 6. Save everything                                                          #
# --------------------------------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved all results to {working_dir}/experiment_data.npy")
