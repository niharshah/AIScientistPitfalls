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
print("Using device:", device)


# --------------------------------------------------------------------------- #
# 1. Locate SPR_BENCH                                                         #
# --------------------------------------------------------------------------- #
def find_spr_bench_root() -> pathlib.Path:
    env = os.getenv("SPR_BENCH_ROOT")
    guesses = [pathlib.Path(p) for p in ([env] if env else [])]
    cwd = pathlib.Path.cwd()
    guesses += [
        cwd / "SPR_BENCH",
        cwd.parent / "SPR_BENCH",
        cwd.parent.parent / "SPR_BENCH",
        pathlib.Path("/workspace/SPR_BENCH"),
        pathlib.Path("/data/SPR_BENCH"),
        pathlib.Path.home() / "SPR_BENCH",
        pathlib.Path.home() / "AI-Scientist-v2" / "SPR_BENCH",
    ] + [p / "SPR_BENCH" for p in cwd.parents]
    for g in guesses:
        if (
            g
            and (g / "train.csv").exists()
            and (g / "dev.csv").exists()
            and (g / "test.csv").exists()
        ):
            print("Found dataset at:", g)
            return g.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


DATA_PATH = find_spr_bench_root()


# --------------------------------------------------------------------------- #
# 2. Benchmark utilities                                                      #
# --------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({sp: _ld(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y, yhat):
    w = [count_color_variety(s) for s in seqs]
    c = [cw if t == p else 0 for cw, t, p in zip(w, y, yhat)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y, yhat):
    w = [count_shape_variety(s) for s in seqs]
    c = [sw if t == p else 0 for sw, t, p in zip(w, y, yhat)]
    return sum(c) / sum(w) if sum(w) else 0.0


def harmonic_csa(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# --------------------------------------------------------------------------- #
# 3. Seeds                                                                    #
# --------------------------------------------------------------------------- #
def set_all_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_all_seeds(0)

# --------------------------------------------------------------------------- #
# 4. Load dataset & preprocessing                                             #
# --------------------------------------------------------------------------- #
spr = load_spr_bench(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))


def glyph_vector(g):
    return [ord(g[0]) - 65, ord(g[1]) - 48 if len(g) >= 2 else 0] if g else [0, 0]


all_glyphs = set(tok for seq in spr["train"]["sequence"] for tok in seq.strip().split())
vecs = np.array([glyph_vector(g) for g in all_glyphs])
k_clusters = 8
kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
glyph_to_cluster = {g: c for g, c in zip(all_glyphs, kmeans.fit_predict(vecs))}


def seq_to_hist(seq: str):
    h = np.zeros(k_clusters, np.float32)
    toks = seq.strip().split()
    for t in toks:
        h[glyph_to_cluster.get(t, 0)] += 1.0
    if toks:
        h /= len(toks)
    return h


class SPRHistDataset(Dataset):
    def __init__(self, seqs, labels):
        self.x = np.stack([seq_to_hist(s) for s in seqs]).astype(np.float32)
        self.y = np.array(labels, np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": torch.from_numpy(self.x[i]), "y": torch.tensor(self.y[i])}


train_ds = SPRHistDataset(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRHistDataset(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRHistDataset(spr["test"]["sequence"], spr["test"]["label"])
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# --------------------------------------------------------------------------- #
# 5. Evaluation helper                                                        #
# --------------------------------------------------------------------------- #
def evaluate(model, loader, sequences):
    model.eval()
    tot_loss, n = 0.0, 0
    preds = []
    gts = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for b in loader:
            b = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
            logits = model(b["x"])
            loss = criterion(logits, b["y"])
            tot_loss += loss.item() * b["y"].size(0)
            n += b["y"].size(0)
            pr = logits.argmax(1)
            preds += pr.cpu().tolist()
            gts += b["y"].cpu().tolist()
    loss = tot_loss / n
    cwa = color_weighted_accuracy(sequences, gts, preds)
    swa = shape_weighted_accuracy(sequences, gts, preds)
    hcs = harmonic_csa(cwa, swa)
    return {
        "loss": loss,
        "CWA": cwa,
        "SWA": swa,
        "HCSA": hcs,
        "preds": preds,
        "gts": gts,
    }


# --------------------------------------------------------------------------- #
# 6. Hyper-parameter sweep over weight_decay                                  #
# --------------------------------------------------------------------------- #
weight_decays = [0.0, 1e-5, 3e-4, 1e-3, 3e-3]
epochs = 10
experiment_data = {"weight_decay": {}}

for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    set_all_seeds(0)
    model = nn.Sequential(
        nn.Linear(k_clusters, 128), nn.ReLU(), nn.Linear(128, num_classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    run_data = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {},
        "ground_truth": {},
    }
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, n = 0.0, 0
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
            n += batch["y"].size(0)
        train_loss = tot_loss / n
        run_data["losses"]["train"].append((epoch, train_loss))

        val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])
        run_data["losses"]["val"].append((epoch, val_stats["loss"]))
        run_data["metrics"]["val"].append(
            (epoch, val_stats["CWA"], val_stats["SWA"], val_stats["HCSA"])
        )
        print(
            f"Ep{epoch}: train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} "
            f"CWA={val_stats['CWA']:.3f} SWA={val_stats['SWA']:.3f} HCSA={val_stats['HCSA']:.3f}"
        )

    # final dev/test
    dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
    test_final = evaluate(model, test_loader, spr["test"]["sequence"])
    run_data["predictions"]["dev"] = dev_final["preds"]
    run_data["ground_truth"]["dev"] = dev_final["gts"]
    run_data["predictions"]["test"] = test_final["preds"]
    run_data["ground_truth"]["test"] = test_final["gts"]

    print(
        f"Dev -> CWA:{dev_final['CWA']:.3f} SWA:{dev_final['SWA']:.3f} HCSA:{dev_final['HCSA']:.3f}"
    )
    print(
        f"Test-> CWA:{test_final['CWA']:.3f} SWA:{test_final['SWA']:.3f} HCSA:{test_final['HCSA']:.3f}"
    )

    experiment_data["weight_decay"][str(wd)] = run_data

# --------------------------------------------------------------------------- #
# 7. Persist                                                                  #
# --------------------------------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", working_dir + "/experiment_data.npy")
