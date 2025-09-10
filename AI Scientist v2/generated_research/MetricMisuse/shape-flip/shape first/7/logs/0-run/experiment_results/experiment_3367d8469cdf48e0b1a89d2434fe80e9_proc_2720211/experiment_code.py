import os, math, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------
# Mandatory working dir & device handling
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------
# Experiment-wide bookkeeping structure
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -----------------------------------------------------------
# Helper functions for metrics
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) if sum(w) > 0 else 1e-6)


# -----------------------------------------------------------
# Data loading (falls back to synthetic if missing)
def load_spr_bench(root: str) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root, csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = _ld("train.csv"), _ld("dev.csv"), _ld("test.csv")
    return d


def make_synthetic_dataset(path: str, n_train=2000, n_dev=500, n_test=500):
    shapes, colors = list("STCH"), list("RGBY")

    def rand_seq():
        L = random.randint(3, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def rule(seq):  # label = 1 if shape_variety > color_variety else 0
        return int(count_shape_variety(seq) > count_color_variety(seq))

    def mk(n, fname):
        rows = ["id,sequence,label"]
        for i in range(n):
            s = rand_seq()
            rows.append(f"{i},{s},{rule(s)}")
        with open(os.path.join(path, fname), "w") as f:
            f.write("\n".join(rows))

    os.makedirs(path, exist_ok=True)
    mk(n_train, "train.csv")
    mk(n_dev, "dev.csv")
    mk(n_test, "test.csv")


root = os.getenv("SPR_PATH", "SPR_BENCH")
if not (
    os.path.exists(root)
    and all(
        os.path.exists(os.path.join(root, f"{sp}.csv"))
        for sp in ["train.csv", "dev.csv", "test.csv"]
    )
):
    print("SPR_BENCH not found – generating synthetic data …")
    make_synthetic_dataset(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})

# -----------------------------------------------------------
# Symbolic feature extractor
all_shapes = list("STCH")
all_colors = list("RGBY")


def featurise(seq: str):
    toks = seq.split()
    shape_counts = [0] * len(all_shapes)
    color_counts = [0] * len(all_colors)
    for tok in toks:
        if len(tok) == 0:
            continue
        s, c = tok[0], tok[1] if len(tok) > 1 else "?"
        if s in all_shapes:
            shape_counts[all_shapes.index(s)] += 1
        if c in all_colors:
            color_counts[all_colors.index(c)] += 1
    features = shape_counts + color_counts
    features += [len(toks), count_shape_variety(seq), count_color_variety(seq)]
    return np.array(features, dtype=np.float32)


input_dim = len(all_shapes) + len(all_colors) + 3  # 11


# -----------------------------------------------------------
# Torch Dataset / DataLoader
class SPRFeatDS(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]
        self.y = hf_ds["label"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(featurise(self.seqs[idx])),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


batch_size = 128
train_dl = DataLoader(SPRFeatDS(spr["train"]), batch_size=batch_size, shuffle=True)
val_dl = DataLoader(SPRFeatDS(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRFeatDS(spr["test"]), batch_size=batch_size)


# -----------------------------------------------------------
# Simple MLP model
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=32, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -----------------------------------------------------------
# Training / evaluation utilities
def run_epoch(dl, train=False):
    model.train() if train else model.eval()
    tot_loss, y_true, y_pred, seqs = 0.0, [], [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["y"].shape[0]
        preds = logits.argmax(dim=-1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch["y"].cpu().tolist())
        seqs.extend(batch["raw"])
    avg_loss = tot_loss / len(dl.dataset)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, swa, y_true, y_pred


# -----------------------------------------------------------
# Main training loop
epochs = 20
for epoch in range(1, epochs + 1):
    tr_loss, tr_swa, _, _ = run_epoch(train_dl, train=True)
    val_loss, val_swa, _, _ = run_epoch(val_dl)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append((epoch, tr_swa))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, val_swa))
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, SWA = {val_swa:.4f}")

# -----------------------------------------------------------
# Final test evaluation
test_loss, test_swa, test_gt, test_pred = run_epoch(test_dl)
print(f"\nTEST SWA = {test_swa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = test_gt

# -----------------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
