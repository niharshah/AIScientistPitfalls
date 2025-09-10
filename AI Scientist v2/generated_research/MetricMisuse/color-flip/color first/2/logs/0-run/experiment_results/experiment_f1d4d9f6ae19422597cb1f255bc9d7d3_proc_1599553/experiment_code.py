import os, pathlib, random, math, time, json, itertools, collections
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# WORKING DIR
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# EXPERIMENT DATA STRUCTURE
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# --------------------------------------------------------------------------- #
# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------------- #
# DATA LOADING
def safe_load_spr(root: pathlib.Path):
    """Load benchmark if it exists, otherwise create tiny synthetic data."""
    if root.exists():
        from datasets import load_dataset, DatasetDict

        def _load(split_csv):
            return load_dataset(
                "csv",
                data_files=str(root / split_csv),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    else:
        # fabricate tiny synthetic data ------------------------------------ #
        def rand_token():
            shapes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            colours = "0123456789"
            return random.choice(shapes) + random.choice(colours)

        def make_row(i):
            seq = " ".join(rand_token() for _ in range(random.randint(4, 10)))
            label = random.randint(0, 4)  # 5-way classification
            return {"id": str(i), "sequence": seq, "label": label}

        train = [make_row(i) for i in range(800)]
        dev = [make_row(i + 10000) for i in range(200)]
        test = [make_row(i + 20000) for i in range(200)]
        import datasets

        d = {}
        d["train"] = datasets.Dataset.from_list(train)
        d["dev"] = datasets.Dataset.from_list(dev)
        d["test"] = datasets.Dataset.from_list(test)
    return d


DATA_PATH = pathlib.Path("./SPR_BENCH")  # change if needed
spr = safe_load_spr(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# --------------------------------------------------------------------------- #
# METRICS
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    good = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    good = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) > 0 else 0.0


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    good = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) > 0 else 0.0


# --------------------------------------------------------------------------- #
# VOCAB DISCOVERY
shapes = set()
colours = set()
for row in spr["train"]:
    for tok in row["sequence"].split():
        if tok:
            shapes.add(tok[0])
            if len(tok) > 1:
                colours.add(tok[1])
shapes = sorted(list(shapes))
colours = sorted(list(colours))
shape2i = {s: i for i, s in enumerate(shapes)}
colour2i = {c: i for i, c in enumerate(colours)}
F_DIM = len(shapes) + len(colours) + 2  # +length +unique shapes
NUM_CLASSES = len(set(spr["train"]["label"]))
print(f"Feature dim: {F_DIM}, Classes: {NUM_CLASSES}")


def featurise(seq: str) -> np.ndarray:
    vec = np.zeros(F_DIM, dtype=np.float32)
    toks = seq.strip().split()
    for tok in toks:
        if tok:
            s_idx = shape2i.get(tok[0], None)
            c_idx = colour2i.get(tok[1], None) if len(tok) > 1 else None
            if s_idx is not None:
                vec[s_idx] += 1
            if c_idx is not None:
                vec[len(shapes) + c_idx] += 1
    vec[-2] = len(toks)
    vec[-1] = count_shape_variety(seq)
    return vec


# --------------------------------------------------------------------------- #
# DATASET WRAPPER
class SPRFeats(Dataset):
    def __init__(self, hf_dataset):
        self.seqs = hf_dataset["sequence"]
        self.labels = hf_dataset["label"]
        self.feats = np.stack([featurise(s) for s in self.seqs])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.feats[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq": self.seqs[idx],
        }


train_ds, dev_ds, test_ds = (
    SPRFeats(spr["train"]),
    SPRFeats(spr["dev"]),
    SPRFeats(spr["test"]),
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)


# --------------------------------------------------------------------------- #
# MODEL
class MLP(nn.Module):
    def __init__(self, inp, hid, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, out))

    def forward(self, x):
        return self.net(x)


model = MLP(F_DIM, 64, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------------------------------------------------------------- #
# TRAIN LOOP
EPOCHS = 10


def run_epoch(loader, train_flag=True):
    if train_flag:
        model.train()
    else:
        model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    seqs = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        with torch.set_grad_enabled(train_flag):
            logits = model(x)
            loss = criterion(logits, y)
            if train_flag:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy())
        seqs.extend(batch["seq"])
    avg_loss = total_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    gcwa = glyph_complexity_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, (cwa, swa, gcwa), y_true, y_pred


train_losses, val_losses = [], []
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_metrics, _, _ = run_epoch(train_loader, True)
    val_loss, val_metrics, _, _ = run_epoch(dev_loader, False)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_metrics)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_metrics)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: "
        f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} | "
        f"VAL CWA={val_metrics[0]:.3f} SWA={val_metrics[1]:.3f} GCWA={val_metrics[2]:.3f}"
    )

# --------------------------------------------------------------------------- #
# FINAL TEST EVAL
test_loss, test_metrics, y_true, y_pred = run_epoch(test_loader, False)
experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
print(
    f"\nTEST => loss={test_loss:.4f} | CWA={test_metrics[0]:.3f} "
    f"SWA={test_metrics[1]:.3f} GCWA={test_metrics[2]:.3f}"
)

# --------------------------------------------------------------------------- #
# SAVE METRICS
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# --------------------------------------------------------------------------- #
# PLOT
plt.figure(figsize=(6, 4))
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["losses"]["train"],
    label="train",
)
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["losses"]["val"],
    label="val",
)
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "spr_loss_curve.png"))
plt.close()
