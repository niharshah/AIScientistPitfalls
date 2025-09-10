import os, pathlib, random, string, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ---------------------------------------------------------------------
# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# metrics helpers (provided by benchmark)
def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# ---------------------------------------------------------------------
# Load SPR_BENCH or create synthetic fallback
def _random_token():
    shape = random.choice(string.ascii_uppercase[:10])  # 10 shapes
    colour = random.choice(string.digits[:5])  # 5 colours
    return shape + colour


def _generate_split(n, seed):
    random.seed(seed)
    seqs, labels = [], []
    for i in range(n):
        length = random.randint(3, 10)
        seq = " ".join(_random_token() for _ in range(length))
        # simple parity rule for synthetic label
        labels.append((count_shape_variety(seq) + count_color_variety(seq)) % 2)
        seqs.append(seq)
    return {"id": list(range(n)), "sequence": seqs, "label": labels}


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    if root.exists():
        print(f"Reading SPR_BENCH from {root}")

        def _ld(fname):
            return load_dataset("csv", data_files=str(root / fname), split="train")

        return DatasetDict(
            train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv")
        )
    print("SPR_BENCH folder not found – using synthetic data")
    return DatasetDict(
        train=HFDataset.from_dict(_generate_split(4000, 1)),
        dev=HFDataset.from_dict(_generate_split(1000, 2)),
        test=HFDataset.from_dict(_generate_split(2000, 3)),
    )


DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

# ---------------------------------------------------------------------
# Featurisation (neural + symbolic histograms)
shape_to_idx = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
color_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
BASE_DIM = 26 + 10  # histogram of shapes + colours
EXTRA_DIM = 6  # engineered symbolic stats
FEAT_DIM = BASE_DIM + EXTRA_DIM


def encode_sequence(seq: str) -> np.ndarray:
    v = np.zeros(FEAT_DIM, dtype=np.float32)
    toks = seq.split()
    for tok in toks:
        v[shape_to_idx[tok[0]]] += 1
        if len(tok) > 1:
            v[26 + color_to_idx[tok[1]]] += 1
    sv = count_shape_variety(seq)
    cv = count_color_variety(seq)
    v[-6] = len(toks)
    v[-5] = sv
    v[-4] = cv
    v[-3] = sv - cv
    v[-2] = sv % 2
    v[-1] = cv % 2
    # simple normalisation
    v[:BASE_DIM] /= len(toks) + 1e-6
    return v


def encode_dataset(hfds):
    feats = np.stack([encode_sequence(s) for s in hfds["sequence"]])
    labels = np.array(hfds["label"], dtype=np.int64)
    sigs = [(count_shape_variety(s), count_color_variety(s)) for s in hfds["sequence"]]
    return feats, labels, sigs


X_tr, y_tr, sig_tr = encode_dataset(dsets["train"])
X_dev, y_dev, sig_dev = encode_dataset(dsets["dev"])
X_te, y_te, sig_te = encode_dataset(dsets["test"])


# ---------------------------------------------------------------------
# PyTorch Dataset / DataLoader
class SPRSet(Dataset):
    def __init__(self, X, y, seqs, sigs):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.seqs = seqs
        self.sigs = sigs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "seq": self.seqs[idx],
            "sig": self.sigs[idx],
        }


train_set = SPRSet(X_tr, y_tr, dsets["train"]["sequence"], sig_tr)
dev_set = SPRSet(X_dev, y_dev, dsets["dev"]["sequence"], sig_dev)
test_set = SPRSet(X_te, y_te, dsets["test"]["sequence"], sig_te)

train_dl = DataLoader(train_set, batch_size=64, shuffle=True)
train_eval_dl = DataLoader(train_set, batch_size=256, shuffle=False)
dev_dl = DataLoader(dev_set, batch_size=256, shuffle=False)
test_dl = DataLoader(test_set, batch_size=256, shuffle=False)

# sets to measure zero-shot behaviour
train_sigs = set(sig_tr)
unseen_dev = {s for s in sig_dev if s not in train_sigs}
unseen_test = {s for s in sig_te if s not in train_sigs}


# ---------------------------------------------------------------------
# Simple hybrid MLP
class HybridMLP(nn.Module):
    def __init__(self, in_dim, hid=128, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, n_cls),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# Evaluation helper
def evaluate(model, loader, unseen_set, criterion=None):
    model.eval()
    tot_loss = tot_items = 0
    all_preds, all_y, all_seq = [], [], []
    all_sigs = []
    with torch.no_grad():
        for batch in loader:
            # move tensors
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            if criterion is not None:
                loss = criterion(logits, batch["y"])
                tot_loss += loss.item() * batch["y"].size(0)
                tot_items += batch["y"].size(0)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_y.extend(batch["y"].cpu().numpy())
            all_seq.extend(batch["seq"])
            all_sigs.extend(batch["sig"])
    acc = (np.array(all_preds) == np.array(all_y)).mean()
    swa = shape_weighted_accuracy(all_seq, all_y, all_preds)
    # zero-shot accuracy on unseen rule signatures
    zs_correct = zs_total = 0
    for pr, gt, sig in zip(all_preds, all_y, all_sigs):
        if sig in unseen_set:
            zs_total += 1
            if pr == gt:
                zs_correct += 1
    zs_acc = zs_correct / zs_total if zs_total else 0.0
    mean_loss = tot_loss / tot_items if tot_items else None
    return mean_loss, acc, swa, zs_acc, all_preds


# ---------------------------------------------------------------------
# experiment bookkeeping
experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train_acc": [],
            "val_acc": [],
            "train_swa": [],
            "val_swa": [],
            "val_zs": [],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_te.tolist(),
        "timestamps": [],
    }
}

# ---------------------------------------------------------------------
# training loop
EPOCHS = 15
model = HybridMLP(FEAT_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_loss = tot = 0
    for batch in train_dl:
        # move tensors to device
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
        tot += batch["y"].size(0)
    train_loss = tot_loss / tot

    # evaluate on training set (no shuffle) and dev set
    _, train_acc, train_swa, _, _ = evaluate(model, train_eval_dl, unseen_set=set())
    val_loss, val_acc, val_swa, val_zs, _ = evaluate(
        model, dev_dl, unseen_set=unseen_dev, criterion=criterion
    )

    # logging
    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"val_acc={val_acc:.3f}  val_SWA={val_swa:.3f}  val_ZS={val_zs:.3f}"
    )

    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train_acc"].append(train_acc)
    ed["metrics"]["val_acc"].append(val_acc)
    ed["metrics"]["train_swa"].append(train_swa)
    ed["metrics"]["val_swa"].append(val_swa)
    ed["metrics"]["val_zs"].append(val_zs)
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["timestamps"].append(time.time())

# ---------------------------------------------------------------------
# final evaluation on test set
test_loss, test_acc, test_swa, test_zs, test_preds = evaluate(
    model, test_dl, unseen_set=unseen_test, criterion=criterion
)
print(
    f"\nFinal Test — loss={test_loss:.4f}  Acc={test_acc:.3f}  "
    f"SWA={test_swa:.3f}  Zero-Shot_Acc={test_zs:.3f}"
)

experiment_data["SPR_BENCH"]["metrics"]["test_swa"] = [test_swa]
experiment_data["SPR_BENCH"]["metrics"]["test_acc"] = [test_acc]
experiment_data["SPR_BENCH"]["metrics"]["test_zs"] = [test_zs]
experiment_data["SPR_BENCH"]["losses"]["test"] = [test_loss]
experiment_data["SPR_BENCH"]["predictions"] = test_preds

# save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy in", working_dir)
