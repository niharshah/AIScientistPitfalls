import os, pathlib, random, string, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# --------- experiment bookkeeping ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"weight_decay_tuning": {}}  # will fill per weight-decay value

# --------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------- helper symbolic functions ----------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.strip().split() if tok})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def rule_signature(seq):
    return (count_shape_variety(seq), count_color_variety(seq))


# --------- fallback synthetic data ----------
def random_token():
    return random.choice(string.ascii_uppercase[:10]) + random.choice(string.digits[:5])


def generate_synth_split(n, seed=0):
    random.seed(seed)
    seqs, labels = [], []
    for _ in range(n):
        length = random.randint(3, 10)
        seq = " ".join(random_token() for _ in range(length))
        label = int(count_shape_variety(seq) == count_color_variety(seq))
        seqs.append(seq)
        labels.append(label)
    return {"id": list(range(n)), "sequence": seqs, "label": labels}


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    if root.exists():
        print("Loading real SPR_BENCH from", root)
        _load = lambda f: load_dataset("csv", data_files=str(root / f), split="train")
        return DatasetDict(
            train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
        )
    print("SPR_BENCH not found – generating synthetic data")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synth_split(2000, 1)),
        dev=HFDataset.from_dict(generate_synth_split(500, 2)),
        test=HFDataset.from_dict(generate_synth_split(1000, 3)),
    )


data_path = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(data_path)

# --------- feature encoding ----------
shape_to_idx = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
colour_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feature_dim = (
    26 + 10 + 3
)  # shapes hist + colours hist + [seq_len, #shapeVar, #colourVar]


def encode_sequence(seq: str) -> np.ndarray:
    v = np.zeros(feature_dim, np.float32)
    toks = seq.split()
    for tok in toks:
        if len(tok) < 2:
            continue
        v[shape_to_idx[tok[0]]] += 1
        v[26 + colour_to_idx[tok[1]]] += 1
    v[-3:] = [len(toks), count_shape_variety(seq), count_color_variety(seq)]
    return v


def encode_dataset(hf):
    feats = np.stack([encode_sequence(s) for s in hf["sequence"]])
    labels = np.array(hf["label"], np.int64)
    sigs = [rule_signature(s) for s in hf["sequence"]]
    return feats, labels, sigs


X_train, y_train, sig_train = encode_dataset(dsets["train"])
X_dev, y_dev, sig_dev = encode_dataset(dsets["dev"])
X_test, y_test, sig_test = encode_dataset(dsets["test"])


class SPRTorchDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i]}


train_loader = DataLoader(SPRTorchDS(X_train, y_train), batch_size=64, shuffle=True)
dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev), batch_size=256)
test_loader = DataLoader(SPRTorchDS(X_test, y_test), batch_size=256)


# --------- model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_cls)
        )

    def forward(self, x):
        return self.net(x)


# --------- evaluation ----------
def eval_loader(model, loader, sigs_all, unseen_sigs):
    model.eval()
    correct = total = correct_u = total_u = idx = 0
    preds_all = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            preds = model(x).argmax(1)
            preds_all.extend(preds.cpu().numpy())
            total += y.size(0)
            correct += (preds == y).sum().item()
            for p, y_true in zip(preds.cpu().numpy(), y.cpu().numpy()):
                sig = sigs_all[idx]
                if sig in unseen_sigs:
                    total_u += 1
                    correct_u += int(p == y_true)
                idx += 1
    acc = correct / total
    ura = correct_u / total_u if total_u else 0.0
    return acc, ura, preds_all


train_sigs = set(sig_train)
unseen_dev = {s for s in sig_dev if s not in train_sigs}
unseen_test = {s for s in sig_test if s not in train_sigs}

# --------- hyperparameter sweep ----------
weight_decays = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
EPOCHS = 5
criterion = nn.CrossEntropyLoss()

for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    model = MLP(feature_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    run_rec = {
        "metrics": {"train_acc": [], "val_acc": [], "val_ura": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "timestamps": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = running_correct = running_total = 0
        for batch in train_loader:
            opt.zero_grad()
            x, y = batch["x"].to(device), batch["y"].to(device)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.size(0)
            running_correct += (model(x).argmax(1) == y).sum().item()
            running_total += y.size(0)
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_acc, val_ura, _ = eval_loader(model, dev_loader, sig_dev, unseen_dev)
        print(
            f"epoch {epoch}: loss {train_loss:.4f}, train_acc {train_acc:.3f}, "
            f"val_acc {val_acc:.3f}, URA {val_ura:.3f}"
        )

        run_rec["losses"]["train"].append(train_loss)
        run_rec["metrics"]["train_acc"].append(train_acc)
        run_rec["metrics"]["val_acc"].append(val_acc)
        run_rec["metrics"]["val_ura"].append(val_ura)
        run_rec["timestamps"].append(time.time())

    # final test evaluation
    test_acc, test_ura, test_preds = eval_loader(
        model, test_loader, sig_test, unseen_test
    )
    print(f"Test: acc {test_acc:.3f}, URA {test_ura:.3f}")
    run_rec["metrics"]["test_acc"] = test_acc
    run_rec["metrics"]["test_ura"] = test_ura
    run_rec["predictions"] = test_preds

    # store under string key to keep numpy compatibility
    experiment_data["weight_decay_tuning"][str(wd)] = run_rec

# --------- save ---------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
