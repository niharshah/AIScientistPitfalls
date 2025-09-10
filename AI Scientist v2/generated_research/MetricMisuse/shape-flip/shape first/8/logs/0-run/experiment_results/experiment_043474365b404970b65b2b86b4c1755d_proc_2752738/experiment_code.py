import os, pathlib, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ---------------------- paths / device ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


# ---------------------- helpers -----------------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def rule_signature(seq):
    return (count_shape_variety(seq), count_color_variety(seq))


def random_token():
    return random.choice(string.ascii_uppercase[:10]) + random.choice(string.digits[:5])


def generate_synthetic_split(n, seed=0):
    random.seed(seed)
    seqs, labs = [], []
    for _ in range(n):
        s = " ".join(random_token() for _ in range(random.randint(3, 10)))
        seqs.append(s)
        labs.append(int(count_shape_variety(s) == count_color_variety(s)))
    return {"id": list(range(n)), "sequence": seqs, "label": labs}


def load_spr_bench(root: pathlib.Path):
    if root.exists():

        def _ld(f):
            return load_dataset("csv", data_files=str(root / f), split="train")

        return DatasetDict(
            train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv")
        )
    print("Dataset not found, generating synthetic.")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synthetic_split(2000, 1)),
        dev=HFDataset.from_dict(generate_synthetic_split(500, 2)),
        test=HFDataset.from_dict(generate_synthetic_split(1000, 3)),
    )


DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

# ---------------------- encoding ----------------------------
shape_to_idx = {c: i for i, c in enumerate(string.ascii_uppercase[:26])}
colour_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feature_dim = 26 + 10 + 3


def encode_sequence(seq):
    v = np.zeros(feature_dim, dtype=np.float32)
    toks = seq.split()
    for tok in toks:
        if len(tok) < 2:
            continue
        v[shape_to_idx[tok[0]]] += 1
        v[26 + colour_to_idx[tok[1]]] += 1
    v[-3] = len(toks)
    v[-2] = count_shape_variety(seq)
    v[-1] = count_color_variety(seq)
    return v


def encode_dataset(hfds):
    X = np.stack([encode_sequence(s) for s in hfds["sequence"]])
    y = np.array(hfds["label"], dtype=np.int64)
    sigs = [rule_signature(s) for s in hfds["sequence"]]
    return X, y, sigs


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


# ---------------------- model -------------------------------
class MLP(nn.Module):
    def __init__(self, indim, hidden=64, classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, hidden), nn.ReLU(), nn.Linear(hidden, classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------- experiment container ----------------
experiment_data = {"batch_size": {"SPR_BENCH": {}}}

# static dev/test loaders (large batch)
dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev), batch_size=256)
test_loader = DataLoader(SPRTorchDS(X_test, y_test), batch_size=256)

train_signatures = set(sig_train)
unseen_dev_sigs = {s for s in sig_dev if s not in train_signatures}
unseen_test_sigs = {s for s in sig_test if s not in train_signatures}


def eval_loader(loader, sigs_all, unseen_set, model):
    model.eval()
    tot = cor = 0
    utot = ucor = 0
    preds = []
    with torch.no_grad():
        idx = 0
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            pred = out.argmax(1)
            preds.extend(pred.cpu().numpy())
            tot += y.size(0)
            cor += (pred == y).sum().item()
            for p, yt in zip(pred.cpu().numpy(), y.cpu().numpy()):
                sig = sigs_all[idx]
                if sig in unseen_set:
                    utot += 1
                    ucor += int(p == yt)
                idx += 1
    return cor / tot, (ucor / utot if utot else 0.0), preds


# ---------------------- hyper-parameter sweep ---------------
batch_sizes = [16, 32, 64, 128, 256]
EPOCHS = 5

for bs in batch_sizes:
    print("\n=== Training with batch_size =", bs, "===")
    train_loader = DataLoader(SPRTorchDS(X_train, y_train), batch_size=bs, shuffle=True)
    model = MLP(feature_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    run_data = {
        "metrics": {"train_acc": [], "val_acc": [], "val_ura": []},
        "losses": {"train": []},
        "test": {"acc": None, "ura": None},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "timestamps": [],
    }

    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss = cor = tot = 0
        for batch in train_loader:
            optim.zero_grad()
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            optim.step()
            run_loss += loss.item() * y.size(0)
            cor += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
        tr_loss = run_loss / tot
        tr_acc = cor / tot
        val_acc, val_ura, _ = eval_loader(dev_loader, sig_dev, unseen_dev_sigs, model)
        print(
            f"Epoch {ep}: loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_acc={val_acc:.3f} URA={val_ura:.3f}"
        )
        run_data["losses"]["train"].append(tr_loss)
        run_data["metrics"]["train_acc"].append(tr_acc)
        run_data["metrics"]["val_acc"].append(val_acc)
        run_data["metrics"]["val_ura"].append(val_ura)
        run_data["timestamps"].append(time.time())

    test_acc, test_ura, test_preds = eval_loader(
        test_loader, sig_test, unseen_test_sigs, model
    )
    run_data["test"]["acc"] = test_acc
    run_data["test"]["ura"] = test_ura
    run_data["predictions"] = test_preds
    print(f"Test: acc={test_acc:.3f}  URA={test_ura:.3f}")

    experiment_data["batch_size"]["SPR_BENCH"][bs] = run_data

# ---------------------- save all ----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
