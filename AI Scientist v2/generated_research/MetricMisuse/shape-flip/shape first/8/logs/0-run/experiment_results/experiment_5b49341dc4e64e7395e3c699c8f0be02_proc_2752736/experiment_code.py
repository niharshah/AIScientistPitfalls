import os, pathlib, random, string, time, copy
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ---------- dir / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------- helper symbolic functions ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def rule_signature(sequence: str):
    return (count_shape_variety(sequence), count_color_variety(sequence))


# ---------- synthetic fallback ----------
def random_token():
    shape = random.choice(string.ascii_uppercase[:10])  # 10 shapes
    colour = random.choice(string.digits[:5])  # 5 colours
    return shape + colour


def generate_synthetic_split(n_rows: int, seed=0):
    random.seed(seed)
    seqs, labels = [], []
    for _ in range(n_rows):
        length = random.randint(3, 10)
        seq = " ".join(random_token() for _ in range(length))
        lbl = int(count_shape_variety(seq) == count_color_variety(seq))
        seqs.append(seq)
        labels.append(lbl)
    return {"id": list(range(n_rows)), "sequence": seqs, "label": labels}


def load_spr_bench(root_path: pathlib.Path) -> DatasetDict:
    if root_path.exists():
        print(f"Loading SPR_BENCH from {root_path}")

        def _load(fname):
            return load_dataset("csv", data_files=str(root_path / fname), split="train")

        return DatasetDict(
            train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
        )
    print("SPR_BENCH not found – generating synthetic data")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synthetic_split(2000, 1)),
        dev=HFDataset.from_dict(generate_synthetic_split(500, 2)),
        test=HFDataset.from_dict(generate_synthetic_split(1000, 3)),
    )


# ---------- data ----------
DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

shape_to_idx = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
colour_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feature_dim = 26 + 10 + 3  # shapes hist + colours hist + {len,varieties}


def encode_sequence(seq: str) -> np.ndarray:
    vec = np.zeros(feature_dim, np.float32)
    toks = seq.split()
    for tok in toks:
        if len(tok) < 2:
            continue
        vec[shape_to_idx[tok[0]]] += 1
        vec[26 + colour_to_idx[tok[1]]] += 1
    vec[-3], vec[-2], vec[-1] = (
        len(toks),
        count_shape_variety(seq),
        count_color_variety(seq),
    )
    return vec


def encode_dataset(hfds):
    feats = np.stack([encode_sequence(s) for s in hfds["sequence"]])
    labels = np.array(hfds["label"], np.int64)
    sigs = [rule_signature(s) for s in hfds["sequence"]]
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

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


train_loader = lambda: DataLoader(
    SPRTorchDS(X_train, y_train), batch_size=64, shuffle=True
)
dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev), batch_size=256)
test_loader = DataLoader(SPRTorchDS(X_test, y_test), batch_size=256)


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, hid=64, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, n_cls)
        )

    def forward(self, x):
        return self.net(x)


criterion = nn.CrossEntropyLoss()


# ---------- evaluation ----------
def eval_loader(model, loader, sigs_all, unseen_sigs):
    model.eval()
    correct = total = correct_u = total_u = 0
    preds_all = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            preds = logits.argmax(1)
            preds_all.extend(preds.cpu().numpy())
            total += y.size(0)
            correct += (preds == y).sum().item()
            for p, y_t in zip(preds.cpu().numpy(), y.cpu().numpy()):
                sig = sigs_all[idx]
                idx += 1
                if sig not in unseen_sigs:
                    continue
                total_u += 1
                correct_u += int(p == y_t)
    acc = correct / total
    ura = correct_u / total_u if total_u else 0.0
    return acc, ura, preds_all


train_sig_set = set(sig_train)
unseen_dev_sigs = {s for s in sig_dev if s not in train_sig_set}
unseen_test_sigs = {s for s in sig_test if s not in train_sig_set}

# ---------- hyperparameter search over epochs ----------
epoch_budgets = [5, 10, 15, 20, 25, 30]
patience = 3

experiment_data = {
    "EPOCH_TUNING": {
        "SPR_BENCH": {
            "hyperparams": [],
            "metrics": {
                "train_acc": [],
                "val_acc": [],
                "test_acc": [],
                "val_ura": [],
                "test_ura": [],
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_test.tolist(),
            "timestamps": [],
        }
    }
}

for max_epochs in epoch_budgets:
    print(f"\n=== Training with max_epochs={max_epochs} ===")
    model = MLP(feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_state, best_val_acc = None, 0.0
    patience_ctr = 0
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        run_loss = run_correct = run_total = 0
        for batch in train_loader():
            optimizer.zero_grad()
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            run_total += y.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
        train_loss = run_loss / run_total
        train_acc = run_correct / run_total

        val_acc, val_ura, _ = eval_loader(model, dev_loader, sig_dev, unseen_dev_sigs)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(None)  # val loss not computed
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        print(
            f"Epoch {epoch}/{max_epochs} - train_loss {train_loss:.4f} "
            f"train_acc {train_acc:.3f} val_acc {val_acc:.3f} URA {val_ura:.3f}"
        )

        # early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

    # reload best model
    model.load_state_dict(best_state)
    train_acc_final, _, _ = eval_loader(
        model,
        DataLoader(SPRTorchDS(X_train, y_train), batch_size=256),
        sig_train,
        set(),
    )
    val_acc_final, val_ura_final, _ = eval_loader(
        model, dev_loader, sig_dev, unseen_dev_sigs
    )
    test_acc, test_ura, test_preds = eval_loader(
        model, test_loader, sig_test, unseen_test_sigs
    )

    # save results
    ed = experiment_data["EPOCH_TUNING"]["SPR_BENCH"]
    ed["hyperparams"].append({"epochs": max_epochs})
    ed["metrics"]["train_acc"].append(train_acc_final)
    ed["metrics"]["val_acc"].append(val_acc_final)
    ed["metrics"]["test_acc"].append(test_acc)
    ed["metrics"]["val_ura"].append(val_ura_final)
    ed["metrics"]["test_ura"].append(test_ura)
    ed["losses"]["train"].append(train_loss_hist)
    ed["losses"]["val"].append(val_loss_hist)
    ed["predictions"].append(test_preds)
    ed["timestamps"].append(time.time())

    print(f"--> Finished: test_acc={test_acc:.3f}, test_ura={test_ura:.3f}")

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved results to", os.path.join(working_dir, "experiment_data.npy"))
