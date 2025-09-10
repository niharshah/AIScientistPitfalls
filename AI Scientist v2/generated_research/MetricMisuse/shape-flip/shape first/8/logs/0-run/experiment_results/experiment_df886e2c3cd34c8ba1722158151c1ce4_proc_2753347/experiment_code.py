import os, pathlib, random, string, time, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ------------ experiment data container ------------
experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {
            "lr_values": [],
            "metrics": {
                "train_acc": [],  # shape : [n_lr , n_epochs]
                "val_acc": [],
                "val_ura": [],
                "test_acc": [],
                "test_ura": [],
            },
            "losses": {"train": [], "val": []},  # same shape convention
            "predictions": [],  # list of list
            "ground_truth": [],  # set once later
            "timestamps": [],
        }
    }
}

# ------------ device ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------ helper symbolic functions ------------
def count_shape_variety(sequence):
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color_variety(sequence):
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def rule_signature(sequence):
    return (count_shape_variety(sequence), count_color_variety(sequence))


# ------------ synthetic data fallback ------------
def random_token():
    return random.choice(string.ascii_uppercase[:10]) + random.choice(string.digits[:5])


def generate_synthetic_split(n, seed=0):
    random.seed(seed)
    seqs, labels = [], []
    for i in range(n):
        length = random.randint(3, 10)
        seq = " ".join(random_token() for _ in range(length))
        label = int(count_shape_variety(seq) == count_color_variety(seq))
        seqs.append(seq)
        labels.append(label)
    return {"id": list(range(n)), "sequence": seqs, "label": labels}


def load_spr_bench(root: pathlib.Path):
    if root.exists():
        print("Loading real SPR_BENCH")
        _ld = lambda f: load_dataset("csv", data_files=str(root / f), split="train")
        return DatasetDict(
            train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv")
        )
    print("SPR_BENCH not found – generating synthetic data")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synthetic_split(2000, 1)),
        dev=HFDataset.from_dict(generate_synthetic_split(500, 2)),
        test=HFDataset.from_dict(generate_synthetic_split(1000, 3)),
    )


DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

# ------------ feature encoding ------------
shape_to_idx = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
colour_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feature_dim = 26 + 10 + 3  # histograms + len/varieties


def encode_sequence(seq):
    vec = np.zeros(feature_dim, np.float32)
    toks = seq.split()
    for tok in toks:
        if len(tok) < 2:
            continue
        vec[shape_to_idx[tok[0]]] += 1
        vec[26 + colour_to_idx[tok[1]]] += 1
    vec[-3] = len(toks)
    vec[-2] = count_shape_variety(seq)
    vec[-1] = count_color_variety(seq)
    return vec


def encode_dataset(hf_ds):
    feats = np.stack([encode_sequence(s) for s in hf_ds["sequence"]])
    labels = np.asarray(hf_ds["label"], np.int64)
    sigs = [rule_signature(s) for s in hf_ds["sequence"]]
    return feats, labels, sigs


X_train, y_train, sig_train = encode_dataset(dsets["train"])
X_dev, y_dev, sig_dev = encode_dataset(dsets["dev"])
X_test, y_test, sig_test = encode_dataset(dsets["test"])
experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"] = y_test.tolist()


class SPRTorchDS(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X), torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


train_base_loader = DataLoader(
    SPRTorchDS(X_train, y_train), batch_size=64, shuffle=True
)
dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev), batch_size=256)
test_loader = DataLoader(SPRTorchDS(X_test, y_test), batch_size=256)


# ------------ model ------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# ------------ evaluation helper ------------
def eval_loader(model, loader, sigs_all, unseen_sigs):
    model.eval()
    correct = tot = correct_u = tot_u = 0
    preds_all = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            logits = model(x)
            preds = logits.argmax(1)
            preds_all.extend(preds.cpu().numpy())
            tot += y.size(0)
            correct += (preds == y).sum().item()
            for p_, y_ in zip(preds.cpu().numpy(), y.cpu().numpy()):
                if sigs_all[idx] in unseen_sigs:
                    tot_u += 1
                    if p_ == y_:
                        correct_u += 1
                idx += 1
    acc = correct / tot
    ura = correct_u / tot_u if tot_u else 0.0
    return acc, ura, preds_all


train_sigs_set = set(sig_train)
unseen_dev_sigs = {s for s in sig_dev if s not in train_sigs_set}
unseen_test_sigs = {s for s in sig_test if s not in train_sigs_set}

# ------------ learning-rate sweep ------------
lr_values = [3e-4, 5e-4, 1e-3, 2e-3]
EPOCHS = 8


def train_loader_from_base():
    # fresh iterator every epoch due to shuffle; reuse Dataset object
    return DataLoader(SPRTorchDS(X_train, y_train), batch_size=64, shuffle=True)


for lr in lr_values:
    experiment_data["learning_rate"]["SPR_BENCH"]["lr_values"].append(lr)
    per_lr_train_acc, per_lr_val_acc, per_lr_val_ura = [], [], []
    per_lr_train_loss, per_lr_val_loss = [], []

    model = MLP(feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = run_corr = run_tot = 0
        for batch in train_loader_from_base():
            optimizer.zero_grad()
            x, y = batch["x"].to(device), batch["y"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            run_corr += (logits.argmax(1) == y).sum().item()
            run_tot += y.size(0)
        train_loss = run_loss / run_tot
        train_acc = run_corr / run_tot
        val_acc, val_ura, _ = eval_loader(model, dev_loader, sig_dev, unseen_dev_sigs)

        per_lr_train_loss.append(train_loss)
        per_lr_val_loss.append(np.nan)  # placeholder, val loss not computed
        per_lr_train_acc.append(train_acc)
        per_lr_val_acc.append(val_acc)
        per_lr_val_ura.append(val_ura)

        print(
            f"[lr={lr:.4g}] epoch {epoch}/{EPOCHS}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} URA={val_ura:.3f}"
        )

    # final test evaluation
    test_acc, test_ura, test_preds = eval_loader(
        model, test_loader, sig_test, unseen_test_sigs
    )
    print(f"--> lr={lr:.4g}  TEST acc={test_acc:.3f}  URA={test_ura:.3f}\n")

    # store results
    ed = experiment_data["learning_rate"]["SPR_BENCH"]
    ed["metrics"]["train_acc"].append(per_lr_train_acc)
    ed["metrics"]["val_acc"].append(per_lr_val_acc)
    ed["metrics"]["val_ura"].append(per_lr_val_ura)
    ed["losses"]["train"].append(per_lr_train_loss)
    ed["losses"]["val"].append(per_lr_val_loss)
    ed["metrics"]["test_acc"].append(test_acc)
    ed["metrics"]["test_ura"].append(test_ura)
    ed["predictions"].append(test_preds)
    ed["timestamps"].append(time.time())

# ------------ save experiment data ------------
os.makedirs("working", exist_ok=True)
np.save(os.path.join("working", "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
