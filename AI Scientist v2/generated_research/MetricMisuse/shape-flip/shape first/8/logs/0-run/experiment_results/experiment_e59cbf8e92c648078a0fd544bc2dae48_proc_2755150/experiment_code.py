import os, pathlib, random, string, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ----------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- reproducibility -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------- helper functions -----------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(s) for s in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def rule_signature(sequence: str):
    return (count_shape_variety(sequence), count_color_variety(sequence))


# ----------------- synthetic fallback -----------------
def random_token():
    return random.choice(string.ascii_uppercase[:10]) + random.choice(string.digits[:5])


def generate_synthetic_split(n):
    seqs, labels = [], []
    for _ in range(n):
        length = random.randint(3, 10)
        seq = " ".join(random_token() for _ in range(length))
        lbl = int(count_shape_variety(seq) == count_color_variety(seq))
        seqs.append(seq)
        labels.append(lbl)
    return {"id": list(range(n)), "sequence": seqs, "label": labels}


def load_spr_bench(root_path: pathlib.Path) -> DatasetDict:
    if root_path.exists():

        def _load(csv_name):
            return load_dataset(
                "csv", data_files=str(root_path / csv_name), split="train"
            )

        return DatasetDict(
            train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
        )
    print("SPR_BENCH not found, using synthetic data.")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synthetic_split(2000)),
        dev=HFDataset.from_dict(generate_synthetic_split(500)),
        test=HFDataset.from_dict(generate_synthetic_split(1000)),
    )


DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

# -------------- feature engineering --------------
shape_to_idx = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
colour_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feature_dim = 26 + 10 + 3  # histograms + length + varieties


def encode_sequence(seq: str) -> np.ndarray:
    vec = np.zeros(feature_dim, dtype=np.float32)
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
    labels = np.array(hf_ds["label"], dtype=np.int64)
    sigs = [rule_signature(s) for s in hf_ds["sequence"]]
    return feats, labels, sigs


X_train, y_train, sig_train = encode_dataset(dsets["train"])
X_dev, y_dev, sig_dev = encode_dataset(dsets["dev"])
X_test, y_test, sig_test = encode_dataset(dsets["test"])


# -------------- datasets ----------------
class SPRTorchDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


train_loader = DataLoader(SPRTorchDS(X_train, y_train), batch_size=64, shuffle=True)
dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev), batch_size=256)
test_loader = DataLoader(SPRTorchDS(X_test, y_test), batch_size=256)


# -------------- neural model --------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(feature_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------- symbolic gating prep --------------
train_signatures = set(sig_train)


def gated_predict(batch_x, batch_seq):
    logits = model(batch_x)
    nn_preds = logits.argmax(dim=1).cpu().numpy()
    final_preds = []
    for seq, nn_pred in zip(batch_seq, nn_preds):
        sig = rule_signature(seq)
        if sig not in train_signatures:  # unseen => symbolic rule
            final_preds.append(int(sig[0] == sig[1]))
        else:
            final_preds.append(int(nn_pred))
    return np.array(final_preds)


# -------------- experiment tracking --------------
experiment_data = {
    "SWA_experiment": {
        "metrics": {"train_swa": [], "val_swa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "timestamps": [],
    }
}

# -------------- training loop --------------
EPOCHS = 15
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = total_items = 0
    for batch in train_loader:
        optimizer.zero_grad()
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_items += y.size(0)
    train_loss = total_loss / total_items

    # --- compute SWA on train (subset for speed) ---
    model.eval()
    with torch.no_grad():
        sample_idx = np.random.choice(len(X_train), size=512, replace=False)
        sample_x = torch.tensor(X_train[sample_idx]).to(device)
        sample_seq = [dsets["train"]["sequence"][i] for i in sample_idx]
        train_preds = gated_predict(sample_x, sample_seq)
        train_swa = shape_weighted_accuracy(
            sample_seq, y_train[sample_idx], train_preds
        )

    # --- validation ---
    val_seqs = dsets["dev"]["sequence"]
    all_val_preds = []
    with torch.no_grad():
        for batch in dev_loader:
            bx = batch["x"].to(device)
            seq_slice = val_seqs[: bx.size(0)]
            val_seqs = val_seqs[bx.size(0) :]
            all_val_preds.extend(gated_predict(bx, seq_slice))
    val_swa = shape_weighted_accuracy(dsets["dev"]["sequence"], y_dev, all_val_preds)
    val_loss = 0.0  # not meaningful when gating; kept placeholder

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | Train SWA={train_swa:.3f} | Val SWA={val_swa:.3f}"
    )

    ed = experiment_data["SWA_experiment"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_swa"].append(train_swa)
    ed["metrics"]["val_swa"].append(val_swa)
    ed["timestamps"].append(time.time())

# -------------- test evaluation --------------
test_seqs = dsets["test"]["sequence"]
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        bx = batch["x"].to(device)
        seq_slice = test_seqs[: bx.size(0)]
        test_seqs = test_seqs[bx.size(0) :]
        all_test_preds.extend(gated_predict(bx, seq_slice))
test_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y_test, all_test_preds)
print(f"\nFinal Test Shape-Weighted Accuracy (SWA): {test_swa:.3f}")
experiment_data["SWA_experiment"]["predictions"] = all_test_preds
experiment_data["SWA_experiment"]["test_swa"] = test_swa

# -------------- save --------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to working directory.")
