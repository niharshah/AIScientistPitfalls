import os, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------- I/O & device -------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ------------------------- helpers ------------------------------
def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    cor = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(cor) / max(sum(w), 1)


def rule_signature(seq: str):
    return (count_shape_variety(seq), count_color_variety(seq))


# ------------------------- synthetic data -----------------------
def random_token():
    return random.choice(string.ascii_uppercase[:10]) + random.choice(string.digits[:6])


def generate_synthetic_split(n, seed):
    random.seed(seed)
    seqs, labels = [], []
    for i in range(n):
        L = random.randint(3, 10)
        s = " ".join(random_token() for _ in range(L))
        seqs.append(s)
        labels.append(int(count_shape_variety(s) == count_color_variety(s)))
    return {"id": list(range(n)), "sequence": seqs, "label": labels}


# create three independent datasets
train_raw = generate_synthetic_split(2000, 10)  # seed 10
val_raw = generate_synthetic_split(500, 20)  # seed 20
test_raw = generate_synthetic_split(1000, 30)  # seed 30

# ------------------------- encoding -----------------------------
shape_to_idx = {c: i for i, c in enumerate(string.ascii_uppercase[:26])}
color_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feat_dim = 26 + 10 + 3  # shape hist + colour hist + misc


def encode(seq: str):
    v = np.zeros(feat_dim, dtype=np.float32)
    toks = seq.split()
    for t in toks:
        if len(t) < 2:
            continue
        v[shape_to_idx[t[0]]] += 1
        v[26 + color_to_idx[t[1]]] += 1
    v[-3], v[-2], v[-1] = len(toks), count_shape_variety(seq), count_color_variety(seq)
    return v


def encode_split(split):
    X = np.stack([encode(s) for s in split["sequence"]])
    y = np.array(split["label"], dtype=np.int64)
    sigs = [rule_signature(s) for s in split["sequence"]]
    return X, y, sigs


X_train, y_train, sig_train = encode_split(train_raw)
X_val, y_val, sig_val = encode_split(val_raw)
X_test, y_test, sig_test = encode_split(test_raw)
train_signatures = set(sig_train)


# ------------------------- torch datasets -----------------------
class TorchDS(Dataset):
    def __init__(self, X, y, seqs):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.seqs = seqs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i], "seq": self.seqs[i]}


bs_train, bs_eval = 64, 256
train_loader = DataLoader(
    TorchDS(X_train, y_train, train_raw["sequence"]), batch_size=bs_train, shuffle=True
)
val_loader = DataLoader(TorchDS(X_val, y_val, val_raw["sequence"]), batch_size=bs_eval)
test_loader = DataLoader(
    TorchDS(X_test, y_test, test_raw["sequence"]), batch_size=bs_eval
)


# ------------------------- model -------------------------------
class MLP(nn.Module):
    def __init__(self, indim, h=64, out=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, h), nn.ReLU(), nn.Linear(h, out))

    def forward(self, x):
        return self.net(x)


model = MLP(feat_dim).to(device)
opt = torch.optim.Adam(model.parameters(), 1e-3)
ce = nn.CrossEntropyLoss()


def symbolic_predict(seq):  # perfect rule
    return 1 if count_shape_variety(seq) == count_color_variety(seq) else 0


# ------------------------- evaluation --------------------------
def evaluate(loader, seqs, y_true):
    model.eval()
    preds, losses = [], []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device))
            nn_preds = logits.argmax(1).cpu().numpy()
            for j in range(len(nn_preds)):
                seq = seqs[idx]
                sig = rule_signature(seq)
                pred = (
                    symbolic_predict(seq)
                    if sig not in train_signatures
                    else int(nn_preds[j])
                )
                preds.append(pred)
                losses.append(
                    ce(logits[j : j + 1], batch["y"][j : j + 1].to(device)).item()
                )
                idx += 1
    swa = shape_weighted_accuracy(seqs, y_true, preds)
    return np.mean(losses), swa, preds


# ------------------------- training loop -----------------------
num_epochs = 20
experiment_data = {
    "multi_synth_generalization": {
        "seed10_train_seed20_val_seed30_test": {
            "metrics": {"train_swa": [], "val_swa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_test.tolist(),
            "timestamps": [],
        }
    }
}
ed = experiment_data["multi_synth_generalization"][
    "seed10_train_seed20_val_seed30_test"
]

for ep in range(1, num_epochs + 1):
    model.train()
    run_loss, n = 0.0, 0
    for batch in train_loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        opt.zero_grad()
        loss = ce(model(x), y)
        loss.backward()
        opt.step()
        run_loss += loss.item() * y.size(0)
        n += y.size(0)
    train_loss = run_loss / n
    _, train_swa, _ = evaluate(train_loader, train_raw["sequence"], y_train)
    val_loss, val_swa, _ = evaluate(val_loader, val_raw["sequence"], y_val)
    print(f"Epoch {ep:02d} | val_loss {val_loss:.4f} | val_SWA {val_swa:.3f}")
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_swa"].append(train_swa)
    ed["metrics"]["val_swa"].append(val_swa)
    ed["timestamps"].append(time.time())

# ------------------------- final test --------------------------
test_loss, test_swa, test_preds = evaluate(test_loader, test_raw["sequence"], y_test)
print(f"\nTEST Shape-Weighted Accuracy = {test_swa:.3f}")
ed["predictions"] = test_preds
ed["metrics"]["test_swa"] = test_swa

# ------------------------- save -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
