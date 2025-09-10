import os, pathlib, random, string, time, warnings, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

warnings.filterwarnings("ignore")
# ------------------- dirs & device -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


# ------------------- helpers -------------------------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def rule_signature(seq):
    return (count_shape_variety(seq), count_color_variety(seq))


def random_token():
    return random.choice(string.ascii_uppercase[:10]) + random.choice(string.digits[:5])


def generate_synth(n, seed=0):
    random.seed(seed)
    seqs, labels = [], []
    for _ in range(n):
        toks = [random_token() for _ in range(random.randint(3, 10))]
        seq = " ".join(toks)
        lbl = int(count_shape_variety(seq) == count_color_variety(seq))
        seqs.append(seq)
        labels.append(lbl)
    return {"id": list(range(n)), "sequence": seqs, "label": labels}


def load_spr(path):
    if path.exists():

        def _ld(f):
            return load_dataset("csv", data_files=str(path / f), split="train")

        print("Loading real SPR_BENCH")
        return DatasetDict(
            train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv")
        )
    print("SPR_BENCH not found – using synthetic")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synth(2000, 1)),
        dev=HFDataset.from_dict(generate_synth(500, 2)),
        test=HFDataset.from_dict(generate_synth(1000, 3)),
    )


DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr(DATA_PATH)

# ------------------- encoding ------------------------
shape_idx = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
colour_idx = {d: i for i, d in enumerate(string.digits[:10])}
feature_dim = 26 + 10 + 3


def encode(seq):
    v = np.zeros(feature_dim, dtype=np.float32)
    toks = seq.split()
    for t in toks:
        if len(t) < 2:
            continue
        v[shape_idx[t[0]]] += 1
        v[26 + colour_idx[t[1]]] += 1
    v[-3:] = [len(toks), count_shape_variety(seq), count_color_variety(seq)]
    return v


def encode_set(ds):
    feats = np.stack([encode(s) for s in ds["sequence"]])
    labels = np.array(ds["label"], dtype=np.int64)
    sigs = [rule_signature(s) for s in ds["sequence"]]
    return feats, labels, sigs


X_train, y_train, sig_train = encode_set(dsets["train"])
X_dev, y_dev, sig_dev = encode_set(dsets["dev"])
X_test, y_test, sig_test = encode_set(dsets["test"])


class SPRTorchDS(Dataset):
    def __init__(s, X, y):
        s.X = torch.tensor(X)
        s.y = torch.tensor(y)

    def __len__(s):
        return len(s.X)

    def __getitem__(s, i):
        return {"x": s.X[i], "y": s.y[i]}


train_loader = lambda bs: DataLoader(
    SPRTorchDS(X_train, y_train), batch_size=bs, shuffle=True
)
dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev), batch_size=256)
test_loader = DataLoader(SPRTorchDS(X_test, y_test), batch_size=256)


# ------------------- model ---------------------------
class MLP(nn.Module):
    def __init__(self, act_name, hidden=64, in_dim=feature_dim, out=2):
        super().__init__()
        act_layer = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }[act_name]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act_layer, nn.Linear(hidden, out)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- evaluation -------------------------
def eval_model(model, loader, sigs_all, unseen):
    model.eval()
    tot = correct = tot_u = cor_u = 0
    preds = []
    with torch.no_grad():
        idx = 0
        for b in loader:
            x, y = b["x"].to(device), b["y"].to(device)
            pr = model(x).argmax(1)
            preds.extend(pr.cpu().numpy())
            tot += y.size(0)
            correct += (pr == y).sum().item()
            for p_true in pr.cpu().numpy():
                sig = sigs_all[idx]
                if sig in unseen:
                    tot_u += 1
                    if p_true == y.cpu().numpy()[idx % y.size(0)]:
                        cor_u += 1
                idx += 1
    return correct / tot, (cor_u / tot_u if tot_u else 0.0), preds


# ------------------- experiment loop -----------------
experiment_data = {"activation_function": {"SPR_BENCH": {}}}
unseen_dev = {s for s in sig_dev if s not in set(sig_train)}
unseen_test = {s for s in sig_test if s not in set(sig_train)}
EPOCHS, LR, BS = 5, 1e-3, 64
for act in ["relu", "leaky_relu", "gelu", "tanh"]:
    print(f"\n=== Activation: {act} ===")
    model = MLP(act).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    edict = {
        "metrics": {"train_acc": [], "val_acc": [], "val_ura": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "timestamps": [],
    }
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0
        cor = tot = 0
        for batch in train_loader(BS):
            optim.zero_grad()
            x, y = batch["x"].to(device), batch["y"].to(device)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * y.size(0)
            pr = out.argmax(1)
            cor += (pr == y).sum().item()
            tot += y.size(0)
        train_loss, train_acc = loss_sum / tot, cor / tot
        val_acc, val_ura, _ = eval_model(model, dev_loader, sig_dev, unseen_dev)
        print(
            f"Ep{epoch}  loss={train_loss:.3f} acc={train_acc:.3f}  val={val_acc:.3f} ura={val_ura:.3f}"
        )
        edict["losses"]["train"].append(train_loss)
        edict["metrics"]["train_acc"].append(train_acc)
        edict["metrics"]["val_acc"].append(val_acc)
        edict["metrics"]["val_ura"].append(val_ura)
        edict["timestamps"].append(time.time())
    test_acc, test_ura, test_preds = eval_model(
        model, test_loader, sig_test, unseen_test
    )
    print(f"Test acc={test_acc:.3f}  URA={test_ura:.3f}")
    edict["metrics"]["test_acc"] = test_acc
    edict["metrics"]["test_ura"] = test_ura
    edict["predictions"] = test_preds
    experiment_data["activation_function"]["SPR_BENCH"][act] = edict

# ------------------- save ----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
