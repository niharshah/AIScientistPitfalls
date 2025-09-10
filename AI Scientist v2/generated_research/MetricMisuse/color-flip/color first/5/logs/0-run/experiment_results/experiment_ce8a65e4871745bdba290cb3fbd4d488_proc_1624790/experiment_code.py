import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List

# ---------- basic setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"epoch_tuning": {"SPR_BENCH": {}}}


# ---------- load (or build) SPR-BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        out[split] = _load(f"{split}.csv")
    return out


DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:  # synthetic fallback

    def _synthetic(n):
        shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]
        seqs, labels = [], []
        for _ in range(n):
            seqs.append(
                " ".join(
                    np.random.choice(
                        [s + c for s in shapes for c in colors],
                        size=np.random.randint(3, 8),
                    )
                )
            )
            labels.append(np.random.choice(["A", "B", "C"]))
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for split, n in [("train", 200), ("dev", 50), ("test", 50)]:
        dsets[split] = load_dataset(
            "json", data_files={"train": _synthetic(n)}, split="train"
        )


# ---------- auxiliary metrics ----------
def _color_var(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def _shape_var(seq):
    return len({tok[0] for tok in seq.split()})


def _wacc(seqs, y_true, y_pred, wfunc):
    w = [wfunc(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def cwa(s, y, yhat):
    return _wacc(s, y, yhat, _color_var)


def swa(s, y, yhat):
    return _wacc(s, y, yhat, _shape_var)


def compwa(s, y, yhat):
    return _wacc(s, y, yhat, lambda x: _color_var(x) * _shape_var(x))


# ---------- vectoriser ----------
vectoriser = CountVectorizer(token_pattern=r"[^ ]+")
vectoriser.fit(dsets["train"]["sequence"])
vocab_size = len(vectoriser.vocabulary_)


def vec(batch: List[str]):
    return vectoriser.transform(batch).toarray().astype(np.float32)


# ---------- label encoding ----------
labels = sorted(set(dsets["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}
num_classes = len(labels)

y_train = np.array([lab2id[l] for l in dsets["train"]["label"]], np.int64)
y_val = np.array([lab2id[l] for l in dsets["dev"]["label"]], np.int64)
y_test = np.array([lab2id[l] for l in dsets["test"]["label"]], np.int64)
X_train, X_val, X_test = map(
    vec,
    [dsets["train"]["sequence"], dsets["dev"]["sequence"], dsets["test"]["sequence"]],
)


# ---------- model definition ----------
class MLP(nn.Module):
    def __init__(self, din, ncls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, ncls))

    def forward(self, x):
        return self.net(x)


# ---------- training routine ----------
def run_experiment(num_epochs: int, batch=64, lr=1e-3):
    torch.manual_seed(0)
    np.random.seed(0)
    model = MLP(vocab_size, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    tl, vl, v_metrics = [], [], []
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch,
    )
    for epoch in range(1, num_epochs + 1):
        # train
        model.train()
        run_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)
        tl.append(run_loss / len(tr_loader.dataset))
        # validate
        model.eval()
        vloss, preds, tgts = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                vloss += loss.item() * xb.size(0)
                preds.extend(logits.argmax(1).cpu().numpy())
                tgts.extend(yb.cpu().numpy())
        vloss /= len(val_loader.dataset)
        vl.append(vloss)
        seqs_val = dsets["dev"]["sequence"]
        acc = (np.array(preds) == np.array(tgts)).mean()
        v_metrics.append(
            {
                "epoch": epoch,
                "acc": acc,
                "cwa": cwa(seqs_val, tgts, preds),
                "swa": swa(seqs_val, tgts, preds),
                "compwa": compwa(seqs_val, tgts, preds),
            }
        )
    # test
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test).to(device))
        test_preds = test_logits.argmax(1).cpu().numpy()
    test_metrics = {
        "acc": (test_preds == y_test).mean(),
        "cwa": cwa(dsets["test"]["sequence"], y_test, test_preds),
        "swa": swa(dsets["test"]["sequence"], y_test, test_preds),
        "compwa": compwa(dsets["test"]["sequence"], y_test, test_preds),
    }
    return {
        "losses": {"train": tl, "val": vl},
        "metrics": {"val": v_metrics, "test": test_metrics},
        "predictions": test_preds,
        "ground_truth": y_test,
        "sequences": dsets["test"]["sequence"],
    }


# ---------- hyper-parameter sweep ----------
for ep in [5, 10, 20, 30]:
    print(f"\n=== Training with {ep} epochs ===")
    res = run_experiment(ep)
    experiment_data["epoch_tuning"]["SPR_BENCH"][f"epochs_{ep}"] = res
    tm = res["metrics"]["test"]
    print(
        f"Test  ACC={tm['acc']:.3f}  CWA={tm['cwa']:.3f}  SWA={tm['swa']:.3f} "
        f"CompWA={tm['compwa']:.3f}"
    )

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"\nSaved results to {os.path.join(working_dir,'experiment_data.npy')}")
