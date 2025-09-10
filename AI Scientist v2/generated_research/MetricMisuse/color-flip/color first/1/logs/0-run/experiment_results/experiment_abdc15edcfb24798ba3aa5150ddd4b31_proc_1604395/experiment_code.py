import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -------------------------- I/O & experiment container -------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"dropout_rate": {}}  # <-- top-level key
# ------------------------------ device -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- data helpers (unchanged) -------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_color_variety(seq: str):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def harmonic_mean_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if cwa + swa > 0 else 0.0


# --------------------------- synthetic fallback --------------------------------
def create_synth(n_train=1000, n_dev=200, n_test=200, n_cls=4):
    def rand_seq():
        toks = [
            "" + random.choice("ABCD") + random.choice("0123")
            for _ in range(random.randint(4, 10))
        ]
        return " ".join(toks)

    def rule(s):
        return (count_color_variety(s) + count_shape_variety(s)) % n_cls

    def split(n):
        seqs = [rand_seq() for _ in range(n)]
        labs = [rule(s) for s in seqs]
        return {"sequence": seqs, "label": labs}

    return DatasetDict(
        {
            k: load_dataset("json", data=split(sz), split=[])
            for k, sz in zip(["train", "dev", "test"], [n_train, n_dev, n_test])
        }
    )


# --------------------------- feature extraction --------------------------------
def seq_to_vec(seq: str) -> np.ndarray:
    v = np.zeros(128, dtype=np.float32)
    chars = seq.replace(" ", "")
    if chars:
        for ch in chars:
            idx = ord(ch) if ord(ch) < 128 else 0
            v[idx] += 1.0
        v /= len(chars)
    return v


class SPRDataset(Dataset):
    def __init__(self, seqs, labs):
        self.X = np.stack([seq_to_vec(s) for s in seqs])
        self.y = np.array(labs, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": torch.tensor(self.X[i]), "y": torch.tensor(self.y[i])}


# --------------------------- model with dropout --------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_cls, drop_prob):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------- main loop -------------------------------------
def run(drop_prob):
    key = f"SPR_BENCH_p{1-drop_prob:.1f}"  # e.g., keep-prob 0.8 => p0.8
    exp_entry = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "dropout": drop_prob,
        "timestamps": [],
    }
    # -------- load data ---------
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        ds = load_spr_bench(DATA_PATH)
        print("Loaded official SPR_BENCH.")
    except Exception:
        print("Official dataset not found, using synthetic.")
        ds = create_synth()
    num_classes = len(set(ds["train"]["label"]))
    train_loader = DataLoader(
        SPRDataset(ds["train"]["sequence"], ds["train"]["label"]),
        batch_size=128,
        shuffle=True,
    )
    dev_loader = DataLoader(
        SPRDataset(ds["dev"]["sequence"], ds["dev"]["label"]),
        batch_size=256,
        shuffle=False,
    )
    test_loader = DataLoader(
        SPRDataset(ds["test"]["sequence"], ds["test"]["label"]),
        batch_size=256,
        shuffle=False,
    )

    model = MLP(128, num_classes, drop_prob).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_hmwa, best_state = 0.0, None
    epochs = 10

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(batch["x"])
            loss = crit(out, batch["y"])
            loss.backward()
            opt.step()
            tr_loss += loss.item() * batch["y"].size(0)
        tr_loss /= len(train_loader.dataset)
        exp_entry["losses"]["train"].append(tr_loss)

        # -------- validation --------
        model.eval()
        val_loss = 0.0
        preds = []
        labels = []
        seqs = []
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                b = {k: v.to(device) for k, v in batch.items()}
                out = model(b["x"])
                loss = crit(out, b["y"])
                val_loss += loss.item() * b["y"].size(0)
                p = out.argmax(-1).cpu().numpy()
                l = b["y"].cpu().numpy()
                preds.extend(p.tolist())
                labels.extend(l.tolist())
                seqs.extend(
                    ds["dev"]["sequence"][
                        i * dev_loader.batch_size : i * dev_loader.batch_size + len(l)
                    ]
                )
        val_loss /= len(dev_loader.dataset)
        exp_entry["losses"]["val"].append(val_loss)

        cwa = color_weighted_accuracy(seqs, labels, preds)
        swa = shape_weighted_accuracy(seqs, labels, preds)
        hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
        exp_entry["metrics"]["val"].append({"cwa": cwa, "swa": swa, "hmwa": hmwa})
        exp_entry["timestamps"].append(time.time())
        print(
            f"[drop={drop_prob}] Epoch {ep}: val_loss={val_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} HMWA={hmwa:.3f}"
        )
        if hmwa > best_hmwa:
            best_hmwa, best_state = hmwa, model.state_dict()
    # ------------------ test using best checkpoint -----------------------------
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    labels = []
    seqs = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            b = {k: v.to(device) for k, v in batch.items()}
            out = model(b["x"])
            p = out.argmax(-1).cpu().numpy()
            l = b["y"].cpu().numpy()
            preds.extend(p.tolist())
            labels.extend(l.tolist())
            seqs.extend(
                ds["test"]["sequence"][
                    i * test_loader.batch_size : i * test_loader.batch_size + len(l)
                ]
            )
    cwa = color_weighted_accuracy(seqs, labels, preds)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
    exp_entry["metrics"]["test"] = {"cwa": cwa, "swa": swa, "hmwa": hmwa}
    exp_entry["predictions"], exp_entry["ground_truth"] = preds, labels
    print(f"[drop={drop_prob}] Test: CWA={cwa:.3f} SWA={swa:.3f} HMWA={hmwa:.3f}")
    experiment_data["dropout_rate"][key] = exp_entry


for dp in (0.2, 0.4, 0.6):
    run(dp)

# --------------------------- save everything -----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved all results to {os.path.join(working_dir,'experiment_data.npy')}")
