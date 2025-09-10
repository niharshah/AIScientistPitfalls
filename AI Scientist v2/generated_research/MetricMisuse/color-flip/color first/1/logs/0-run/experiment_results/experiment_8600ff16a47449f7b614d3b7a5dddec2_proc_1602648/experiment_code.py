import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------ set-up ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------------- metrics & utils --------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [w_i if yt == yp else 0 for w_i, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [w_i if yt == yp else 0 for w_i, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def harmonic_mean_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if cwa + swa > 0 else 0.0


# ---------------- synthetic fallback -----------------------------------------
def create_synthetic_dataset(n_train=1000, n_dev=200, n_test=200, n_classes=4):
    def rseq():
        toks = [
            random.choice("ABCD") + random.choice("0123")
            for _ in range(random.randint(4, 10))
        ]
        return " ".join(toks)

    def lbl(seq):
        return (count_color_variety(seq) + count_shape_variety(seq)) % n_classes

    def mk(n):
        seqs = [rseq() for _ in range(n)]
        labs = [lbl(s) for s in seqs]
        return {"sequence": seqs, "label": labs}

    return DatasetDict(
        train=load_dataset("json", data=mk(n_train), split=[]),
        dev=load_dataset("json", data=mk(n_dev), split=[]),
        test=load_dataset("json", data=mk(n_test), split=[]),
    )


# ---------------- feature extraction -----------------------------------------
def seq_to_vec(seq):
    v = np.zeros(128, dtype=np.float32)
    chars = seq.replace(" ", "")
    if chars:
        for ch in chars:
            idx = ord(ch) if ord(ch) < 128 else 0
            v[idx] += 1
        v /= len(chars)
    return v


class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.X = np.stack([seq_to_vec(s) for s in seqs]).astype(np.float32)
        self.y = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.tensor(self.X[idx]), "y": torch.tensor(self.y[idx])}


# ---------------- model -------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x)


# ---------------- experiment log ---------------------------------------------
experiment_data = {"batch_size_tuning": {"SPR_BENCH": {}}}

# ---------------- data load ---------------------------------------------------
try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded official SPR_BENCH dataset.")
except Exception:
    print("Official dataset not found. Using synthetic data.")
    spr = create_synthetic_dataset()

num_classes = len(set(spr["train"]["label"]))
print(f"Classes: {num_classes}")

train_ds = SPRDataset(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRDataset(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRDataset(spr["test"]["sequence"], spr["test"]["label"])

batch_sizes = [32, 64, 128, 256, 512]
overall_best = {"hmwa": 0.0, "bs": None, "state": None}

for bs in batch_sizes:
    print(f"\n--- Training with batch size {bs} ---")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=max(256, bs * 2), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=max(256, bs * 2), shuffle=False)

    model = MLP(128, num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10
    run_log = {"metrics": {"train": [], "val": []}, "losses": {"train": [], "val": []}}

    best_hmwa_bs = 0.0
    best_state_bs = None

    for ep in range(1, epochs + 1):
        # train
        model.train()
        run_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(batch["x"])
            loss = crit(out, batch["y"])
            loss.backward()
            opt.step()
            run_loss += loss.item() * batch["y"].size(0)
        train_loss = run_loss / len(train_ds)
        run_log["losses"]["train"].append(train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        preds = []
        labels = []
        seqs = []
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["x"])
                loss = crit(out, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                p = out.argmax(-1).cpu().numpy()
                l = batch["y"].cpu().numpy()
                start = i * dev_loader.batch_size
                seqs.extend(spr["dev"]["sequence"][start : start + len(l)])
                preds.extend(p.tolist())
                labels.extend(l.tolist())
        val_loss /= len(dev_ds)
        run_log["losses"]["val"].append(val_loss)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        swa = shape_weighted_accuracy(seqs, labels, preds)
        hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
        run_log["metrics"]["val"].append({"cwa": cwa, "swa": swa, "hmwa": hmwa})
        print(
            f"Epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} HMWA={hmwa:.4f}"
        )

        if hmwa > best_hmwa_bs:
            best_hmwa_bs = hmwa
            best_state_bs = model.state_dict()

    # save config data
    experiment_data["batch_size_tuning"]["SPR_BENCH"][f"bs_{bs}"] = run_log

    # test with best checkpoint for this batch size
    model.load_state_dict(best_state_bs)
    model.eval()
    preds = []
    labels = []
    seqs = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x"])
            p = out.argmax(-1).cpu().numpy()
            l = batch["y"].cpu().numpy()
            start = i * test_loader.batch_size
            seqs.extend(spr["test"]["sequence"][start : start + len(l)])
            preds.extend(p.tolist())
            labels.extend(l.tolist())
    cwa_t = color_weighted_accuracy(seqs, labels, preds)
    swa_t = shape_weighted_accuracy(seqs, labels, preds)
    hmwa_t = harmonic_mean_weighted_accuracy(cwa_t, swa_t)
    print(f"Batch size {bs} test: HMWA={hmwa_t:.4f}")
    run_log["predictions"] = preds
    run_log["ground_truth"] = labels
    run_log["test_hmwa"] = hmwa_t

    # overall best across batch sizes
    if best_hmwa_bs > overall_best["hmwa"]:
        overall_best = {"hmwa": best_hmwa_bs, "bs": bs, "state": best_state_bs}

print(
    f"\nBest dev HMWA {overall_best['hmwa']:.4f} achieved with batch size {overall_best['bs']}."
)

# ----------------- save logs --------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
