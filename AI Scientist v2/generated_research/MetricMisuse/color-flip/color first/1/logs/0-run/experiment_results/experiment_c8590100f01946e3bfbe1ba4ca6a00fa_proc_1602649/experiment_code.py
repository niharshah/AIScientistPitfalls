import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, load_from_disk

# ---------------- basic setup --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"hidden_dim_tuning": {}}  # master container


# ---------------- data helpers -------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper for each split
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_color_variety(seq):  # number of distinct colors
    return len(set(token[1] for token in seq.strip().split() if len(token) > 1))


def count_shape_variety(seq):  # number of distinct shapes
    return len(set(token[0] for token in seq.strip().split() if token))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def harmonic_mean_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0


# --------------- synthetic fallback -------------------------------------------
def create_synth(n_train=1000, n_dev=200, n_test=200, n_cls=4):
    def rand_seq():
        ln = random.randint(4, 10)
        return " ".join(
            random.choice("ABCD") + random.choice("0123") for _ in range(ln)
        )

    def rule(s):  # simple rule for label
        return (count_color_variety(s) + count_shape_variety(s)) % n_cls

    def make(n):
        seqs = [rand_seq() for _ in range(n)]
        return {"sequence": seqs, "label": [rule(s) for s in seqs]}

    return DatasetDict(
        train=load_dataset("json", data=make(n_train), split=[]),
        dev=load_dataset("json", data=make(n_dev), split=[]),
        test=load_dataset("json", data=make(n_test), split=[]),
    )


# --------------- feature extraction -------------------------------------------
def seq_to_vec(seq: str) -> np.ndarray:
    v = np.zeros(128, dtype=np.float32)
    chars = seq.replace(" ", "")
    for ch in chars:
        idx = ord(ch) if ord(ch) < 128 else 0
        v[idx] += 1.0
    if len(chars):
        v /= len(chars)
    return v


class SPRDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = np.stack([seq_to_vec(s) for s in sequences])
        self.y = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.tensor(self.X[idx]), "y": torch.tensor(self.y[idx])}


# --------------- model --------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# --------------- attempt data load -------------------------------------------
try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded official SPR_BENCH.")
except Exception as e:
    print("Official data not found – generating synthetic dataset.")
    spr = create_synth()

num_classes = len(set(spr["train"]["label"]))
train_ds = SPRDataset(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRDataset(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRDataset(spr["test"]["sequence"], spr["test"]["label"])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# --------------- hyperparameter search ---------------------------------------
for hidden_dim in [32, 64, 128, 256]:
    tag = f"hidden_dim_{hidden_dim}"
    experiment_data["hidden_dim_tuning"][tag] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    model = MLP(128, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_hmwa, best_state = 0, None
    epochs = 10
    # ---------------- training loop -----------------
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            loss.backward()
            optim.step()
            run_loss += loss.item() * batch["y"].size(0)
        train_loss = run_loss / len(train_ds)
        experiment_data["hidden_dim_tuning"][tag]["losses"]["train"].append(train_loss)
        # -------------- validation -------------------
        model.eval()
        val_loss = 0.0
        all_preds, all_lbls, all_seqs = [], [], []
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["x"])
                loss = criterion(out, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                preds = out.argmax(-1).cpu().numpy()
                labels = batch["y"].cpu().numpy()
                seqs = spr["dev"]["sequence"][
                    i * dev_loader.batch_size : i * dev_loader.batch_size + len(labels)
                ]
                all_preds.extend(preds.tolist())
                all_lbls.extend(labels.tolist())
                all_seqs.extend(seqs)
        val_loss /= len(dev_ds)
        experiment_data["hidden_dim_tuning"][tag]["losses"]["val"].append(val_loss)
        cwa = color_weighted_accuracy(all_seqs, all_lbls, all_preds)
        swa = shape_weighted_accuracy(all_seqs, all_lbls, all_preds)
        hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
        experiment_data["hidden_dim_tuning"][tag]["metrics"]["val"].append(
            {"cwa": cwa, "swa": swa, "hmwa": hmwa}
        )
        experiment_data["hidden_dim_tuning"][tag]["timestamps"].append(time.time())
        print(f"[{tag}] Epoch {epoch}: val_loss={val_loss:.4f}, HMWA={hmwa:.4f}")
        if hmwa > best_hmwa:
            best_hmwa, best_state = hmwa, model.state_dict()
    # ------------- test with best checkpoint -------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_preds, all_lbls, all_seqs = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x"])
            preds = out.argmax(-1).cpu().numpy()
            labels = batch["y"].cpu().numpy()
            seqs = spr["test"]["sequence"][
                i * test_loader.batch_size : i * test_loader.batch_size + len(labels)
            ]
            all_preds.extend(preds.tolist())
            all_lbls.extend(labels.tolist())
            all_seqs.extend(seqs)
    cwa_t = color_weighted_accuracy(all_seqs, all_lbls, all_preds)
    swa_t = shape_weighted_accuracy(all_seqs, all_lbls, all_preds)
    hmwa_t = harmonic_mean_weighted_accuracy(cwa_t, swa_t)
    print(f"[{tag}] Test: CWA={cwa_t:.4f}, SWA={swa_t:.4f}, HMWA={hmwa_t:.4f}")
    experiment_data["hidden_dim_tuning"][tag]["predictions"] = all_preds
    experiment_data["hidden_dim_tuning"][tag]["ground_truth"] = all_lbls
    experiment_data["hidden_dim_tuning"][tag]["metrics"]["test"] = {
        "cwa": cwa_t,
        "swa": swa_t,
        "hmwa": hmwa_t,
    }

# --------------- save all results ---------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
