import os, pathlib, random, time, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ----------------- bookkeeping / saving ---------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"adam_beta1": {"SPR_BENCH": {}}}  # will hold a dict per beta1 value

# ----------------- device -----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- data utilities ---------------------------------------------
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


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


def harmonic_mean_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if cwa + swa > 0 else 0.0


# ------------- synthetic fallback ---------------------------------------------
def create_synth_split(n, n_classes=4):
    def rand_seq():
        l = random.randint(4, 10)
        toks = [random.choice("ABCD") + random.choice("0123") for _ in range(l)]
        return " ".join(toks)

    seqs = [rand_seq() for _ in range(n)]
    labels = [
        (count_color_variety(s) + count_shape_variety(s)) % n_classes for s in seqs
    ]
    return {"sequence": seqs, "label": labels}


def create_synthetic_dataset():
    return DatasetDict(
        {
            "train": load_dataset("json", split=[], data=create_synth_split(1000)),
            "dev": load_dataset("json", split=[], data=create_synth_split(200)),
            "test": load_dataset("json", split=[], data=create_synth_split(200)),
        }
    )


# ---------------- feature extraction ------------------------------------------
def seq_to_vec(seq: str):
    vec = np.zeros(128, dtype=np.float32)
    for ch in seq.replace(" ", ""):
        vec[ord(ch) if ord(ch) < 128 else 0] += 1
    l = len(seq.replace(" ", ""))
    return vec / l if l else vec


class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.X = np.stack([seq_to_vec(s) for s in seqs])
        self.y = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.tensor(self.X[idx]), "y": torch.tensor(self.y[idx])}


# ---------------- model --------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- load data ----------------------------------------------------
try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded official SPR_BENCH.")
except Exception as e:
    print("Official dataset not found, using synthetic toy data.")
    spr = create_synthetic_dataset()

n_classes = len(set(spr["train"]["label"]))
train_ds = SPRDataset(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRDataset(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRDataset(spr["test"]["sequence"], spr["test"]["label"])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# ---------------- hyperparameter grid -----------------------------------------
beta1_values = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95]
max_epochs = 10

for b1 in beta1_values:
    key = f"beta1_{b1:.2f}"
    experiment_data["adam_beta1"]["SPR_BENCH"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    # reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = MLP(128, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(b1, 0.999))

    best_state, best_hmwa = None, 0.0

    for epoch in range(1, max_epochs + 1):
        # -------- train --------
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["y"].size(0)
        train_loss = running_loss / len(train_ds)
        experiment_data["adam_beta1"]["SPR_BENCH"][key]["losses"]["train"].append(
            train_loss
        )

        # -------- validation --------
        model.eval()
        val_loss = 0.0
        preds = []
        labels = []
        seqs = []
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["x"])
                loss = criterion(out, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                pr = out.argmax(dim=-1).cpu().numpy()
                lb = batch["y"].cpu().numpy()
                seq_slice = spr["dev"]["sequence"][
                    i * dev_loader.batch_size : i * dev_loader.batch_size + len(lb)
                ]
                preds.extend(pr.tolist())
                labels.extend(lb.tolist())
                seqs.extend(seq_slice)
        val_loss /= len(dev_ds)
        experiment_data["adam_beta1"]["SPR_BENCH"][key]["losses"]["val"].append(
            val_loss
        )

        cwa = color_weighted_accuracy(seqs, labels, preds)
        swa = shape_weighted_accuracy(seqs, labels, preds)
        hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
        experiment_data["adam_beta1"]["SPR_BENCH"][key]["metrics"]["val"].append(
            {"cwa": cwa, "swa": swa, "hmwa": hmwa}
        )
        experiment_data["adam_beta1"]["SPR_BENCH"][key]["timestamps"].append(
            time.time()
        )

        if hmwa > best_hmwa:
            best_hmwa = hmwa
            best_state = model.state_dict()
        print(
            f"[β1={b1:.2f}] Epoch {epoch}: loss={val_loss:.4f}, CWA={cwa:.3f}, SWA={swa:.3f}, HMWA={hmwa:.3f}"
        )

    # -------------- test with best checkpoint ---------------------------------
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    labels = []
    seqs = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x"])
            pr = out.argmax(dim=-1).cpu().numpy()
            lb = batch["y"].cpu().numpy()
            seq_slice = spr["test"]["sequence"][
                i * test_loader.batch_size : i * test_loader.batch_size + len(lb)
            ]
            preds.extend(pr.tolist())
            labels.extend(lb.tolist())
            seqs.extend(seq_slice)
    cwa_test = color_weighted_accuracy(seqs, labels, preds)
    swa_test = shape_weighted_accuracy(seqs, labels, preds)
    hmwa_test = harmonic_mean_weighted_accuracy(cwa_test, swa_test)
    print(
        f"[β1={b1:.2f}] TEST: CWA={cwa_test:.3f}, SWA={swa_test:.3f}, HMWA={hmwa_test:.3f}"
    )

    # store test predictions
    d = experiment_data["adam_beta1"]["SPR_BENCH"][key]
    d["predictions"] = preds
    d["ground_truth"] = labels
    d["metrics"]["test"] = {"cwa": cwa_test, "swa": swa_test, "hmwa": hmwa_test}

    # free gpu memory
    del model
    torch.cuda.empty_cache()

# ---------------- save all data -----------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved all results to {os.path.join(working_dir,'experiment_data.npy')}")
