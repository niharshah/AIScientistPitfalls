import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- saving dict --------------------------------------------------
experiment_data = {}

# ---------------- GPU ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- data helpers -------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def harmonic_mean_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# ---------------- synthetic fallback ------------------------------------------
def create_synthetic_dataset(n_train=1000, n_dev=200, n_test=200, n_classes=4):
    def random_seq():
        toks = [
            random.choice("ABCD") + random.choice("0123")
            for _ in range(random.randint(4, 10))
        ]
        return " ".join(toks)

    def label(seq):
        return (count_color_variety(seq) + count_shape_variety(seq)) % n_classes

    def make_split(n):
        seqs = [random_seq() for _ in range(n)]
        return {"sequence": seqs, "label": [label(s) for s in seqs]}

    ds = DatasetDict()
    ds["train"] = load_dataset("json", split=[], data=make_split(n_train))
    ds["dev"] = load_dataset("json", split=[], data=make_split(n_dev))
    ds["test"] = load_dataset("json", split=[], data=make_split(n_test))
    return ds


# ---------------- vectorizer ---------------------------------------------------
def seq_to_vec(seq: str) -> np.ndarray:
    vec = np.zeros(128, dtype=np.float32)
    chars = seq.replace(" ", "")
    for ch in chars:
        idx = ord(ch) if ord(ch) < 128 else 0
        vec[idx] += 1.0
    if len(chars):
        vec /= len(chars)
    return vec


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
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- load data ----------------------------------------------------
try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded official SPR_BENCH.")
except Exception:
    print("Official dataset not found, using synthetic data.")
    spr = create_synthetic_dataset()

num_classes = len(set(spr["train"]["label"]))
train_ds = SPRDataset(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRDataset(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRDataset(spr["test"]["sequence"], spr["test"]["label"])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# ---------------- training loop per hidden_dim --------------------------------
hidden_dims = [32, 64, 128, 256]
epochs = 10

for hd in hidden_dims:
    tag = f"hidden_dim_{hd}"
    print(f"\n--- Training model with {hd} hidden units ---")
    experiment_data[tag] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
    model = MLP(128, hd, num_classes).to(device)
    criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(
        model.parameters(), lr=1e-3
    )
    best_hmwa, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        # training
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
        tr_loss = running_loss / len(train_ds)
        experiment_data[tag]["SPR_BENCH"]["losses"]["train"].append(tr_loss)

        # validation
        model.eval()
        val_loss = 0.0
        preds, labels, seqs = [], [], []
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                bt = {k: v.to(device) for k, v in batch.items()}
                out = model(bt["x"])
                loss = criterion(out, bt["y"])
                val_loss += loss.item() * bt["y"].size(0)
                p = out.argmax(-1).cpu().numpy()
                l = bt["y"].cpu().numpy()
                s = spr["dev"]["sequence"][
                    i * dev_loader.batch_size : i * dev_loader.batch_size + len(l)
                ]
                preds.extend(p.tolist())
                labels.extend(l.tolist())
                seqs.extend(s)
        val_loss /= len(dev_ds)
        experiment_data[tag]["SPR_BENCH"]["losses"]["val"].append(val_loss)

        cwa = color_weighted_accuracy(seqs, labels, preds)
        swa = shape_weighted_accuracy(seqs, labels, preds)
        hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
        experiment_data[tag]["SPR_BENCH"]["metrics"]["val"].append(
            {"cwa": cwa, "swa": swa, "hmwa": hmwa}
        )
        experiment_data[tag]["SPR_BENCH"]["timestamps"].append(time.time())
        print(
            f"Epoch {ep}: val_loss={val_loss:.4f}, CWA={cwa:.4f}, SWA={swa:.4f}, HMWA={hmwa:.4f}"
        )

        if hmwa > best_hmwa:
            best_hmwa, best_state = hmwa, model.state_dict()

    # test with best model
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            bt = {k: v.to(device) for k, v in batch.items()}
            out = model(bt["x"])
            p = out.argmax(-1).cpu().numpy()
            l = bt["y"].cpu().numpy()
            s = spr["test"]["sequence"][
                i * test_loader.batch_size : i * test_loader.batch_size + len(l)
            ]
            preds.extend(p.tolist())
            labels.extend(l.tolist())
            seqs.extend(s)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
    print(f"Hidden_dim {hd} test: CWA={cwa:.4f}, SWA={swa:.4f}, HMWA={hmwa:.4f}")
    ed = experiment_data[tag]["SPR_BENCH"]
    ed["predictions"], ed["ground_truth"] = preds, labels
    ed["metrics"]["test"] = {"cwa": cwa, "swa": swa, "hmwa": hmwa}

    # free memory
    del model
    torch.cuda.empty_cache()

# ---------------- save all -----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    f"\nAll experiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}"
)
