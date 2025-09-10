import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------------- paths / saving ----------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {}  # will be filled per batch-size setting

# ---------------------- device ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- data helpers ------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        out[split] = _load(f"{split}.csv")
    return out


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seq, y_true, y_pred):
    w = [count_color_variety(s) for s in seq]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(
        sum(w), 1e-9
    )


def shape_weighted_accuracy(seq, y_true, y_pred):
    w = [count_shape_variety(s) for s in seq]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(
        sum(w), 1e-9
    )


def harmonic_mean_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-9)


# ---------------------- synthetic data fallback -------------------------------
def create_synthetic_dataset(n_train=1000, n_dev=200, n_test=200, n_classes=4):
    def random_seq():
        length = random.randint(4, 10)
        toks = [random.choice("ABCD") + random.choice("0123") for _ in range(length)]
        return " ".join(toks)

    def label_rule(seq):
        return (count_color_variety(seq) + count_shape_variety(seq)) % n_classes

    def make_split(n):
        seqs = [random_seq() for _ in range(n)]
        return {"sequence": seqs, "label": [label_rule(s) for s in seqs]}

    ds = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [n_train, n_dev, n_test]):
        ds[split] = load_dataset("json", data_files=None, split=[], data=make_split(n))
    return ds


# ---------------------- vectoriser & dataset ----------------------------------
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
        self.y = np.array(labels, np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.tensor(self.X[idx]), "y": torch.tensor(self.y[idx])}


# ---------------------- simple MLP --------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------- load dataset ------------------------------------------
try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded official SPR_BENCH.")
except Exception:
    print("Official dataset not found -> using synthetic data.")
    spr = create_synthetic_dataset()

num_classes = len(set(spr["train"]["label"]))
print(f"Number of classes: {num_classes}")

train_ds = SPRDataset(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRDataset(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRDataset(spr["test"]["sequence"], spr["test"]["label"])

# ---------------------- sweep over batch sizes --------------------------------
batch_sizes = [32, 64, 128, 256]

for bs in batch_sizes:
    tag = f"batch_size_{bs}"
    experiment_data[tag] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = MLP(128, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_hmwa, best_state = 0.0, None
    epochs = 10
    for epoch in range(1, epochs + 1):
        # training
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * batch["y"].size(0)
        tr_loss /= len(train_ds)
        experiment_data[tag]["SPR_BENCH"]["losses"]["train"].append(tr_loss)

        # validation
        model.eval()
        val_loss, all_p, all_l, all_s = 0.0, [], [], []
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["x"])
                loss = criterion(out, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                preds = out.argmax(-1).cpu().numpy()
                labels = batch["y"].cpu().numpy()
                seq_chunk = spr["dev"]["sequence"][
                    i * dev_loader.batch_size : i * dev_loader.batch_size + len(labels)
                ]
                all_p.extend(preds.tolist())
                all_l.extend(labels.tolist())
                all_s.extend(seq_chunk)
        val_loss /= len(dev_ds)
        experiment_data[tag]["SPR_BENCH"]["losses"]["val"].append(val_loss)

        cwa = color_weighted_accuracy(all_s, all_l, all_p)
        swa = shape_weighted_accuracy(all_s, all_l, all_p)
        hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
        experiment_data[tag]["SPR_BENCH"]["metrics"]["val"].append(
            {"cwa": cwa, "swa": swa, "hmwa": hmwa}
        )
        experiment_data[tag]["SPR_BENCH"]["timestamps"].append(time.time())

        print(
            f"[bs={bs}] Epoch {epoch:02d} | val_loss={val_loss:.4f} | CWA={cwa:.4f} | SWA={swa:.4f} | HMWA={hmwa:.4f}"
        )

        if hmwa > best_hmwa:
            best_hmwa, best_state = hmwa, model.state_dict()

    # ------------------ test with best model -----------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_p, all_l, all_s = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x"])
            preds = out.argmax(-1).cpu().numpy()
            labels = batch["y"].cpu().numpy()
            seq_chunk = spr["test"]["sequence"][
                i * test_loader.batch_size : i * test_loader.batch_size + len(labels)
            ]
            all_p.extend(preds.tolist())
            all_l.extend(labels.tolist())
            all_s.extend(seq_chunk)
    cwa_t = color_weighted_accuracy(all_s, all_l, all_p)
    swa_t = shape_weighted_accuracy(all_s, all_l, all_p)
    hmwa_t = harmonic_mean_weighted_accuracy(cwa_t, swa_t)
    print(f"[bs={bs}] Test  | CWA={cwa_t:.4f} | SWA={swa_t:.4f} | HMWA={hmwa_t:.4f}\n")

    ed = experiment_data[tag]["SPR_BENCH"]
    ed["predictions"] = all_p
    ed["ground_truth"] = all_l
    ed["metrics"]["test"] = {"cwa": cwa_t, "swa": swa_t, "hmwa": hmwa_t}

# ---------------------- save everything ---------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved results to {os.path.join(working_dir, 'experiment_data.npy')}")
