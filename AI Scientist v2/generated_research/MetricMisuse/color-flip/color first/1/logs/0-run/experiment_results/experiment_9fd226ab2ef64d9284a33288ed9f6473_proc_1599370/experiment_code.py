# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, load_from_disk

# ---------------- GPU handling -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- utility: data loader from prompt ----------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def harmonic_mean_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) > 0 else 0.0


# ----------------- fallback synthetic data ------------------------------------
def create_synthetic_dataset(n_train=1000, n_dev=200, n_test=200, n_classes=4):
    def random_seq():
        length = random.randint(4, 10)
        toks = []
        for _ in range(length):
            shape = random.choice("ABCD")
            color = random.choice("0123")
            toks.append(shape + color)
        return " ".join(toks)

    def label_rule(seq):
        # simple rule: class is (color variety + shape variety) mod n_classes
        return (count_color_variety(seq) + count_shape_variety(seq)) % n_classes

    def make_split(n):
        seqs = [random_seq() for _ in range(n)]
        labs = [label_rule(s) for s in seqs]
        return {"sequence": seqs, "label": labs}

    ds = DatasetDict()
    ds["train"] = load_dataset(
        "json", data_files=None, split=[], data=make_split(n_train)
    )
    ds["dev"] = load_dataset("json", data_files=None, split=[], data=make_split(n_dev))
    ds["test"] = load_dataset(
        "json", data_files=None, split=[], data=make_split(n_test)
    )
    return ds


# ---------------- feature extraction ------------------------------------------
def seq_to_vec(seq: str) -> np.ndarray:
    vec = np.zeros(128, dtype=np.float32)
    chars = seq.replace(" ", "")
    if len(chars) == 0:
        return vec
    for ch in chars:
        idx = ord(ch) if ord(ch) < 128 else 0
        vec[idx] += 1.0
    vec /= len(chars)
    return vec


class SPRDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = np.stack([seq_to_vec(s) for s in sequences])
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


# ---------------- experiment data structure -----------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ---------------- main flow ----------------------------------------------------
def main():
    # attempt to load official data
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        spr = load_spr_bench(DATA_PATH)
        print("Loaded SPR_BENCH from disk.")
    except Exception as e:
        print("Official dataset not found, falling back to synthetic toy data.")
        spr = create_synthetic_dataset()

    num_classes = len(set(spr["train"]["label"]))
    print(f"Number of classes: {num_classes}")

    train_ds = SPRDataset(spr["train"]["sequence"], spr["train"]["label"])
    dev_ds = SPRDataset(spr["dev"]["sequence"], spr["dev"]["label"])
    test_ds = SPRDataset(spr["test"]["sequence"], spr["test"]["label"])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = MLP(128, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_hmwa = 0.0
    best_state = None
    epochs = 10

    for epoch in range(1, epochs + 1):
        # ---- training ----
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["y"].size(0)
        train_loss = running_loss / len(train_ds)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_seqs = [], [], []
        with torch.no_grad():
            for i, batch_idx in enumerate(dev_loader):
                batch = {
                    k: v.to(device)
                    for k, v in batch_idx.items()
                    if isinstance(v, torch.Tensor)
                }
                out = model(batch["x"])
                loss = criterion(out, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                preds = out.argmax(dim=-1).cpu().numpy()
                labels = batch["y"].cpu().numpy()
                seqs_idx = spr["dev"]["sequence"][
                    i * dev_loader.batch_size : i * dev_loader.batch_size + len(labels)
                ]
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                all_seqs.extend(seqs_idx)
        val_loss /= len(dev_ds)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

        cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
        swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
        hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            {"cwa": cwa, "swa": swa, "hmwa": hmwa}
        )
        experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f}, CWA={cwa:.4f}, SWA={swa:.4f}, HMWA={hmwa:.4f}"
        )

        if hmwa > best_hmwa:
            best_hmwa = hmwa
            best_state = model.state_dict()

    # ----------------- test evaluation with best model -------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_preds, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            out = model(batch["x"])
            preds = out.argmax(dim=-1).cpu().numpy()
            labels = batch["y"].cpu().numpy()
            seqs_idx = spr["test"]["sequence"][
                i * test_loader.batch_size : i * test_loader.batch_size + len(labels)
            ]
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_seqs.extend(seqs_idx)
    cwa_test = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    swa_test = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    hmwa_test = harmonic_mean_weighted_accuracy(cwa_test, swa_test)
    print(f"\nTest set: CWA={cwa_test:.4f}, SWA={swa_test:.4f}, HMWA={hmwa_test:.4f}")

    experiment_data["SPR_BENCH"]["predictions"] = all_preds
    experiment_data["SPR_BENCH"]["ground_truth"] = all_labels
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print(f'All metrics saved to {os.path.join(working_dir, "experiment_data.npy")}')


# execute immediately
main()
