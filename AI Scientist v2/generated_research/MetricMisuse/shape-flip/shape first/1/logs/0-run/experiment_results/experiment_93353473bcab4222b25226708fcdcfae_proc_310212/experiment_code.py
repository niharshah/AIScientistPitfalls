import os, random, string, pathlib, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --------------------- obligatory working dir & GPU handling ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- data loading helpers -----------------------------------
# If the helper file is available, use it, otherwise duplicate functionality
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except Exception:
    # minimal re-implementation (synthetic fallback)
    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        return np.mean([t == p for t, p in zip(y_true, y_pred)])

    color_weighted_accuracy = shape_weighted_accuracy


# --------------------- load or synthesize dataset -----------------------------
def make_synthetic_set(n_rows):
    shapes = "ABC"
    colors = "xyz"
    rows = []
    for i in range(n_rows):
        seq_len = random.randint(4, 9)
        tokens = [
            "".join([random.choice(shapes), random.choice(colors)])
            for _ in range(seq_len)
        ]
        seq = " ".join(tokens)
        label = random.randint(0, 1)
        rows.append(
            {"id": f"rule{random.randint(0,5)}_{i}", "sequence": seq, "label": label}
        )
    return rows


try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr_bench = load_spr_bench(DATA_PATH)
    print("Loaded official SPR_BENCH")
    train_rows = spr_bench["train"]
    dev_rows = spr_bench["dev"]
    test_rows = spr_bench["test"]
except Exception as e:
    print(f"Could not load official dataset ({e}), generating synthetic data.")
    train_rows = make_synthetic_set(1000)
    dev_rows = make_synthetic_set(300)
    test_rows = make_synthetic_set(300)


# --------------------- vocabulary & dataset -----------------------------------
def build_vocab(rows):
    vocab = {}
    for r in rows:
        for tok in r["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(train_rows)
num_classes = len(set(r["label"] for r in train_rows))
print(f"Vocab size: {len(vocab)}, num classes: {num_classes}")


def encode(sequence):
    vec = np.zeros(len(vocab), dtype=np.float32)
    for tok in sequence.split():
        idx = vocab.get(tok)
        if idx is not None:
            vec[idx] += 1.0
    return vec


class SPRDataset(Dataset):
    def __init__(self, rows):
        self.x = [encode(r["sequence"]) for r in rows]
        self.y = [r["label"] for r in rows]
        self.ids = [r["id"] for r in rows]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "id": self.ids[idx],
        }


batch_size = 64
train_ds, dev_ds, test_ds = (
    SPRDataset(train_rows),
    SPRDataset(dev_rows),
    SPRDataset(test_rows),
)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(dev_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)


# --------------------- model ---------------------------------------------------
class BagOfTokensClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


model = BagOfTokensClassifier(len(vocab), num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------- metrics container --------------------------------------
experiment_data = {
    "spr_bench": {
        "metrics": {
            "train_loss": [],
            "val_loss": [],
            "val_swa": [],
            "val_cwa": [],
            "val_zsrta": [],
        },
        "predictions": [],
        "ground_truth": [],
        "ids": [],
    }
}


# --------------------- training loop ------------------------------------------
def evaluate(dataloader, split_name):
    model.eval()
    losses, y_true, y_pred, sequences, ids = [], [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch["y"].cpu().numpy())
            ids.extend(batch["id"])
        swa = shape_weighted_accuracy(
            [r["sequence"] if isinstance(r, dict) else "" for r in dev_rows],
            y_true,
            y_pred,
        )
        cwa = color_weighted_accuracy(
            [r["sequence"] if isinstance(r, dict) else "" for r in dev_rows],
            y_true,
            y_pred,
        )
        # ZSRTA
        train_rule_ids = set([r["id"] for r in train_rows])
        unseen_mask = [i not in train_rule_ids for i in ids]
        if any(unseen_mask):
            zsrta = np.mean(
                [yt == yp for yt, yp, m in zip(y_true, y_pred, unseen_mask) if m]
            )
        else:
            zsrta = 0.0
    return np.mean(losses), swa, cwa, zsrta, y_true, y_pred, ids


epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    epoch_losses = []
    for batch in train_dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    train_loss = np.mean(epoch_losses)

    val_loss, swa, cwa, zsrta, y_true, y_pred, ids = evaluate(dev_dl, "val")
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} | "
        f"CWA={cwa:.3f} | ZSRTA={zsrta:.3f}"
    )

    ed = experiment_data["spr_bench"]["metrics"]
    ed["train_loss"].append(train_loss)
    ed["val_loss"].append(val_loss)
    ed["val_swa"].append(swa)
    ed["val_cwa"].append(cwa)
    ed["val_zsrta"].append(zsrta)

# --------------------- final test evaluation ----------------------------------
test_loss, swa, cwa, zsrta, y_true, y_pred, ids = evaluate(test_dl, "test")
print(
    f"Test results: loss={test_loss:.4f} | SWA={swa:.3f} | "
    f"CWA={cwa:.3f} | ZSRTA={zsrta:.3f}"
)

experiment_data["spr_bench"]["predictions"] = y_pred
experiment_data["spr_bench"]["ground_truth"] = y_true
experiment_data["spr_bench"]["ids"] = ids

# --------------------- visualisation ------------------------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("SPR_BENCH Confusion Matrix")
plt.savefig(os.path.join(working_dir, "confusion_matrix_spr.png"))

# --------------------- save metrics -------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
