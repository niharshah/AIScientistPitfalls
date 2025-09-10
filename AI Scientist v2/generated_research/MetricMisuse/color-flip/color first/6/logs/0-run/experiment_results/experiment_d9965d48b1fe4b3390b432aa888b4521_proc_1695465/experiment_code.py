import os, pathlib, random, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ---------------------------------------------------------------
# working directory & device set-up
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------
# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------------
# helper to load SPR_BENCH (same logic as provided utility)
def load_spr_bench(root: pathlib.Path):
    def _l(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return {"train": _l("train.csv"), "dev": _l("dev.csv"), "test": _l("test.csv")}


# locate dataset folder
DATA_PATH = None
for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if p.exists():
        DATA_PATH = p
        break
if DATA_PATH is None:
    raise FileNotFoundError("SPR_BENCH dataset folder not found")

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ----------------------------------------------------------------
# build vocabularies STRICTLY from the training split (bug-fix)
def extract_shapes_colors(split):
    shapes, colors, labels = set(), set(), set()
    for s, lab in zip(split["sequence"], split["label"]):
        for tok in s.split():
            if not tok:
                continue
            shapes.add(tok[0])
            colors.add(tok[1:])
        labels.add(lab)
    return shapes, colors, labels


train_shapes, train_colors, train_labels = extract_shapes_colors(dsets["train"])

# index 0 = PAD, 1 = UNK
shape2idx = {"<PAD>": 0, "<UNK>": 1}
color2idx = {"<PAD>": 0, "<UNK>": 1}
for s in sorted(train_shapes):
    shape2idx[s] = len(shape2idx)
for c in sorted(train_colors):
    color2idx[c] = len(color2idx)

label2idx = {l: i for i, l in enumerate(sorted(train_labels))}
idx2label = {i: l for l, i in label2idx.items()}

print(
    f"#shapes(train)={len(shape2idx)-2}, #colors(train)={len(color2idx)-2}, "
    f"#labels={len(label2idx)}"
)


# ----------------------------------------------------------------
# metrics (official definitions)
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def pcwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ----------------------------------------------------------------
# PyTorch Dataset -------------------------------------------------
class SPRFactorDataset(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.lab = hf_split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        tokens = self.seq[idx].split()
        shape_ids = [shape2idx.get(tok[0], shape2idx["<UNK>"]) for tok in tokens]
        color_ids = [color2idx.get(tok[1:], color2idx["<UNK>"]) for tok in tokens]
        return {
            "shape_ids": torch.tensor(shape_ids, dtype=torch.long),
            "color_ids": torch.tensor(color_ids, dtype=torch.long),
            "length": len(tokens),
            "label": label2idx[self.lab[idx]],
            "seq_raw": self.seq[idx],
        }


def collate_fn(batch):
    max_len = max(b["length"] for b in batch)
    bs = len(batch)
    shp = torch.zeros(bs, max_len, dtype=torch.long)
    col = torch.zeros(bs, max_len, dtype=torch.long)
    lengths = torch.zeros(bs, dtype=torch.long)
    labels, raws = [], []
    for i, b in enumerate(batch):
        l = b["length"]
        shp[i, :l] = b["shape_ids"]
        col[i, :l] = b["color_ids"]
        lengths[i] = l
        labels.append(b["label"])
        raws.append(b["seq_raw"])
    return {
        "shape_ids": shp,
        "color_ids": col,
        "lengths": lengths,
        "labels": torch.tensor(labels, dtype=torch.long),
        "seq_raw": raws,
    }


batch_size = 128
train_loader = DataLoader(
    SPRFactorDataset(dsets["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
dev_loader = DataLoader(
    SPRFactorDataset(dsets["dev"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    SPRFactorDataset(dsets["test"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


# ----------------------------------------------------------------
# model -----------------------------------------------------------
class FactorMeanClassifier(nn.Module):
    def __init__(self, n_shapes, n_colors, emb_dim, n_labels):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, emb_dim, padding_idx=0)
        self.color_emb = nn.Embedding(n_colors, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, n_labels)

    def forward(self, shp_ids, col_ids, lengths):
        vec = self.shape_emb(shp_ids) + self.color_emb(col_ids)
        mask = (shp_ids != 0).unsqueeze(-1)
        summed = (vec * mask).sum(dim=1)
        mean = summed / lengths.unsqueeze(1).clamp(min=1).float()
        return self.fc(mean)


model = FactorMeanClassifier(
    len(shape2idx), len(color2idx), emb_dim=64, n_labels=len(label2idx)
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------------------------------------------
# experiment data container --------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ----------------------------------------------------------------
# training / evaluation helpers ----------------------------------
def run_epoch(loader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, total_n = 0.0, 0
    seqs, ys, ps = [], [], []
    for batch in loader:
        # move tensors to device
        batch_t = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
        }
        shp, col, lens, lab = (
            batch_t["shape_ids"],
            batch_t["color_ids"],
            batch_t["lengths"],
            batch_t["labels"],
        )
        logits = model(shp, col, lens)
        loss = criterion(logits, lab)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * shp.size(0)
        total_n += shp.size(0)

        pred = logits.argmax(1).detach().cpu().tolist()
        true = lab.detach().cpu().tolist()
        ys.extend([idx2label[i] for i in true])
        ps.extend([idx2label[i] for i in pred])
        seqs.extend(batch["seq_raw"])  # still on CPU list
    avg_loss = total_loss / total_n
    return avg_loss, cwa(seqs, ys, ps), swa(seqs, ys, ps), pcwa(seqs, ys, ps), ys, ps


# ----------------------------------------------------------------
# main training loop ---------------------------------------------
EPOCHS = 4
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, *_ = run_epoch(train_loader, train=True)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))

    val_loss, val_cwa, val_swa, val_pcwa, _, _ = run_epoch(dev_loader, train=False)
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (epoch, {"CWA": val_cwa, "SWA": val_swa, "PCWA": val_pcwa})
    )

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
        f"CWA {val_cwa:.4f} | SWA {val_swa:.4f} | PCWA {val_pcwa:.4f} "
        f"({time.time()-t0:.1f}s)"
    )

# ----------------------------------------------------------------
# final test evaluation ------------------------------------------
test_loss, test_cwa, test_swa, test_pcwa, ys, ps = run_epoch(test_loader, train=False)
print(
    f"\nTEST | loss {test_loss:.4f} | CWA {test_cwa:.4f} | "
    f"SWA {test_swa:.4f} | PCWA {test_pcwa:.4f}"
)

experiment_data["SPR_BENCH"]["predictions"] = ps
experiment_data["SPR_BENCH"]["ground_truth"] = ys

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
