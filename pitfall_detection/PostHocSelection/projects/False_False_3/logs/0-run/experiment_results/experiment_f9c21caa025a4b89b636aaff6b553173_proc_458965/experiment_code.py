import os, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ------------------ mandatory working dir ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------ data loading ------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path(
    os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
if not DATA_PATH.exists():
    raise FileNotFoundError(f"SPR_BENCH not found at {DATA_PATH}")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------------ helper functions ------------------
def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.strip().split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    num = sum(w for w, t, p in zip(weights, y_true, y_pred) if t == p)
    den = max(sum(weights), 1)
    return num / den


# ------------------ vocabularies ------------------
def all_tokens(dset):
    for seq in dset["sequence"]:
        for tok in seq.split():
            yield tok


shapes = set(tok[0] for tok in all_tokens(spr["train"]))
colors = set((tok[1] if len(tok) > 1 else "-") for tok in all_tokens(spr["train"]))
labels = sorted(set(spr["train"]["label"]))

shape2id = {s: i + 1 for i, s in enumerate(sorted(shapes))}
color2id = {c: i + 1 for i, c in enumerate(sorted(colors))}
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# 0 is padding
n_shape, n_color, n_cls = len(shape2id) + 1, len(color2id) + 1, len(labels)


# ------------------ model ------------------
class HybridClassifier(nn.Module):
    def __init__(self, n_shape, n_color, emb_dim=32, sym_dim=3, hidden=64, n_cls=2):
        super().__init__()
        self.shape_emb = nn.EmbeddingBag(n_shape, emb_dim, mode="mean")
        self.color_emb = nn.EmbeddingBag(n_color, emb_dim, mode="mean")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + sym_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, shape_ids, color_ids, offsets, sym_feats):
        sh = self.shape_emb(shape_ids, offsets)
        co = self.color_emb(color_ids, offsets)
        x = torch.cat([sh, co, sym_feats], dim=1)
        return self.mlp(x)


# ------------------ collate ------------------
def collate(batch):
    shape_ids, color_ids, offsets = [], [], [0]
    labels_, sym_feats, seqs = [], [], []
    for ex in batch:
        seq = ex["sequence"]
        tokens = seq.split()
        seqs.append(seq)

        shapes = [tok[0] for tok in tokens]
        colors = [tok[1] if len(tok) > 1 else "-" for tok in tokens]

        shape_ids.extend([shape2id[s] for s in shapes])
        color_ids.extend([color2id[c] for c in colors])
        offsets.append(offsets[-1] + len(tokens))

        labels_.append(label2id[ex["label"]])

        sym_feats.append(
            [
                count_shape_variety(seq) / 10.0,
                count_color_variety(seq) / 10.0,
                len(tokens) / 20.0,
            ]
        )  # simple scaling

    shape_ids = torch.tensor(shape_ids, dtype=torch.long)
    color_ids = torch.tensor(color_ids, dtype=torch.long)
    offsets = torch.tensor(offsets[:-1], dtype=torch.long)
    labels_ = torch.tensor(labels_, dtype=torch.long)
    sym_feats = torch.tensor(sym_feats, dtype=torch.float)

    return (
        shape_ids.to(device),
        color_ids.to(device),
        offsets.to(device),
        sym_feats.to(device),
        labels_.to(device),
        seqs,
    )  # seqs kept on CPU


batch_size = 128
train_loader = DataLoader(
    spr["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    spr["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    spr["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)

# ------------------ experiment tracking ------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------ training setup ------------------
model = HybridClassifier(n_shape, n_color, n_cls=n_cls).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 15


# ------------------ evaluation helper ------------------
def run_eval(loader):
    model.eval()
    ys, ps, seqs_all, loss_sum = [], [], [], 0.0
    with torch.no_grad():
        for sh, co, off, feats, labs, seqs in loader:
            out = model(sh, co, off, feats)
            loss = criterion(out, labs)
            loss_sum += loss.item() * labs.size(0)
            preds = out.argmax(1).cpu().tolist()
            ys.extend([id2label[i] for i in labs.cpu().tolist()])
            ps.extend([id2label[p] for p in preds])
            seqs_all.extend(seqs)
    swa = shape_weighted_accuracy(seqs_all, ys, ps)
    return loss_sum / len(ys), swa, ys, ps


# ------------------ training loop ------------------
for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0
    for sh, co, off, feats, labs, _ in train_loader:
        optimizer.zero_grad()
        out = model(sh, co, off, feats)
        loss = criterion(out, labs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labs.size(0)
    train_loss /= len(spr["train"])

    val_loss, val_swa, _, _ = run_eval(dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, SWA = {val_swa:.4f}")

# ------------------ final test evaluation ------------------
test_loss, test_swa, y_true, y_pred = run_eval(test_loader)
print(f"Test | loss={test_loss:.4f} | SWA={test_swa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
experiment_data["SPR_BENCH"]["metrics"]["test"] = test_swa
experiment_data["SPR_BENCH"]["losses"]["test"] = test_loss

# ------------------ save all ------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {working_dir}")
