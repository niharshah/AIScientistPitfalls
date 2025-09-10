import os, pathlib, random, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ---------- work dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- seeds for reproducibility ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------- helpers copied / adapted from SPR.py ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
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


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ---------- dataset path ---------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")  # fallback
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- vocabularies ---------------
shape_set, color_set = set(), set()
for row in spr["train"]:
    for tok in row["sequence"].split():
        shape_set.add(tok[0])
        if len(tok) > 1:
            color_set.add(tok[1])
shape2id = {
    "<pad>": 0,
    "<unk>": 1,
    **{s: i + 2 for i, s in enumerate(sorted(shape_set))},
}
color2id = {
    "<pad>": 0,
    "<unk>": 1,
    **{c: i + 2 for i, c in enumerate(sorted(color_set))},
}
label_set = sorted({row["label"] for row in spr["train"]})
label2id = {l: i for i, l in enumerate(label_set)}
print(f"Shapes {len(shape2id)}  Colors {len(color2id)}  Classes {len(label2id)}")


# ---------- converters ---------------
def seq_to_indices(seq):
    s_idx, c_idx = [], []
    for tok in seq.strip().split():
        s_idx.append(shape2id.get(tok[0], shape2id["<unk>"]))
        if len(tok) > 1:
            c_idx.append(color2id.get(tok[1], color2id["<unk>"]))
        else:
            c_idx.append(color2id["<pad>"])
    return s_idx, c_idx


# ---------- torch Dataset -------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        s_idx, c_idx = seq_to_indices(row["sequence"])
        return {
            "shape_idx": torch.tensor(s_idx, dtype=torch.long),
            "color_idx": torch.tensor(c_idx, dtype=torch.long),
            "label": torch.tensor(label2id[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


def collate(batch):
    shapes = [b["shape_idx"] for b in batch]
    colors = [b["color_idx"] for b in batch]
    lens = [len(x) for x in shapes]
    pad_s = nn.utils.rnn.pad_sequence(
        shapes, batch_first=True, padding_value=shape2id["<pad>"]
    )
    pad_c = nn.utils.rnn.pad_sequence(
        colors, batch_first=True, padding_value=color2id["<pad>"]
    )
    labels = torch.stack([b["label"] for b in batch])
    raws = [b["raw_seq"] for b in batch]
    # symbolic scalar feats
    sv = torch.tensor([count_shape_variety(r) for r in raws], dtype=torch.float)
    cv = torch.tensor([count_color_variety(r) for r in raws], dtype=torch.float)
    ln = torch.tensor(lens, dtype=torch.float)
    # histogram features
    sh_hist = torch.zeros(len(batch), len(shape2id), dtype=torch.float)
    co_hist = torch.zeros(len(batch), len(color2id), dtype=torch.float)
    for i, (s_idx, c_idx) in enumerate(zip(shapes, colors)):
        sh_hist[i].scatter_add_(0, s_idx, torch.ones_like(s_idx, dtype=torch.float))
        co_hist[i].scatter_add_(0, c_idx, torch.ones_like(c_idx, dtype=torch.float))
    sh_hist = sh_hist / ln.unsqueeze(1)
    co_hist = co_hist / ln.unsqueeze(1)
    sym_feats = torch.cat(
        [sv.unsqueeze(1), cv.unsqueeze(1), ln.unsqueeze(1), sh_hist, co_hist], dim=1
    )
    return {
        "shape_idx": pad_s,
        "color_idx": pad_c,
        "sym": sym_feats,
        "label": labels,
        "raw_seq": raws,
    }


batch_size = 256
train_loader = DataLoader(
    SPRTorchDataset(spr["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)


# ---------- model: Bag-of-Embeddings (no Transformer) -------------
class BagOfEmbeddings(nn.Module):
    def __init__(self, shp_vocab, col_vocab, sym_dim, num_classes, d_model=64):
        super().__init__()
        self.sh_emb = nn.Embedding(shp_vocab, d_model, padding_idx=shape2id["<pad>"])
        self.co_emb = nn.Embedding(col_vocab, d_model, padding_idx=color2id["<pad>"])
        self.sym_proj = nn.Linear(sym_dim, d_model)
        self.gate = nn.Linear(d_model * 2, d_model, bias=False)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(d_model, num_classes))

    def forward(self, shape_idx, color_idx, sym_feats):
        token_rep = self.sh_emb(shape_idx) + self.co_emb(color_idx)
        mask = shape_idx != shape2id["<pad>"]  # True where not pad
        summed = (token_rep * mask.unsqueeze(-1)).sum(1)
        mean_rep = summed / mask.sum(1, keepdim=True).clamp(min=1)
        sym_rep = self.sym_proj(sym_feats)
        fusion = torch.sigmoid(self.gate(torch.cat([mean_rep, sym_rep], dim=1)))
        joint = fusion * mean_rep + (1 - fusion) * sym_rep
        return self.classifier(joint)


sym_dim_total = 3 + len(shape2id) + len(color2id)
model = BagOfEmbeddings(
    len(shape2id), len(color2id), sym_dim_total, num_classes=len(label2id)
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------- containers for logging ----
experiment_data = {
    "bag_of_emb": {
        "spr_bench": {
            "metrics": {"train": [], "val": []},
            "swa": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": {},
            "ground_truth": {
                "val": [label2id[l] for l in spr["dev"]["label"]],
                "test": [label2id[l] for l in spr["test"]["label"]],
            },
        }
    }
}


def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    preds, gts, raws = [], [], []
    for batch in loader:
        batch_tensors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(
            batch_tensors["shape_idx"], batch_tensors["color_idx"], batch_tensors["sym"]
        )
        loss = criterion(logits, batch_tensors["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            p = logits.argmax(-1)
        loss_sum += loss.item() * batch_tensors["label"].size(0)
        correct += (p == batch_tensors["label"]).sum().item()
        tot += batch_tensors["label"].size(0)
        preds.extend(p.cpu().tolist())
        gts.extend(batch_tensors["label"].cpu().tolist())
        raws.extend(batch["raw_seq"])
    acc = correct / tot
    avg_loss = loss_sum / tot
    swa = shape_weighted_accuracy(raws, gts, preds)
    return avg_loss, acc, swa, preds


# ---------- training loop -------------
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    tr_loss, tr_acc, tr_swa, _ = run_epoch(train_loader, True)
    val_loss, val_acc, val_swa, val_pred = run_epoch(dev_loader, False)
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  val_swa={val_swa:.3f}"
    )
    # log
    edict = experiment_data["bag_of_emb"]["spr_bench"]
    edict["metrics"]["train"].append(tr_acc)
    edict["metrics"]["val"].append(val_acc)
    edict["swa"]["train"].append(tr_swa)
    edict["swa"]["val"].append(val_swa)
    edict["losses"]["train"].append(tr_loss)
    edict["losses"]["val"].append(val_loss)
    if epoch == num_epochs:
        edict["predictions"]["val"] = val_pred

# ---------- test evaluation -----------
test_loss, test_acc, test_swa, test_pred = run_epoch(test_loader, False)
print(f"TEST: loss={test_loss:.4f} acc={test_acc:.3f} SWA={test_swa:.3f}")
edict = experiment_data["bag_of_emb"]["spr_bench"]
edict["predictions"]["test"] = test_pred
edict["test_metrics"] = {"loss": test_loss, "acc": test_acc, "swa": test_swa}

# ---------- save artefacts ------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to ./working")
