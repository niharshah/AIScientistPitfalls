import os, pathlib, random, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ------------- working dir & device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------- determinism -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


# ------------- helpers (SPR utilities) -----------------
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


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ------------- load dataset ----------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ------------- vocabularies ----------------------------
shape_set, color_set = set(), set()
for row in spr["train"]:
    for tok in row["sequence"].split():
        if tok:
            shape_set.add(tok[0])
            if len(tok) > 1:
                color_set.add(tok[1])
shape2id = {s: i + 2 for i, s in enumerate(sorted(shape_set))}
shape2id["<pad>"] = 0
shape2id["<unk>"] = 1
color2id = {c: i + 2 for i, c in enumerate(sorted(color_set))}
color2id["<pad>"] = 0
color2id["<unk>"] = 1
label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: l for l, i in label2id.items()}
print(f"Shapes:{len(shape2id)} Colors:{len(color2id)} Classes:{len(label2id)}")


# ------------- torch dataset ---------------------------
def seq_to_idx(seq: str):
    shp, col = [], []
    for tok in seq.strip().split():
        shp.append(shape2id.get(tok[0], shape2id["<unk>"]))
        col.append(
            color2id.get(tok[1], color2id["<unk>"])
            if len(tok) > 1
            else color2id["<pad>"]
        )
    return shp, col


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.data = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        s_idx, c_idx = seq_to_idx(row["sequence"])
        return {
            "shape_idx": torch.tensor(s_idx, dtype=torch.long),
            "color_idx": torch.tensor(c_idx, dtype=torch.long),
            "label": torch.tensor(label2id[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


def collate(batch):
    shapes = [b["shape_idx"] for b in batch]
    colors = [b["color_idx"] for b in batch]
    shp_pad = nn.utils.rnn.pad_sequence(
        shapes, batch_first=True, padding_value=shape2id["<pad>"]
    )
    col_pad = nn.utils.rnn.pad_sequence(
        colors, batch_first=True, padding_value=color2id["<pad>"]
    )
    labels = torch.stack([b["label"] for b in batch])

    raw = [b["raw_seq"] for b in batch]
    seq_len = torch.tensor([len(s) for s in shapes], dtype=torch.float)
    uniq_shape = torch.tensor([count_shape_variety(r) for r in raw], dtype=torch.float)
    uniq_color = torch.tensor([count_color_variety(r) for r in raw], dtype=torch.float)
    shape_div = uniq_shape / seq_len
    color_div = uniq_color / seq_len
    sym = torch.stack([uniq_shape, uniq_color, seq_len, shape_div, color_div], dim=1)
    return {
        "shape_idx": shp_pad,
        "color_idx": col_pad,
        "sym": sym,
        "label": labels,
        "raw_seq": raw,
    }


train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=256, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ------------- model -----------------------------------
class HybridGateModel(nn.Module):
    def __init__(
        self,
        shape_vocab,
        color_vocab,
        emb_dim=64,
        layers=3,
        nhead=4,
        sym_dim=5,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(
            shape_vocab, emb_dim, padding_idx=shape2id["<pad>"]
        )
        self.color_emb = nn.Embedding(
            color_vocab, emb_dim, padding_idx=color2id["<pad>"]
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.sym_proj = nn.Linear(sym_dim, emb_dim)
        self.gate = nn.Linear(sym_dim, emb_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, s_idx, c_idx, sym):
        tok_emb = self.shape_emb(s_idx) + self.color_emb(c_idx)
        mask = s_idx == shape2id["<pad>"]
        enc = self.encoder(tok_emb, src_key_padding_mask=mask)
        # mean pooling (ignoring pads)
        rep = (enc * (~mask).unsqueeze(-1)).sum(1) / (~mask).sum(1, keepdim=True).clamp(
            min=1
        )
        sym_emb = self.sym_proj(sym)
        gate = torch.sigmoid(self.gate(sym))
        fused = rep * gate + sym_emb
        return self.classifier(fused)


# ------------- training utils --------------------------
def run_epoch(model, loader, optim=None):
    training = optim is not None
    model.train() if training else model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_pred, all_gt, all_raw = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["shape_idx"], batch["color_idx"], batch["sym"])
        loss = nn.functional.cross_entropy(logits, batch["label"])
        if training:
            optim.zero_grad()
            loss.backward()
            optim.step()
        with torch.no_grad():
            pred = logits.argmax(-1)
        loss_sum += loss.item() * batch["label"].size(0)
        correct += (pred == batch["label"]).sum().item()
        total += batch["label"].size(0)
        all_pred.extend(pred.cpu().tolist())
        all_gt.extend(batch["label"].cpu().tolist())
        all_raw.extend(batch["raw_seq"])
    acc = correct / total
    swa = shape_weighted_accuracy(all_raw, all_gt, all_pred)
    return loss_sum / total, acc, swa, all_pred


# ------------- experiment container --------------------
experiment_data = {
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

# ------------- instantiate model -----------------------
model = HybridGateModel(len(shape2id), len(color2id), num_classes=len(label2id)).to(
    device
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)

# ------------- training loop ---------------------------
epochs = 6
for ep in range(1, epochs + 1):
    tr_loss, tr_acc, tr_swa, _ = run_epoch(model, train_loader, optimizer)
    val_loss, val_acc, val_swa, val_pred = run_epoch(model, dev_loader, None)
    scheduler.step()
    print(f"Epoch {ep}: validation_loss = {val_loss:.4f}")
    experiment_data["spr_bench"]["metrics"]["train"].append(tr_acc)
    experiment_data["spr_bench"]["metrics"]["val"].append(val_acc)
    experiment_data["spr_bench"]["swa"]["train"].append(tr_swa)
    experiment_data["spr_bench"]["swa"]["val"].append(val_swa)
    experiment_data["spr_bench"]["losses"]["train"].append(tr_loss)
    experiment_data["spr_bench"]["losses"]["val"].append(val_loss)
    experiment_data["spr_bench"]["predictions"][f"val_epoch{ep}"] = val_pred

# ------------- plot & save -----------------------------
plt.figure()
plt.plot(experiment_data["spr_bench"]["swa"]["train"], label="train")
plt.plot(experiment_data["spr_bench"]["swa"]["val"], label="val")
plt.xlabel("Epoch")
plt.ylabel("SWA")
plt.title("Shape-Weighted Accuracy")
plt.legend()
plt.savefig(os.path.join(working_dir, "swa_curve.png"))
plt.close()

# ------------- final test ------------------------------
test_loss, test_acc, test_swa, test_pred = run_epoch(model, test_loader, None)
print(f"TEST: loss={test_loss:.4f} acc={test_acc:.3f} SWA={test_swa:.3f}")
experiment_data["spr_bench"]["predictions"]["test"] = test_pred
experiment_data["spr_bench"]["test_metrics"] = {
    "loss": test_loss,
    "acc": test_acc,
    "swa": test_swa,
}

# ------------- persist artefacts -----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to ./working")
