import os, pathlib, random, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# --------------- working dir & device -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------- deterministic --------------------------
seed = 13
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --------------- SPR helpers (from given SPR.py) --------
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


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# --------------- load data ------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# --------------- build vocabularies ---------------------
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
num_shapes, len_shapes = len(shape2id), len(shape2id)
num_colors, len_colors = len(color2id), len(color2id)
label_set = sorted({row["label"] for row in spr["train"]})
label2id = {l: i for i, l in enumerate(label_set)}
num_classes = len(label2id)
print(f"Shapes:{len_shapes} Colors:{len_colors} Classes:{num_classes}")


# --------------- Dataset -------------------------------
def seq_to_sc(seq):
    shapes, colors = [], []
    for tok in seq.strip().split():
        shapes.append(shape2id.get(tok[0], shape2id["<unk>"]))
        if len(tok) > 1:
            colors.append(color2id.get(tok[1], color2id["<unk>"]))
        else:
            colors.append(color2id["<pad>"])
    return shapes, colors


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        s_idx, c_idx = seq_to_sc(row["sequence"])
        return {
            "shapes": torch.tensor(s_idx, dtype=torch.long),
            "colors": torch.tensor(c_idx, dtype=torch.long),
            "label": torch.tensor(label2id[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


def collate(batch):
    shapes = [b["shapes"] for b in batch]
    colors = [b["colors"] for b in batch]
    lens = [len(x) for x in shapes]
    pad_shapes = nn.utils.rnn.pad_sequence(
        shapes, batch_first=True, padding_value=shape2id["<pad>"]
    )
    pad_colors = nn.utils.rnn.pad_sequence(
        colors, batch_first=True, padding_value=color2id["<pad>"]
    )
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    # symbolic counts
    shape_cnt = torch.tensor([count_shape_variety(r) for r in raw], dtype=torch.float)
    color_cnt = torch.tensor([count_color_variety(r) for r in raw], dtype=torch.float)
    seq_len = torch.tensor(lens, dtype=torch.float)
    sym = torch.stack([shape_cnt, color_cnt, seq_len], dim=1)
    return {
        "shape_idx": pad_shapes,
        "color_idx": pad_colors,
        "sym": sym,
        "label": labels,
        "raw_seq": raw,
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


# --------------- Model ---------------------------------
class HybridNSModel(nn.Module):
    def __init__(
        self,
        shape_vocab,
        color_vocab,
        emb_dim=32,
        nhead=4,
        nlayers=2,
        sym_dim=3,
        num_classes=10,
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(
            shape_vocab, emb_dim, padding_idx=shape2id["<pad>"]
        )
        self.color_emb = nn.Embedding(
            color_vocab, emb_dim, padding_idx=color2id["<pad>"]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim + sym_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, shape_idx, color_idx, sym_feats):
        x = self.shape_emb(shape_idx) + self.color_emb(color_idx)
        mask = shape_idx == shape2id["<pad>"]
        enc = self.encoder(x, src_key_padding_mask=mask)
        mean_rep = (enc * (~mask).unsqueeze(-1)).sum(1) / (~mask).sum(
            1, keepdim=True
        ).clamp(min=1)
        out = torch.cat([mean_rep, sym_feats], dim=1)
        return self.fc(out)


# --------------- training utilities --------------------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optim=None):
    train = optim is not None
    if train:
        model.train()
    else:
        model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    preds, gts, raws = [], [], []
    for batch in loader:
        shape_idx = batch["shape_idx"].to(device)
        color_idx = batch["color_idx"].to(device)
        sym = batch["sym"].to(device)
        labels = batch["label"].to(device)
        logits = model(shape_idx, color_idx, sym)
        loss = criterion(logits, labels)
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        with torch.no_grad() if not train else torch.enable_grad():
            p = logits.argmax(-1)
        loss_sum += loss.item() * labels.size(0)
        correct += (p == labels).sum().item()
        tot += labels.size(0)
        preds.extend(p.detach().cpu().tolist())
        gts.extend(labels.cpu().tolist())
        raws.extend(batch["raw_seq"])
    acc = correct / tot
    loss_avg = loss_sum / tot
    swa = shape_weighted_accuracy(raws, gts, preds)
    return loss_avg, acc, swa, preds, gts, raws


# --------------- Experiment -----------------------------
num_epochs = 4
lr = 1e-3
model = HybridNSModel(len_shapes, len_colors, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

for epoch in range(1, num_epochs + 1):
    tr_loss, tr_acc, tr_swa, _, _, _ = run_epoch(model, train_loader, optimizer)
    val_loss, val_acc, val_swa, val_pred, val_gt, val_raw = run_epoch(
        model, dev_loader, None
    )
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    experiment_data["spr_bench"]["metrics"]["train"].append(tr_acc)
    experiment_data["spr_bench"]["metrics"]["val"].append(val_acc)
    experiment_data["spr_bench"]["swa"]["train"].append(tr_swa)
    experiment_data["spr_bench"]["swa"]["val"].append(val_swa)
    experiment_data["spr_bench"]["losses"]["train"].append(tr_loss)
    experiment_data["spr_bench"]["losses"]["val"].append(val_loss)

# Plot SWA
plt.figure()
plt.plot(experiment_data["spr_bench"]["swa"]["train"], label="train")
plt.plot(experiment_data["spr_bench"]["swa"]["val"], label="val")
plt.title("Shape-Weighted Accuracy")
plt.xlabel("Epoch")
plt.ylabel("SWA")
plt.legend()
plt.savefig(os.path.join(working_dir, "swa_curve.png"))
plt.close()

# ----------- Final Test Evaluation ----------------------
test_loss, test_acc, test_swa, test_pred, _, _ = run_epoch(model, test_loader, None)
print(f"TEST: loss={test_loss:.4f} acc={test_acc:.3f} SWA={test_swa:.3f}")
experiment_data["spr_bench"]["predictions"]["test"] = test_pred
experiment_data["spr_bench"]["test_metrics"] = {
    "loss": test_loss,
    "acc": test_acc,
    "swa": test_swa,
}

# --------------- Save artefacts -------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to ./working")
