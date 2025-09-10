import os, pathlib, random, json, time, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ---------------- work dir & device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- reproducibility -------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------- helpers ---------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


def count_shape_variety(seq: str) -> int:
    return len(set(t[0] for t in seq.split() if t))


def count_color_variety(seq: str) -> int:
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ---------------- data paths ------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------------- vocabularies ----------------------
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
label_set = sorted({r["label"] for r in spr["train"]})
label2id = {l: i for i, l in enumerate(label_set)}
print(f"Shapes {len(shape2id)} Colors {len(color2id)} Classes {len(label2id)}")


# ---------------- converters ------------------------
def seq_to_indices(seq):
    s_idx, c_idx = [], []
    for tok in seq.split():
        s_idx.append(shape2id.get(tok[0], shape2id["<unk>"]))
        c_idx.append(
            color2id.get(tok[1], color2id["<unk>"])
            if len(tok) > 1
            else color2id["<pad>"]
        )
    return s_idx, c_idx


# ---------------- torch Dataset ---------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf):
        self.data = hf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        s, c = seq_to_indices(row["sequence"])
        return {
            "shape_idx": torch.tensor(s),
            "color_idx": torch.tensor(c),
            "label": torch.tensor(label2id[row["label"]]),
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

    sv = torch.tensor([count_shape_variety(r) for r in raws], dtype=torch.float)
    cv = torch.tensor([count_color_variety(r) for r in raws], dtype=torch.float)
    ln = torch.tensor(lens, dtype=torch.float)
    sym_feats = torch.stack([sv, cv, ln], dim=1)  # HISTOGRAM-FREE

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


# ---------------- model -----------------------------
class GatedHybrid(nn.Module):
    def __init__(
        self, shp_vocab, col_vocab, sym_dim, num_cls, d_model=64, nhead=8, nlayers=2
    ):
        super().__init__()
        self.sh_emb = nn.Embedding(shp_vocab, d_model, padding_idx=shape2id["<pad>"])
        self.co_emb = nn.Embedding(col_vocab, d_model, padding_idx=color2id["<pad>"])
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.sym_proj = nn.Linear(sym_dim, d_model)
        self.gate = nn.Linear(d_model * 2, d_model, bias=False)
        self.cls = nn.Sequential(nn.ReLU(), nn.Linear(d_model, num_cls))

    def forward(self, sh_idx, co_idx, sym):
        tok_rep = self.sh_emb(sh_idx) + self.co_emb(co_idx)
        mask = sh_idx == shape2id["<pad>"]
        enc = self.encoder(tok_rep, src_key_padding_mask=mask)
        mean = (enc * (~mask).unsqueeze(-1)).sum(1) / (~mask).sum(
            1, keepdim=True
        ).clamp(min=1)
        sym_r = self.sym_proj(sym)
        alpha = torch.sigmoid(self.gate(torch.cat([mean, sym_r], 1)))
        joint = alpha * mean + (1 - alpha) * sym_r
        return self.cls(joint)


sym_dim_total = 3
model = GatedHybrid(len(shape2id), len(color2id), sym_dim_total, len(label2id)).to(
    device
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------------- logging dict ----------------------
experiment_data = {
    "hist_free": {
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


# ---------------- epoch runner ---------------------
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    preds, gts, raws = [], [], []
    for batch in loader:
        bt = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(bt["shape_idx"], bt["color_idx"], bt["sym"])
        loss = criterion(logits, bt["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            p = logits.argmax(-1)
        loss_sum += loss.item() * bt["label"].size(0)
        correct += (p == bt["label"]).sum().item()
        tot += bt["label"].size(0)
        preds.extend(p.cpu().tolist())
        gts.extend(bt["label"].cpu().tolist())
        raws.extend(batch["raw_seq"])
    acc = correct / tot
    avg_loss = loss_sum / tot
    swa = shape_weighted_accuracy(raws, gts, preds)
    return avg_loss, acc, swa, preds


# ---------------- training loop --------------------
num_epochs = 5
for ep in range(1, num_epochs + 1):
    tr_l, tr_a, tr_swa, _ = run_epoch(train_loader, True)
    val_l, val_a, val_swa, val_p = run_epoch(dev_loader, False)
    print(f"Epoch {ep}: val_loss={val_l:.4f} acc={val_a:.3f} swa={val_swa:.3f}")
    exp = experiment_data["hist_free"]["spr_bench"]
    exp["metrics"]["train"].append(tr_a)
    exp["metrics"]["val"].append(val_a)
    exp["swa"]["train"].append(tr_swa)
    exp["swa"]["val"].append(val_swa)
    exp["losses"]["train"].append(tr_l)
    exp["losses"]["val"].append(val_l)
    if ep == num_epochs:
        exp["predictions"]["val"] = val_p

# ---------------- final test -----------------------
test_l, test_a, test_swa, test_p = run_epoch(test_loader, False)
print(f"TEST: loss={test_l:.4f} acc={test_a:.3f} swa={test_swa:.3f}")
exp["predictions"]["test"] = test_p
exp["test_metrics"] = {"loss": test_l, "acc": test_a, "swa": test_swa}

# ---------------- save artefacts -------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to ./working/experiment_data.npy")
