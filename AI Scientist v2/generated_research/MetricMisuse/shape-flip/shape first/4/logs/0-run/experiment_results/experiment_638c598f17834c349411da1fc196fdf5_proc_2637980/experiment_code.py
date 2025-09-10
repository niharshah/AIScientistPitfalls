import os, pathlib, random, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ---------- mandatory boilerplate ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- reproducibility ---------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------- SPR utilities -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if w else 0.0


# ---------- dataset & vocab ----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
ds = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in ds.items()})

shape_set, color_set = set(), set()
for ex in ds["train"]:
    for tok in ex["sequence"].split():
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
label2id = {l: i for i, l in enumerate(sorted({ex["label"] for ex in ds["train"]}))}
id2label = {v: k for k, v in label2id.items()}


def seq_to_idx(seq):
    s_idx, c_idx = [], []
    for tok in seq.split():
        s_idx.append(shape2id.get(tok[0], 1))
        c_idx.append(color2id.get(tok[1], 1) if len(tok) > 1 else color2id["<pad>"])
    return s_idx, c_idx


def histogram(seq):
    h = np.zeros(len(shape_set) + len(color_set), dtype=np.float32)
    for tok in seq.split():
        h_idx = sorted(shape_set).index(tok[0])
        h[h_idx] += 1
        if len(tok) > 1:
            c_idx = sorted(color_set).index(tok[1]) + len(shape_set)
            h[c_idx] += 1
    return h / len(seq.split())  # normalised


class SPRTorch(Dataset):
    def __init__(self, split):
        self.d = split

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        row = self.d[i]
        s, c = seq_to_idx(row["sequence"])
        return {
            "shape": torch.tensor(s),
            "color": torch.tensor(c),
            "hist": torch.tensor(histogram(row["sequence"])),
            "sym": torch.tensor(
                [
                    count_shape_variety(row["sequence"]),
                    len(set(tok[1] for tok in row["sequence"].split() if len(tok) > 1)),
                    len(row["sequence"].split()),
                ],
                dtype=torch.float,
            ),
            "label": torch.tensor(label2id[row["label"]]),
            "raw": row["sequence"],
        }


def collate(batch):
    shp = [b["shape"] for b in batch]
    col = [b["color"] for b in batch]
    shp_pad = nn.utils.rnn.pad_sequence(
        shp, batch_first=True, padding_value=shape2id["<pad>"]
    )
    col_pad = nn.utils.rnn.pad_sequence(
        col, batch_first=True, padding_value=color2id["<pad>"]
    )
    labels = torch.stack([b["label"] for b in batch])
    sym = torch.stack([b["sym"] for b in batch])
    hist = torch.stack([b["hist"] for b in batch])
    raws = [b["raw"] for b in batch]
    return {
        "shape": shp_pad,
        "color": col_pad,
        "sym": sym,
        "hist": hist,
        "label": labels,
        "raw": raws,
    }


batch_size = 256
train_loader = DataLoader(
    SPRTorch(ds["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(ds["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(ds["test"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------- model ---------------------------------------
class NeuroSymRuleNet(nn.Module):
    def __init__(self, shape_vocab, color_vocab, num_cls, emb=32, nhead=4, layers=2):
        super().__init__()
        self.s_emb = nn.Embedding(shape_vocab, emb, padding_idx=shape2id["<pad>"])
        self.c_emb = nn.Embedding(color_vocab, emb, padding_idx=color2id["<pad>"])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        # symbolic branch
        self.sym_mlp = nn.Sequential(
            nn.Linear(len(shape_set) + len(color_set), 64),
            nn.ReLU(),
            nn.Linear(64, num_cls),
        )
        # gate network
        self.gate = nn.Sequential(
            nn.Linear(len(shape_set) + len(color_set), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        # small head for neural branch
        self.neural_head = nn.Linear(emb, num_cls)

    def forward(self, shape_idx, color_idx, hist):
        # neural branch
        x = self.s_emb(shape_idx) + self.c_emb(color_idx)
        mask = shape_idx == shape2id["<pad>"]
        enc = self.encoder(x, src_key_padding_mask=mask)
        mean = (enc * (~mask).unsqueeze(-1)).sum(1) / (~mask).sum(
            1, keepdim=True
        ).clamp(min=1)
        n_logits = self.neural_head(mean)
        # symbolic branch
        s_logits = self.sym_mlp(hist)
        g = self.gate(hist)  # (B,1)
        logits = g * s_logits + (1 - g) * n_logits
        return logits, s_logits


# ---------- training utils ------------------------------
def run_epoch(model, loader, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tot, correct, lsum = 0, 0, 0.0
    preds = []
    gts = []
    raws = []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits, _ = model(batch["shape"], batch["color"], batch["hist"])
        loss = loss_fn(logits, batch["label"])
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        with torch.no_grad():
            p = logits.argmax(-1)
        lsum += loss.item() * len(p)
        correct += (p == batch["label"]).sum().item()
        tot += len(p)
        preds.extend(p.cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
        raws.extend(batch["raw"])
    acc = correct / tot
    loss_avg = lsum / tot
    swa = shape_weighted_accuracy(raws, gts, preds)
    return loss_avg, acc, swa, preds


# ---------- experiment ----------------------------------
exp = {
    "spr_bench": {
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": []},
        "swa": {"train": [], "val": []},
        "predictions": {},
    }
}

model = NeuroSymRuleNet(len(shape2id), len(color2id), len(label2id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 5

for ep in range(1, epochs + 1):
    tr_loss, tr_acc, tr_swa, _ = run_epoch(model, train_loader, optimizer)
    val_loss, val_acc, val_swa, val_pred = run_epoch(model, dev_loader)
    print(f"Epoch {ep}: validation_loss = {val_loss:.4f}")
    exp["spr_bench"]["losses"]["train"].append(tr_loss)
    exp["spr_bench"]["losses"]["val"].append(val_loss)
    exp["spr_bench"]["metrics"]["train"].append(tr_acc)
    exp["spr_bench"]["metrics"]["val"].append(val_acc)
    exp["spr_bench"]["swa"]["train"].append(tr_swa)
    exp["spr_bench"]["swa"]["val"].append(val_swa)
    exp["spr_bench"]["predictions"][f"val_ep{ep}"] = val_pred

# ---------- test evaluation -----------------------------
test_loss, test_acc, test_swa, test_pred = run_epoch(model, test_loader)
print(f"TEST: loss={test_loss:.4f} acc={test_acc:.3f} SWA={test_swa:.3f}")
exp["spr_bench"]["test"] = {
    "loss": test_loss,
    "acc": test_acc,
    "swa": test_swa,
    "pred": test_pred,
}

# ---------- save + plot --------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), exp)
plt.figure()
plt.plot(exp["spr_bench"]["swa"]["train"], label="train")
plt.plot(exp["spr_bench"]["swa"]["val"], label="val")
plt.xlabel("epoch")
plt.ylabel("SWA")
plt.legend()
plt.title("Shape-Weighted Accuracy")
plt.savefig(os.path.join(working_dir, "swa_curve_v2.png"))
plt.close()
print("artefacts saved in ./working")
