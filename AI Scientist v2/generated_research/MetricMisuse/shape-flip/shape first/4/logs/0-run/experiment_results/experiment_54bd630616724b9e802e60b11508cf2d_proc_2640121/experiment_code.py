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

# ---------- seeds ----------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


# ---------- SPR loading helpers --------
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


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------- dataset path ---------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
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


# ---------- converters -----------------
def seq_to_indices(seq):
    s_idx, c_idx = [], []
    for tok in seq.strip().split():
        s_idx.append(shape2id.get(tok[0], shape2id["<unk>"]))
        if len(tok) > 1:
            c_idx.append(color2id.get(tok[1], color2id["<unk>"]))
        else:
            c_idx.append(color2id["<pad>"])
    return s_idx, c_idx


# ---------- torch Dataset --------------
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
    labels = torch.stack([b["label"] for b in batch])
    raws = [b["raw_seq"] for b in batch]
    pad_s = nn.utils.rnn.pad_sequence(
        shapes, batch_first=True, padding_value=shape2id["<pad>"]
    )
    pad_c = nn.utils.rnn.pad_sequence(
        colors, batch_first=True, padding_value=color2id["<pad>"]
    )
    return {"shape_idx": pad_s, "color_idx": pad_c, "label": labels, "raw_seq": raws}


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


# ---------- Token-only model -----------
class TokenOnlyModel(nn.Module):
    def __init__(
        self, shp_vocab, col_vocab, num_classes, d_model=64, nhead=8, nlayers=2
    ):
        super().__init__()
        self.sh_emb = nn.Embedding(shp_vocab, d_model, padding_idx=shape2id["<pad>"])
        self.co_emb = nn.Embedding(col_vocab, d_model, padding_idx=color2id["<pad>"])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(d_model, num_classes))

    def forward(self, shape_idx, color_idx):
        token_rep = self.sh_emb(shape_idx) + self.co_emb(color_idx)
        mask = shape_idx == shape2id["<pad>"]
        enc_out = self.encoder(token_rep, src_key_padding_mask=mask)
        summed = (enc_out * (~mask).unsqueeze(-1)).sum(1)
        mean_rep = summed / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.classifier(mean_rep)


model = TokenOnlyModel(len(shape2id), len(color2id), num_classes=len(label2id)).to(
    device
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------- logging container ----------
experiment_data = {
    "token_only": {
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


# ---------- epoch runner --------------
def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    preds, gts, raws = [], [], []
    for batch in loader:
        shape_idx = batch["shape_idx"].to(device)
        color_idx = batch["color_idx"].to(device)
        labels = batch["label"].to(device)
        logits = model(shape_idx, color_idx)
        loss = criterion(logits, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            p = logits.argmax(-1)
        loss_sum += loss.item() * labels.size(0)
        correct += (p == labels).sum().item()
        tot += labels.size(0)
        preds.extend(p.cpu().tolist())
        gts.extend(labels.cpu().tolist())
        raws.extend(batch["raw_seq"])
    return (
        loss_sum / tot,
        correct / tot,
        shape_weighted_accuracy(raws, gts, preds),
        preds,
    )


# ---------- training loop -------------
num_epochs = 5
for ep in range(1, num_epochs + 1):
    tr_loss, tr_acc, tr_swa, _ = run_epoch(train_loader, True)
    val_loss, val_acc, val_swa, val_p = run_epoch(dev_loader, False)
    print(f"Epoch {ep}: val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
    dslog = experiment_data["token_only"]["spr_bench"]
    dslog["metrics"]["train"].append(tr_acc)
    dslog["metrics"]["val"].append(val_acc)
    dslog["swa"]["train"].append(tr_swa)
    dslog["swa"]["val"].append(val_swa)
    dslog["losses"]["train"].append(tr_loss)
    dslog["losses"]["val"].append(val_loss)
    if ep == num_epochs:
        dslog["predictions"]["val"] = val_p

# ---------- test evaluation -----------
test_loss, test_acc, test_swa, test_p = run_epoch(test_loader, False)
print(f"TEST: loss={test_loss:.4f}  acc={test_acc:.3f}  SWA={test_swa:.3f}")
experiment_data["token_only"]["spr_bench"]["predictions"]["test"] = test_p
experiment_data["token_only"]["spr_bench"]["test_metrics"] = {
    "loss": test_loss,
    "acc": test_acc,
    "swa": test_swa,
}

# ---------- save artefacts ------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to ./working/experiment_data.npy")
