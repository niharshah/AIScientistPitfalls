# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# No-LayerNorm ablation study for SPR – single-file script
import os, pathlib, random, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, f1_score
import matplotlib.pyplot as plt

# ---------- paths / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- data loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    out = DatasetDict()
    out["train"], out["dev"], out["test"] = (
        _ld("train.csv"),
        _ld("dev.csv"),
        _ld("test.csv"),
    )
    return out


def get_spr() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return load_spr_bench(p)

    # synthetic fallback
    print("SPR_BENCH not found — generating toy data")

    def _synth(n):
        rows, shapes = "ABCD"
        data = []
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 15)))
            lbl = int(seq.count("A") % 2 == 0 and seq[-1] in "BC")
            data.append({"id": i, "sequence": seq, "label": lbl})
        return load_dataset(
            "json", data_files={"data": data}, field="data", split="train"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = _synth(4000), _synth(1000), _synth(1000)
    return d


spr = get_spr()

# ---------- vocabulary ----------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 = PAD
itos = {i: ch for ch, i in enumerate(["<PAD>"] + vocab)}
vocab_size = len(stoi) + 1
max_len = min(120, max(map(len, spr["train"]["sequence"])))


def encode(seq: str):
    out = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return out + [0] * (max_len - len(out))


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.float),
        }


def make_loader(name, bs=128, shuffle=False, max_items=None):
    ds = SPRDataset(spr[name])
    if max_items and len(ds) > max_items:
        idx = torch.randperm(len(ds))[:max_items]
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


train_loader = lambda: make_loader("train", shuffle=True, max_items=10000)
dev_loader = lambda: make_loader("dev", max_items=2000)
test_loader = lambda: make_loader("test")


# ---------- positional enc ----------
def positional_encoding(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    pos = torch.arange(0, seq_len, device=device).float().unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


# ---------- No-LayerNorm encoder layer ----------
class TransformerEncoderLayerNoLN(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self-attention block without LayerNorm
        attn_out, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        src = src + self.dropout(attn_out)

        # Feed-forward block without LayerNorm
        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(src))))
        src = src + self.dropout(ff)
        return src


# ---------- Char transformer w/o LayerNorm ----------
class CharTransformerNoLN(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        layer = TransformerEncoderLayerNoLN(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=None)
        self.fc = nn.Linear(d_model, 1)
        self.register_buffer(
            "pe", positional_encoding(max_len, d_model, torch.device("cpu"))
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        pad_mask = x == 0
        h = self.emb(x) + self.pe[:, : x.size(1), :].to(x.device)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        lengths = (~pad_mask).sum(1).clamp(min=1).unsqueeze(1)
        pooled = (h.masked_fill(pad_mask.unsqueeze(2), 0).sum(1)) / lengths
        pooled = self.drop(pooled)
        return self.fc(pooled).squeeze(1)


# ---------- experiment bookkeeping ----------
experiment_data = {
    "no_layernorm": {
        "SPR": {
            "metrics": {"train_MCC": [], "val_MCC": []},
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}

criterion = nn.BCEWithLogitsLoss()
dropouts = [0.1, 0.3]
epochs = 6
best_dev_mcc, best_state, best_dp = -1, None, None

# ---------- training loop ----------
for dp in dropouts:
    print(f"\n=== Dropout {dp} ===")
    model = CharTransformerNoLN(
        vocab_size, d_model=128, nhead=4, num_layers=2, dropout=dp
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        # train
        model.train()
        tr_losses, tr_preds, tr_labels = [], [], []
        for batch in train_loader():
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optim.step()
            tr_losses.append(loss.item())
            tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tr_labels.extend(batch["label"].cpu().numpy())
        train_mcc = matthews_corrcoef(tr_labels, tr_preds)

        # validation
        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for batch in dev_loader():
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                val_losses.append(criterion(logits, batch["label"]).item())
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_labels.extend(batch["label"].cpu().numpy())
        val_mcc = matthews_corrcoef(val_labels, val_preds)

        print(
            f"Epoch {ep}: val_loss={np.mean(val_losses):.4f} | train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}"
        )

        # log
        e = experiment_data["no_layernorm"]["SPR"]
        e["metrics"]["train_MCC"].append(train_mcc)
        e["metrics"]["val_MCC"].append(val_mcc)
        e["losses"]["train"].append(np.mean(tr_losses))
        e["losses"]["val"].append(np.mean(val_losses))
        e["epochs"].append((dp, ep))

        if val_mcc > best_dev_mcc:
            best_dev_mcc, best_dp = val_mcc, dp
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

# ---------- test ----------
print(f"\nBest dev MCC={best_dev_mcc:.3f} (dropout={best_dp}) — evaluating on test")
best_model = CharTransformerNoLN(
    vocab_size, d_model=128, nhead=4, num_layers=2, dropout=best_dp
).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader():
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = best_model(batch["input_ids"])
        test_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        test_labels.extend(batch["label"].cpu().numpy())
test_mcc = matthews_corrcoef(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average="macro")
print(f"Test MCC={test_mcc:.3f} | Test macro-F1={test_f1:.3f}")

# store predictions
e = experiment_data["no_layernorm"]["SPR"]
e["predictions"] = test_preds
e["ground_truth"] = test_labels
e["test_MCC"] = test_mcc
e["test_F1"] = test_f1

# ---------- plots ----------
plt.figure(figsize=(6, 4))
plt.plot(e["losses"]["train"], label="train")
plt.plot(e["losses"]["val"], label="val")
plt.xlabel("update (epochs aggregated)")
plt.ylabel("BCE loss")
plt.legend()
plt.title("Loss curve – No-LayerNorm Transformer")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_curve_no_layernorm.png"))
plt.close()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
