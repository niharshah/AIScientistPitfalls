import os, pathlib, random, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

# ---------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load SPR_BENCH (falls back to synthetic if absent) ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


def get_spr() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return load_spr_bench(p)
    print("SPR_BENCH not found; generating synthetic toy data")

    def synth(n):
        rows, shapes = "ABCD"
        data = []
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 15)))
            lbl = int(seq.count("A") % 2 == 0 and seq[-1] in "BC")
            data.append({"id": i, "sequence": seq, "label": lbl})
        return load_dataset(
            "json", data_files={"data": data}, field="data", split="train"
        )

    return DatasetDict({"train": synth(4000), "dev": synth(1000), "test": synth(1000)})


spr = get_spr()

# ---------- vocab & encoding ----------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 pad
itos = {i: ch for ch, i in zip(vocab, range(1, len(vocab) + 1))}
vocab_size = len(stoi) + 1
max_len = min(120, max(map(len, spr["train"]["sequence"])))


def encode(seq: str):
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


# ---------- handcrafted feature extractor ----------
def hand_features(seq: str):
    length = len(seq)
    counts = np.array([seq.count(ch) for ch in vocab], dtype=np.float32)
    if length == 0:
        length = 1
    norm_counts = counts / length
    parity = (counts % 2).astype(np.float32)
    return np.concatenate(
        ([length / max_len], norm_counts, parity)
    )  # dim = 1+2*|vocab|


feat_dim = 1 + 2 * len(vocab)


# ---------- Dataset & DataLoader ----------
class SPRDataset(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.lbl = split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "feats": torch.tensor(hand_features(self.seq[idx]), dtype=torch.float),
            "label": torch.tensor(float(self.lbl[idx]), dtype=torch.float),
        }


def make_loader(name, batch_size=128, shuffle=False, max_items=None):
    ds = SPRDataset(spr[name])
    if max_items and len(ds) > max_items:
        ids = torch.randperm(len(ds))[:max_items]
        ds = Subset(ds, ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


train_loader = lambda: make_loader("train", shuffle=True, max_items=10000)
dev_loader = lambda: make_loader("dev", shuffle=False, max_items=2000)
test_loader = lambda: make_loader("test", shuffle=False)


# ---------- positional encoding ----------
def pos_encode(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    pos = torch.arange(0, seq_len, device=device).float().unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


# ---------- Hybrid Transformer Model ----------
class HybridSPR(nn.Module):
    def __init__(self, vocab_size, feat_dim, d_model=128, nhead=4, layers=2, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.register_buffer("pe", pos_encode(max_len, d_model, torch.device("cpu")))
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(drop)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 64, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, input_ids, feats):
        mask = input_ids == 0
        h = self.emb(input_ids) + self.pe[:, : input_ids.size(1), :].to(
            input_ids.device
        )
        h = self.encoder(h, src_key_padding_mask=mask)
        lengths = (~mask).sum(1).clamp(min=1).unsqueeze(1)
        pooled = (h.masked_fill(mask.unsqueeze(2), 0.0).sum(1)) / lengths
        pooled = self.drop(pooled)
        feat_vec = self.feat_proj(feats)
        out = torch.cat([pooled, feat_vec], dim=1)
        return self.classifier(out).squeeze(1)


# ---------- experiment tracking ----------
experiment_data = {
    "hybrid": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}

criterion = nn.BCEWithLogitsLoss()
epochs = 6
best_mcc = -1

model = HybridSPR(vocab_size, feat_dim, drop=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    # ---- training ----
    model.train()
    tr_losses = []
    tr_preds = []
    tr_lbls = []
    for batch in train_loader():
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["feats"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.item())
        tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        tr_lbls.extend(batch["label"].cpu().numpy())
    train_mcc = matthews_corrcoef(tr_lbls, tr_preds)

    # ---- validation ----
    model.eval()
    val_losses = []
    val_preds = []
    val_lbls = []
    with torch.no_grad():
        for batch in dev_loader():
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["feats"])
            val_losses.append(criterion(logits, batch["label"]).item())
            val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            val_lbls.extend(batch["label"].cpu().numpy())
    val_mcc = matthews_corrcoef(val_lbls, val_preds)
    print(
        f"Epoch {epoch}: validation_loss = {np.mean(val_losses):.4f} | train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}"
    )

    experiment_data["hybrid"]["metrics"]["train_MCC"].append(train_mcc)
    experiment_data["hybrid"]["metrics"]["val_MCC"].append(val_mcc)
    experiment_data["hybrid"]["losses"]["train"].append(np.mean(tr_losses))
    experiment_data["hybrid"]["losses"]["val"].append(np.mean(val_losses))
    experiment_data["hybrid"]["epochs"].append(epoch)

    if val_mcc > best_mcc:
        best_mcc = val_mcc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ---------- test evaluation ----------
print(f"\nBest dev MCC={best_mcc:.3f}. Evaluating on test set...")
best_model = HybridSPR(vocab_size, feat_dim, drop=0.1).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_preds = []
test_lbls = []
with torch.no_grad():
    for batch in test_loader():
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = best_model(batch["input_ids"], batch["feats"])
        test_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        test_lbls.extend(batch["label"].cpu().numpy())
test_mcc = matthews_corrcoef(test_lbls, test_preds)
print(f"Test MCC={test_mcc:.3f}")

experiment_data["hybrid"]["predictions"] = test_preds
experiment_data["hybrid"]["ground_truth"] = test_lbls
experiment_data["hybrid"]["test_MCC"] = test_mcc

# ---------- plot ----------
plt.figure(figsize=(6, 4))
plt.plot(experiment_data["hybrid"]["losses"]["train"], label="train")
plt.plot(experiment_data["hybrid"]["losses"]["val"], label="val")
plt.xlabel("epoch")
plt.ylabel("BCE loss")
plt.legend()
plt.title("Hybrid Loss")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_curve_hybrid.png"))
plt.close()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
