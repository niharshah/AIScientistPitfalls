import os, pathlib, random, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# set up working dir and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
#  SPR loader  (same logic, synthetic fallback)
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = _ld("train.csv"), _ld("dev.csv"), _ld("test.csv")
    return d


def get_spr() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return load_spr_bench(p)
    print("SPR_BENCH not found – generating small synthetic benchmark")

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

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = synth(4000), synth(1000), synth(1000)
    return d


spr = get_spr()

# ------------------------------------------------------------
# vocabulary + encode util
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 = PAD
itos = ["<PAD>"] + vocab
vocab_size = len(itos)
max_len = min(120, max(map(len, spr["train"]["sequence"])))


def encode_seq(seq: str):
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


# ------------------------------------------------------------
# feature extraction (cheap symbolic features)
def extract_features(seq: str) -> np.ndarray:
    L = len(seq)
    f = [L / max_len, float(L % 2 == 0)]  # length + parity
    counts = [seq.count(ch) / L for ch in vocab]  # normalized counts
    parities = [float(seq.count(ch) % 2 == 0) for ch in vocab]  # per symbol parity
    first_id = stoi.get(seq[0], 0) / vocab_size
    last_id = stoi.get(seq[-1], 0) / vocab_size
    return np.array(f + counts + parities + [first_id, last_id], dtype=np.float32)


feat_dim = len(extract_features(spr["train"]["sequence"][0]))
print("Feature dimension:", feat_dim)


# ------------------------------------------------------------
# Dataset / DataLoader
class SPRDataset(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.y = split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_seq(self.seq[idx]), dtype=torch.long),
            "features": torch.tensor(extract_features(self.seq[idx])),
            "label": torch.tensor(float(self.y[idx]), dtype=torch.float),
        }


def make_loader(name, batch=128, shuffle=False, max_items=None):
    ds = SPRDataset(spr[name])
    if max_items and len(ds) > max_items:
        ids = torch.randperm(len(ds))[:max_items]
        ds = Subset(ds, ids)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


train_loader = lambda: make_loader("train", shuffle=True, max_items=12000)
dev_loader = lambda: make_loader("dev", shuffle=False, max_items=2500)
test_loader = lambda: make_loader("test", shuffle=False)


# ------------------------------------------------------------
# positional encodings
def sinusoid_pe(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    pos = torch.arange(seq_len, device=device).float().unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


# ------------------------------------------------------------
# Hybrid Transformer + feature MLP
class HybridSPR(nn.Module):
    def __init__(
        self, vocab_size, d_model=128, nhead=4, layers=2, feat_dim=20, drop=0.1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.register_buffer("pe", sinusoid_pe(max_len, d_model, torch.device("cpu")))
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(drop)
        )
        self.classifier = nn.Linear(d_model + 64, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, ids, feat):
        pad_mask = ids == 0
        h = self.emb(ids) + self.pe[:, : ids.size(1), :].to(ids.device)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        lens = (~pad_mask).sum(1).unsqueeze(1).clamp(min=1)
        pooled = (h.masked_fill(pad_mask.unsqueeze(2), 0).sum(1)) / lens
        pooled = self.drop(pooled)
        feat_vec = self.feat_mlp(feat)
        out = torch.cat([pooled, feat_vec], dim=1)
        return self.classifier(out).squeeze(1)


# ------------------------------------------------------------
# experiment bookkeeping
experiment_data = {
    "hybrid": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
model = HybridSPR(vocab_size, feat_dim=feat_dim, drop=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 8
best_val, best_state = -1, None

for epoch in range(1, epochs + 1):
    # --- training
    model.train()
    tr_losses, tr_preds, tr_labels = [], [], []
    for batch in train_loader():
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["features"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.item())
        tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        tr_labels.extend(batch["label"].cpu().numpy())
    train_mcc = matthews_corrcoef(tr_labels, tr_preds)

    # --- validation
    model.eval()
    val_losses, val_preds, val_labels = [], [], []
    with torch.no_grad():
        for batch in dev_loader():
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["features"])
            val_losses.append(criterion(logits, batch["label"]).item())
            val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            val_labels.extend(batch["label"].cpu().numpy())
    val_mcc = matthews_corrcoef(val_labels, val_preds)
    print(
        f"Epoch {epoch}: validation_loss = {np.mean(val_losses):.4f} | train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}"
    )

    # store
    experiment_data["hybrid"]["metrics"]["train_MCC"].append(train_mcc)
    experiment_data["hybrid"]["metrics"]["val_MCC"].append(val_mcc)
    experiment_data["hybrid"]["losses"]["train"].append(np.mean(tr_losses))
    experiment_data["hybrid"]["losses"]["val"].append(np.mean(val_losses))
    experiment_data["hybrid"]["epochs"].append(epoch)

    if val_mcc > best_val:
        best_val = val_mcc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ------------------------------------------------------------
#   test evaluation with best epoch
print(f"\nBest dev MCC={best_val:.3f}. Evaluating on test set...")
best_model = HybridSPR(vocab_size, feat_dim=feat_dim, drop=0.1).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader():
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = best_model(batch["input_ids"], batch["features"])
        test_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        test_labels.extend(batch["label"].cpu().numpy())
test_mcc = matthews_corrcoef(test_labels, test_preds)
print(f"Test MCC={test_mcc:.3f}")

experiment_data["hybrid"]["predictions"] = test_preds
experiment_data["hybrid"]["ground_truth"] = test_labels
experiment_data["hybrid"]["test_MCC"] = test_mcc

# ------------------------------------------------------------
# plotting
plt.figure(figsize=(6, 4))
plt.plot(experiment_data["hybrid"]["losses"]["train"], label="train")
plt.plot(experiment_data["hybrid"]["losses"]["val"], label="val")
plt.xlabel("epochs")
plt.ylabel("BCE loss")
plt.legend()
plt.title("Hybrid loss")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_hybrid.png"))
plt.close()

# ------------------------------------------------------------
# save data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data at", os.path.join(working_dir, "experiment_data.npy"))
