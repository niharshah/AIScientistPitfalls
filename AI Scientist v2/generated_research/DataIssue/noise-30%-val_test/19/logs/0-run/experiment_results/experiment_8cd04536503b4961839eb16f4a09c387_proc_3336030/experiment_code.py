import os, pathlib, random, math, time, json

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, f1_score
import matplotlib.pyplot as plt

# ---------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------- data loader ---------------
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
    # synthetic fallback
    print("SPR_BENCH not found, generating synthetic toy data")

    def synth(n):
        rows, shapes = "ABCD"
        data = [
            {
                "id": i,
                "sequence": "".join(random.choices(shapes, k=random.randint(5, 15))),
                "label": random.randint(0, 1),
            }
            for i in range(n)
        ]
        return load_dataset(
            "json", data_files={"data": data}, field="data", split="train"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = synth(4000), synth(1000), synth(1000)
    return d


spr = get_spr()

# ----------- vocab / encoding ------------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 = PAD
vocab_size = len(stoi) + 1
max_len = min(120, max(map(len, spr["train"]["sequence"])))


def encode_seq(seq: str):
    ids = [stoi[ch] for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


def symbol_features(seq: str):
    # counts and parity per symbol + scaled length
    length = len(seq) / max_len
    counts = [seq.count(ch) for ch in vocab]
    parity = [c % 2 for c in counts]
    return counts + parity + [length]


feat_dim = 2 * len(vocab) + 1


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(encode_seq(seq), dtype=torch.long),
            "sym_feat": torch.tensor(symbol_features(seq), dtype=torch.float),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.float),
        }


def make_loader(name, bs=128, shuffle=False, max_items=None):
    ds = SPRDataset(spr[name])
    if max_items and len(ds) > max_items:
        idx = torch.randperm(len(ds))[:max_items]
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


train_loader = lambda: make_loader("train", shuffle=True, max_items=10000)
dev_loader = lambda: make_loader("dev", shuffle=False, max_items=2000)
test_loader = lambda: make_loader("test", shuffle=False)


# --------- positional encoding ----------
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


# --------- Hybrid model -----------------
class HybridSPR(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        nlayers=2,
        feat_dim=feat_dim,
        dropout=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.register_buffer(
            "pe", positional_encoding(max_len, d_model, torch.device("cpu"))
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(dropout)
        )
        self.out = nn.Linear(d_model + 64, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids, feats):
        mask = ids == 0
        h = self.emb(ids) + self.pe[:, : ids.size(1), :].to(ids.device)
        h = self.encoder(h, src_key_padding_mask=mask)
        lengths = (~mask).sum(1).clamp(min=1).unsqueeze(1)
        pooled = (h.masked_fill(mask.unsqueeze(2), 0).sum(1)) / lengths
        pooled = self.drop(pooled)
        feat_h = self.feat_mlp(feats)
        concat = torch.cat([pooled, feat_h], dim=1)
        return self.out(concat).squeeze(1)


# --------- experiment tracker -----------
experiment_data = {
    "hybrid": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------- training -----------------
def train_eval(dropout=0.1, lr=1e-3, epochs=6):
    model = HybridSPR(vocab_size, dropout=dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    best_mcc, best_state = -1, None
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss, tr_pred, tr_lbl = [], [], []
        for batch in train_loader():
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optim.zero_grad()
            logits = model(batch["input_ids"], batch["sym_feat"])
            loss = crit(logits, batch["label"])
            loss.backward()
            optim.step()
            tr_loss.append(loss.item())
            tr_pred.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tr_lbl.extend(batch["label"].cpu().numpy())
        train_mcc = matthews_corrcoef(tr_lbl, tr_pred)
        # ---- validate ----
        model.eval()
        val_loss, val_pred, val_lbl = [], [], []
        with torch.no_grad():
            for batch in dev_loader():
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"], batch["sym_feat"])
                val_loss.append(crit(logits, batch["label"]).item())
                val_pred.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_lbl.extend(batch["label"].cpu().numpy())
        val_mcc = matthews_corrcoef(val_lbl, val_pred)
        print(
            f"Epoch {ep}: validation_loss = {np.mean(val_loss):.4f} | "
            f"train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}"
        )
        # store
        experiment_data["hybrid"]["metrics"]["train_MCC"].append(train_mcc)
        experiment_data["hybrid"]["metrics"]["val_MCC"].append(val_mcc)
        experiment_data["hybrid"]["losses"]["train"].append(np.mean(tr_loss))
        experiment_data["hybrid"]["losses"]["val"].append(np.mean(val_loss))
        experiment_data["hybrid"]["epochs"].append(ep)
        # best
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best_mcc, model.dropout if hasattr(model, "dropout") else dropout


best_state, best_dev_mcc, _ = train_eval(dropout=0.1)

# --------------- test --------------------
best_model = HybridSPR(vocab_size, dropout=0.1).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_pred, test_lbl = [], []
with torch.no_grad():
    for batch in test_loader():
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = best_model(batch["input_ids"], batch["sym_feat"])
        test_pred.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        test_lbl.extend(batch["label"].cpu().numpy())
test_mcc = matthews_corrcoef(test_lbl, test_pred)
test_f1 = f1_score(test_lbl, test_pred, average="macro")
print(
    f"\nBest dev MCC={best_dev_mcc:.3f}. Test MCC={test_mcc:.3f} | Test Macro-F1={test_f1:.3f}"
)

experiment_data["hybrid"]["predictions"] = test_pred
experiment_data["hybrid"]["ground_truth"] = test_lbl
experiment_data["hybrid"]["test_MCC"] = test_mcc
experiment_data["hybrid"]["test_F1"] = test_f1

# --------------- plot --------------------
plt.figure(figsize=(6, 4))
plt.plot(experiment_data["hybrid"]["losses"]["train"], label="train")
plt.plot(experiment_data["hybrid"]["losses"]["val"], label="val")
plt.xlabel("Epoch")
plt.ylabel("BCE loss")
plt.legend()
plt.title("Hybrid SPR loss curve")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_curve_hybrid.png"))
plt.close()

# ---------- save data --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
