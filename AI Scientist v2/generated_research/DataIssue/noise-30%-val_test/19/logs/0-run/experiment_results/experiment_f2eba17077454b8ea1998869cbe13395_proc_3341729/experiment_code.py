import os, pathlib, random, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, f1_score
import matplotlib.pyplot as plt

# ---------- dirs / device ----------
work_dir = os.path.join(os.getcwd(), "working")
os.makedirs(work_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# ---------- load / build SPR_BENCH ----------
def _load_csv(p, name):
    return load_dataset(
        "csv", data_files=str(p / name), split="train", cache_dir=".cache_dsets"
    )


def load_real_spr():
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            return DatasetDict(
                {s: _load_csv(p, f"{s}.csv") for s in ["train", "dev", "test"]}
            )
    return None


spr = load_real_spr()
if spr is None:  # synthetic fallback
    print("Real SPR_BENCH not found – creating toy data.")

    def synth(n):
        rows = []
        for i in range(n):
            seq = "".join(random.choices("ABCD", k=random.randint(5, 15)))
            lbl = int(seq.count("A") % 2 == 0 and seq[-1] in "BC")
            rows.append({"id": i, "sequence": seq, "label": lbl})
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    spr = DatasetDict(
        {s: synth(m) for s, m in zip(["train", "dev", "test"], [4000, 1000, 1000])}
    )

# ---------- vocab / encoding ----------
all_chars = sorted(set("".join(spr["train"]["sequence"])))
stoi = {ch: i + 1 for i, ch in enumerate(all_chars)}  # 0 = PAD
itos = {i: ch for ch, i in enumerate(["<PAD>"] + all_chars)}
vocab_size = len(stoi) + 1
max_len = min(120, max(map(len, spr["train"]["sequence"])))


def encode(seq):
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


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
        ds = Subset(ds, torch.randperm(len(ds))[:max_items])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


train_loader = lambda: make_loader("train", shuffle=True, max_items=10000)
dev_loader = lambda: make_loader("dev", shuffle=False, max_items=2000)
test_loader = lambda: make_loader("test", shuffle=False)


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


# ---------- Transformer w/ MAX pooling ----------
class CharTransformerMax(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, layers=2, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, 1)
        self.register_buffer(
            "pe", positional_encoding(max_len, d_model, torch.device("cpu"))
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        mask = x == 0
        h = self.emb(x) + self.pe[:, : x.size(1), :].to(x.device)
        h = self.encoder(h, src_key_padding_mask=mask)
        # element-wise max over sequence dim ignoring PADs
        h_masked = h.masked_fill(mask.unsqueeze(2), -1e9)
        pooled = h_masked.max(1).values
        pooled = self.drop(pooled)
        return self.fc(pooled).squeeze(1)


# ---------- experiment container ----------
experiment_data = {
    "max_pool": {
        "SPR_BENCH": {
            "metrics": {"train_MCC": [], "val_MCC": []},
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- training loop ----------
criterion = nn.BCEWithLogitsLoss()
epochs = 6
dropouts = [0.1, 0.3]
best_val = -1
best_state = None
best_dp = None
for dp in dropouts:
    print(f"\n=== Dropout={dp} ===")
    model = CharTransformerMax(vocab_size, drop=dp).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        # train
        model.train()
        tr_loss = []
        tr_pred = []
        tr_lbl = []
        for batch in train_loader():
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optim.step()
            tr_loss.append(loss.item())
            tr_pred.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tr_lbl.extend(batch["label"].cpu().numpy())
        train_mcc = matthews_corrcoef(tr_lbl, tr_pred)
        # val
        model.eval()
        v_loss = []
        v_pred = []
        v_lbl = []
        with torch.no_grad():
            for batch in dev_loader():
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                v_loss.append(criterion(logits, batch["label"]).item())
                v_pred.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                v_lbl.extend(batch["label"].cpu().numpy())
        val_mcc = matthews_corrcoef(v_lbl, v_pred)
        print(
            f"Epoch {ep}: val_loss={np.mean(v_loss):.4f} | train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}"
        )
        # log
        ed = experiment_data["max_pool"]["SPR_BENCH"]
        ed["metrics"]["train_MCC"].append(train_mcc)
        ed["metrics"]["val_MCC"].append(val_mcc)
        ed["losses"]["train"].append(np.mean(tr_loss))
        ed["losses"]["val"].append(np.mean(v_loss))
        ed["epochs"].append((dp, ep))
        # best
        if val_mcc > best_val:
            best_val = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_dp = dp

# ---------- evaluate best on test ----------
print(f"\nBest dev MCC={best_val:.3f} (dropout={best_dp}) – testing.")
best_model = CharTransformerMax(vocab_size, drop=best_dp).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_pred, test_lbl = [], []
with torch.no_grad():
    for batch in test_loader():
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = best_model(batch["input_ids"])
        test_pred.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        test_lbl.extend(batch["label"].cpu().numpy())
test_mcc = matthews_corrcoef(test_lbl, test_pred)
test_f1 = f1_score(test_lbl, test_pred, average="macro")
print(f"Test MCC={test_mcc:.3f} | Test Macro-F1={test_f1:.3f}")

ed = experiment_data["max_pool"]["SPR_BENCH"]
ed["predictions"] = test_pred
ed["ground_truth"] = test_lbl
ed["test_MCC"] = test_mcc
ed["test_F1"] = test_f1

# ---------- plots ----------
plt.figure(figsize=(6, 4))
plt.plot(ed["losses"]["train"], label="train")
plt.plot(ed["losses"]["val"], label="val")
plt.xlabel("update (epochs aggregated)")
plt.ylabel("BCE loss")
plt.title("Loss curve – MAX pool")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(work_dir, "loss_curve_max_pool.png"))
plt.close()

# ---------- save ----------
np.save(os.path.join(work_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(work_dir, "experiment_data.npy"))
