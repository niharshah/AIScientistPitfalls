import os, pathlib, random, math, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, f1_score
import matplotlib.pyplot as plt

# ---------- misc ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


# ---------- SPR loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):  # helper
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    dd = DatasetDict()
    dd["train"], dd["dev"], dd["test"] = (
        _ld("train.csv"),
        _ld("dev.csv"),
        _ld("test.csv"),
    )
    return dd


def get_spr() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return load_spr_bench(p)
    print("SPR_BENCH not found -> generating synthetic toy data")

    def synth(n):
        rows = "ABCD"
        data = []
        for i in range(n):
            seq = "".join(random.choices(rows, k=random.randint(5, 15)))
            lbl = int(seq.count("A") % 2 == 0 and seq[-1] in "BC")
            data.append({"id": i, "sequence": seq, "label": lbl})
        return load_dataset(
            "json", data_files={"data": data}, field="data", split="train"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = synth(4000), synth(1000), synth(1000)
    return d


spr = get_spr()

# ---------- vocab ----------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 pad
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 1
max_len = min(120, max(map(len, spr["train"]["sequence"])))


def encode(seq: str):
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
        ids = torch.randperm(len(ds))[:max_items]
        ds = Subset(ds, ids)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


train_loader = lambda: make_loader("train", shuffle=True, max_items=10000)
dev_loader = lambda: make_loader("dev", shuffle=False, max_items=2000)
test_loader = lambda: make_loader("test", shuffle=False)


# ---------- positional encoding ----------
def positional_encoding(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    pos = torch.arange(0, seq_len, device=device).float().unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1,seq,d)


# ---------- No-FFN encoder layer ----------
class NoFFNEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # second sublayer: identity + norm, keep dropout & residual
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # sub-layer 1: MHA
        attn_out = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        # sub-layer 2: identity
        id_out = self.dropout2(src)  # optional dropout on the pass-through
        src = src + id_out  # residual (effectively 1+dropout mask)
        src = self.norm2(src)
        return src


# ---------- Transformer (attention-only) ----------
class CharTransformerNoFFN(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        layers = [NoFFNEncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)]
        self.encoder = nn.ModuleList(layers)
        self.fc = nn.Linear(d_model, 1)
        self.register_buffer(
            "pe", positional_encoding(max_len, d_model, torch.device("cpu"))
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        mask = x == 0  # pad mask
        h = self.emb(x) + self.pe[:, : x.size(1), :].to(x.device)
        for layer in self.encoder:
            h = layer(h, src_key_padding_mask=mask)
        lengths = (~mask).sum(1).clamp(min=1).unsqueeze(1)
        pooled = (h.masked_fill(mask.unsqueeze(2), 0.0).sum(1)) / lengths
        pooled = self.drop(pooled)
        return self.fc(pooled).squeeze(1)


# ---------- experiment dict ----------
experiment_data = {
    "no_ffn": {
        "SPR": {
            "metrics": {"train_MCC": [], "val_MCC": []},
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- training ----------
dropouts = [0.1, 0.3]
best_dev_mcc = -1
best_state = None
criterion = nn.BCEWithLogitsLoss()
epochs = 6
for dp in dropouts:
    print(f"\n=== Dropout={dp} (No-FFN) ===")
    model = CharTransformerNoFFN(
        vocab_size, d_model=128, nhead=4, num_layers=2, dropout=dp
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        # train
        model.train()
        tr_loss, tr_preds, tr_lbls = [], [], []
        for batch in train_loader():
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            optim.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optim.step()
            tr_loss.append(loss.item())
            tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tr_lbls.extend(batch["label"].cpu().numpy())
        train_mcc = matthews_corrcoef(tr_lbls, tr_preds)

        # val
        model.eval()
        val_loss, val_preds, val_lbls = [], [], []
        with torch.no_grad():
            for batch in dev_loader():
                batch = {
                    k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
                }
                logits = model(batch["input_ids"])
                val_loss.append(criterion(logits, batch["label"]).item())
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_lbls.extend(batch["label"].cpu().numpy())
        val_mcc = matthews_corrcoef(val_lbls, val_preds)
        print(
            f"Epoch {ep}: val_loss={np.mean(val_loss):.4f} | "
            f"train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}"
        )

        # log
        exp = experiment_data["no_ffn"]["SPR"]
        exp["metrics"]["train_MCC"].append(train_mcc)
        exp["metrics"]["val_MCC"].append(val_mcc)
        exp["losses"]["train"].append(np.mean(tr_loss))
        exp["losses"]["val"].append(np.mean(val_loss))
        exp["epochs"].append((dp, ep))
        if val_mcc > best_dev_mcc:
            best_dev_mcc = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_dp = dp

# ---------- test ----------
print(f"\nBest dev MCC={best_dev_mcc:.3f} (dropout={best_dp}) -> Testing")
best_model = CharTransformerNoFFN(vocab_size, 128, 4, 2, best_dp).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_preds, test_lbls = [], []
with torch.no_grad():
    for batch in test_loader():
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        logits = best_model(batch["input_ids"])
        test_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        test_lbls.extend(batch["label"].cpu().numpy())
test_mcc = matthews_corrcoef(test_lbls, test_preds)
test_f1 = f1_score(test_lbls, test_preds, average="macro")
print(f"Test MCC={test_mcc:.3f} | Macro-F1={test_f1:.3f}")
exp = experiment_data["no_ffn"]["SPR"]
exp["predictions"] = test_preds
exp["ground_truth"] = test_lbls
exp["test_MCC"] = test_mcc
exp["test_F1"] = test_f1

# ---------- plot & save ----------
plt.figure(figsize=(6, 4))
plt.plot(exp["losses"]["train"], label="train")
plt.plot(exp["losses"]["val"], label="val")
plt.xlabel("update (epochs aggregated)")
plt.ylabel("BCE loss")
plt.legend()
plt.title("Loss curve No-FFN Transformer")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_curve_no_ffn.png"))
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to", os.path.join(working_dir, "experiment_data.npy"))
