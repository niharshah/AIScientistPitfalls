import os, pathlib, random, time, math
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, f1_score
import matplotlib.pyplot as plt

# ----- working dir -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset helpers ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


def get_dataset() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loaded real SPR_BENCH from", p)
            return load_spr_bench(p)

    print("SPR_BENCH not found – using synthetic toy data")

    def toy_rows(n):
        rows, syms = [], "ABCD"
        for i in range(n):
            seq = "".join(random.choices(syms, k=random.randint(5, 20)))
            rows.append(
                {"id": i, "sequence": seq, "label": int(seq.count("A") % 2 == 0)}
            )
        return rows

    def to_ds(rows):
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        to_ds(toy_rows(3000)),
        to_ds(toy_rows(600)),
        to_ds(toy_rows(600)),
    )
    return d


spr = get_dataset()

# ---------- vocab & encoding ----------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 2 for i, ch in enumerate(vocab)}  # 0: PAD, 1: UNK
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 2
max_len = min(120, max(map(len, spr["train"]["sequence"])))  # cap to 120 tokens


def encode(seq: str):
    ids = [stoi.get(ch, 1) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.lab = hf_split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(float(self.lab[idx]), dtype=torch.float),
        }


def make_loader(split: str, bs: int = 128, shuffle=False):
    return DataLoader(
        SPRDataset(spr[split]), batch_size=bs, shuffle=shuffle, num_workers=0
    )


train_loader = make_loader("train", shuffle=True)
dev_loader = make_loader("dev")
test_loader = make_loader("test")


# ---------- model ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self, vocab_sz, emb_dim=64, nhead=4, nlayer=2, hidden=128, dropout=0.3
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=hidden,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(emb_dim, 1)

    def forward(self, ids):
        x = self.emb(ids)  # (B,L,E)
        x = self.pos(x)
        mask = ids == 0  # pad mask
        z = self.transformer(x, src_key_padding_mask=mask)
        pooled = z.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(
            1, keepdim=True
        ).clamp(min=1)
        return self.cls(self.drop(pooled)).squeeze(1)


model = TransformerClassifier(vocab_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# ---------- experiment tracking ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    all_logits, all_labels, losses = [], [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        if train:
            loss = criterion(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = criterion(logits, batch["label"])
        losses.append(loss.item())
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["label"].cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    mcc = matthews_corrcoef(labels, preds)
    return np.mean(losses), mcc, preds, labels


epochs = 8
for epoch in range(1, epochs + 1):
    t0 = time.time()
    train_loss, train_mcc, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_mcc, _, _ = run_epoch(dev_loader, train=False)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_MCC"].append(train_mcc)
    experiment_data["SPR_BENCH"]["metrics"]["val_MCC"].append(val_mcc)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f} "
        f"[{time.time()-t0:.1f}s]"
    )

# ---------- final test evaluation ----------
test_loss, test_mcc, test_preds, test_labels = run_epoch(test_loader, train=False)
print(f"\nTest set: loss={test_loss:.4f}  MCC={test_mcc:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels
experiment_data["SPR_BENCH"]["test_MCC"] = test_mcc

# ---------- save everything ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)

# quick plot for visual sanity
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["losses"]["train"],
    label="train",
)
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["losses"]["val"],
    label="val",
)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Transformer loss")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_curve_transformer.png"))
plt.close()
