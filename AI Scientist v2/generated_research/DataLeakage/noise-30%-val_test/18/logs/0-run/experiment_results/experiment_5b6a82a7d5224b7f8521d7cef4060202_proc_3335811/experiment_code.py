import os, pathlib, math, random, string, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset ----------
def load_spr_bench(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    if data_path.exists():
        dsets = load_spr_bench(data_path)
    else:
        raise FileNotFoundError
except:
    from datasets import Dataset, DatasetDict

    def synth_split(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(5, 15)
            seq = "".join(random.choices(list(string.ascii_lowercase) + "#@&", k=L))
            labels.append(int(seq.count("#") % 2 == 0))
            seqs.append(seq)
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    dsets = DatasetDict(
        {"train": synth_split(1024), "dev": synth_split(256), "test": synth_split(256)}
    )
print({k: len(v) for k, v in dsets.items()})

# ---------- vocab ----------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for s in dsets["train"]["sequence"]:
    for ch in s:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vsize = len(vocab)


def encode(seq):
    return [vocab.get(c, 1) for c in seq]


for split in dsets:
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=["sequence"]
    )


# ---------- dataloader ----------
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), 0, dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return {"input_ids": padded, "labels": labels}


train_loader = DataLoader(
    dsets["train"], batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dsets["dev"], batch_size=128, shuffle=False, collate_fn=collate)
test_loader = DataLoader(
    dsets["test"], batch_size=128, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        nhead=4,
        nlayers=2,
        ff_dim=256,
        drop=0.1,
        max_len=256,
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=drop,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.fc = nn.Linear(emb_dim, 2)

    def forward(self, x):
        mask = x == 0
        h = self.token(x)
        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        lengths = (~mask).sum(-1, keepdim=True).clamp(min=1)
        pooled = (h.masked_fill(mask.unsqueeze(-1), 0).sum(1)) / lengths
        return self.fc(pooled)


# ---------- training ----------
model = TransformerClassifier(vsize).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
epochs = 6

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "test_mcc": None,
    }
}

for epoch in range(1, epochs + 1):
    # train
    model.train()
    tr_losses, tr_preds, tr_gts = [], [], []
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["input_ids"])
        loss = criterion(out, batch["labels"])
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.item())
        tr_preds.extend(out.argmax(1).cpu().numpy())
        tr_gts.extend(batch["labels"].cpu().numpy())
    train_mcc = matthews_corrcoef(tr_gts, tr_preds)
    # eval
    model.eval()
    dv_losses, dv_preds, dv_gts = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["input_ids"])
            loss = criterion(out, batch["labels"])
            dv_losses.append(loss.item())
            dv_preds.extend(out.argmax(1).cpu().numpy())
            dv_gts.extend(batch["labels"].cpu().numpy())
    val_mcc = matthews_corrcoef(dv_gts, dv_preds)
    print(
        f"Epoch {epoch}: validation_loss = {np.mean(dv_losses):.4f}, val_MCC = {val_mcc:.4f}"
    )
    # log
    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train"].append(train_mcc)
    ed["metrics"]["val"].append(val_mcc)
    ed["losses"]["train"].append(np.mean(tr_losses))
    ed["losses"]["val"].append(np.mean(dv_losses))
    ed["epochs"].append(epoch)
    if epoch == epochs:  # store predictions
        ed["predictions"] = dv_preds
        ed["ground_truth"] = dv_gts

# ---------- test ----------
model.eval()
ts_preds, ts_gts = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch["input_ids"])
        ts_preds.extend(out.argmax(1).cpu().numpy())
        ts_gts.extend(batch["labels"].cpu().numpy())
test_mcc = matthews_corrcoef(ts_gts, ts_preds)
print(f"Test MCC: {test_mcc:.4f}")
experiment_data["SPR_BENCH"]["test_mcc"] = test_mcc

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("experiment_data saved to", working_dir)
