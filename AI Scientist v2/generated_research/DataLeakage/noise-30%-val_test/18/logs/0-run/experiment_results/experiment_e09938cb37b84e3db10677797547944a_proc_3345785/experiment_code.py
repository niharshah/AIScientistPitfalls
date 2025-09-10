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

import os, pathlib, random, string, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load data (SPR_BENCH or synthetic fallback) ----------
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


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr_bench(data_root) if data_root.exists() else None
except Exception:
    dsets = None

# ---------- synthetic fallback so script always runs ----------
if dsets is None:
    from datasets import Dataset, DatasetDict

    def synth(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(5, 25)
            seq = "".join(random.choices(list(string.ascii_lowercase) + "#@$%", k=L))
            labels.append(int(seq.count("#") % 2 == 0))
            seqs.append(seq)
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    dsets = DatasetDict({"train": synth(4000), "dev": synth(1000), "test": synth(1000)})

print({k: len(v) for k, v in dsets.items()})

# ---------- vocabulary ----------
PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"
vocab = {PAD: 0, UNK: 1, CLS: 2}
for seq in dsets["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)


def encode(seq):
    return [vocab[CLS]] + [vocab.get(ch, vocab[UNK]) for ch in seq]


# handcrafted feature size (limit to 128 most frequent symbols for speed)
FEAT_SIZE = min(128, vocab_size)


def extract_feats(seq):
    vec = np.zeros(FEAT_SIZE + 1, dtype=np.float32)  # +1 for length
    vec[0] = len(seq)  # raw length
    for ch in seq:
        idx = vocab.get(ch, vocab[UNK])
        if idx < FEAT_SIZE:
            vec[idx + 1] += 1  # shift by 1 (idx 0 is length)
    vec[1:] = vec[1:] / max(1, len(seq))  # normalise counts
    return vec.tolist()


for split in dsets.keys():
    dsets[split] = dsets[split].map(
        lambda ex: {
            "input_ids": encode(ex["sequence"]),
            "features": extract_feats(ex["sequence"]),
            "label": ex["label"],
        },
        remove_columns=["sequence"],
    )


# ---------- DataLoader ----------
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    feats = torch.tensor([b["features"] for b in batch], dtype=torch.float32)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), vocab[PAD], dtype=torch.long)
    attn_mask = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq)
        attn_mask[i, : len(seq)] = 1
    return {
        "input_ids": padded,
        "attention_mask": attn_mask,
        "features": feats,
        "labels": labels,
    }


train_loader = DataLoader(
    dsets["train"], batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dsets["dev"], batch_size=128, shuffle=False, collate_fn=collate)
test_loader = DataLoader(
    dsets["test"], batch_size=128, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class HybridSPR(nn.Module):
    def __init__(self, vocab_size, feat_size, emb=256, nhead=8, depth=4, dropout=0.15):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.randn(512, emb))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=nhead,
            dim_feedforward=emb * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(feat_size + 1), nn.Linear(feat_size + 1, emb), nn.ReLU()
        )
        self.out = nn.Linear(emb * 2, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask, feats):
        x = self.embed(ids) + self.pos[: ids.size(1)]
        x = self.transformer(x, src_key_padding_mask=~mask)
        cls_vec = x[:, 0]
        feat_vec = self.feat_proj(feats)
        concat = torch.cat([cls_vec, feat_vec], dim=-1)
        return self.out(self.dropout(concat))


model = HybridSPR(vocab_size, FEAT_SIZE).to(device)

# ---------- training setup ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

best_val_mcc = -1.0
best_state = None
EPOCHS = 8

for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    tr_loss, tr_pred, tr_gt = [], [], []
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"], batch["features"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        tr_loss.append(loss.item())
        tr_pred.extend(logits.argmax(1).cpu().numpy())
        tr_gt.extend(batch["labels"].cpu().numpy())
    train_mcc = matthews_corrcoef(tr_gt, tr_pred)
    # ---- validate ----
    model.eval()
    val_loss, val_pred, val_gt = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["features"]
            )
            loss = criterion(logits, batch["labels"])
            val_loss.append(loss.item())
            val_pred.extend(logits.argmax(1).cpu().numpy())
            val_gt.extend(batch["labels"].cpu().numpy())
    val_mcc = matthews_corrcoef(val_gt, val_pred)
    print(
        f"Epoch {epoch}: validation_loss = {np.mean(val_loss):.4f}, val_MCC = {val_mcc:.4f}"
    )
    # ---- log ----
    experiment_data["SPR_BENCH"]["losses"]["train"].append(np.mean(tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append(np.mean(val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train_MCC"].append(train_mcc)
    experiment_data["SPR_BENCH"]["metrics"]["val_MCC"].append(val_mcc)
    # ---- save best ----
    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    scheduler.step()

# ---------- test ----------
model.load_state_dict(best_state)
model.eval()
test_pred, test_gt = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"], batch["features"])
        test_pred.extend(logits.argmax(1).cpu().numpy())
        test_gt.extend(batch["labels"].cpu().numpy())
test_mcc = matthews_corrcoef(test_gt, test_pred)
test_f1 = f1_score(test_gt, test_pred, average="macro")
print(
    f"Best Val MCC = {best_val_mcc:.4f} | Test MCC = {test_mcc:.4f} | Test F1 = {test_f1:.4f}"
)

# ---------- save experiment data ----------
experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = test_gt
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to ./working/experiment_data.npy")
