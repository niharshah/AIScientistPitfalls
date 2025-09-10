import os, pathlib, random, string, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score

# ---------------- compulsory working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- load SPR_BENCH or synth fallback -------------
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

# ---------------- vocab -----------------
PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"
vocab = {PAD: 0, UNK: 1, CLS: 2}
for seq in dsets["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)


def encode(seq):
    return [vocab[CLS]] + [vocab.get(ch, vocab[UNK]) for ch in seq]


# ----------- build frequency feature space ------------
from collections import Counter

freq_counter = Counter()
for seq in dsets["train"]["sequence"]:
    freq_counter.update(seq)
top_k_chars = [c for c, _ in freq_counter.most_common(20)]
feat_dim = 2 + len(top_k_chars)  # length/100, length parity, per-symbol freq


def build_features(seq):
    L = len(seq)
    vec = [min(L / 100, 1.0), L % 2]  # basic length stats
    cnt = Counter(seq)
    for c in top_k_chars:
        vec.append(cnt[c] / L if L > 0 else 0.0)
    return vec


for split in dsets.keys():
    dsets[split] = dsets[split].map(
        lambda ex: {
            "input_ids": encode(ex["sequence"]),
            "features": build_features(ex["sequence"]),
        },
        remove_columns=["sequence"],
    )


# ---------------- data loaders -----------------
def collate(batch):
    ids = [ex["input_ids"] for ex in batch]
    feats = torch.tensor([ex["features"] for ex in batch], dtype=torch.float32)
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), vocab[PAD], dtype=torch.long)
    attn_mask = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attn_mask[i, : len(seq)] = 1
    return {
        "input_ids": padded,
        "attention_mask": attn_mask,
        "features": feats,
        "labels": labels,
    }


batch_size = 128
train_loader = DataLoader(
    dsets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    dsets["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    dsets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------------- model -----------------
class HybridSPR(nn.Module):
    def __init__(
        self, vocab_size, feat_dim, embed_dim=128, n_heads=4, depth=3, dropout=0.2
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.randn(512, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, embed_dim), nn.ReLU()
        )
        self.fc = nn.Linear(embed_dim * 2, 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids, attn_mask, feats):
        x = self.embed(ids) + self.pos[: ids.size(1)]
        x = self.encoder(x, src_key_padding_mask=~attn_mask)
        cls_vec = x[:, 0]
        feat_vec = self.feat_proj(feats)
        h = torch.cat([cls_vec, feat_vec], dim=-1)
        return self.fc(self.drop(h))


model = HybridSPR(vocab_size, feat_dim).to(device)

# -------------- training utilities ------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

best_val_mcc, best_state = -1.0, None
epochs = 8
for epoch in range(1, epochs + 1):
    # ---- train ----
    model.train()
    tr_loss, tr_preds, tr_gts = [], [], []
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
        tr_preds.extend(logits.argmax(1).cpu().numpy())
        tr_gts.extend(batch["labels"].cpu().numpy())
    train_mcc = matthews_corrcoef(tr_gts, tr_preds)
    # ---- validate ----
    model.eval()
    val_loss, val_preds, val_gts = [], [], []
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
            val_preds.extend(logits.argmax(1).cpu().numpy())
            val_gts.extend(batch["labels"].cpu().numpy())
    val_mcc = matthews_corrcoef(val_gts, val_preds)
    print(
        f"Epoch {epoch}: validation_loss = {np.mean(val_loss):.4f}, val_MCC = {val_mcc:.4f}"
    )
    # ---- log ----
    experiment_data["SPR_BENCH"]["losses"]["train"].append(np.mean(tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append(np.mean(val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train_MCC"].append(train_mcc)
    experiment_data["SPR_BENCH"]["metrics"]["val_MCC"].append(val_mcc)
    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# ------------ test with best model ------------
model.load_state_dict(best_state)
model.to(device)
model.eval()
test_preds, test_gts = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"], batch["features"])
        test_preds.extend(logits.argmax(1).cpu().numpy())
        test_gts.extend(batch["labels"].cpu().numpy())
test_mcc = matthews_corrcoef(test_gts, test_preds)
test_f1 = f1_score(test_gts, test_preds, average="macro")
print(
    f"Best Val MCC = {best_val_mcc:.4f} | Test MCC = {test_mcc:.4f} | Test F1 = {test_f1:.4f}"
)

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to ./working/experiment_data.npy")
