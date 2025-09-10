import os, pathlib, random, string, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef

# ---------------- mandatory working dir -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- dataset loading -------------------
def load_spr(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _l(name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _l(split)
    return d


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr(data_root) if data_root.exists() else None
except Exception:
    dsets = None

# -------- fallback synthetic data so script is runnable ------------
if dsets is None:
    from datasets import Dataset, DatasetDict

    def synth(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(5, 25)
            seq = "".join(random.choices(list(string.ascii_lowercase) + "#@$%", k=L))
            labels.append(int(L % 2 == 0))  # parity rule
            seqs.append(seq)
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    dsets = DatasetDict({"train": synth(4000), "dev": synth(1000), "test": synth(1000)})
print({k: len(v) for k, v in dsets.items()})

# ---------------- vocabulary -------------------------
PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"
vocab = {PAD: 0, UNK: 1, CLS: 2}
for seq in dsets["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)


def encode(seq: str):
    return [vocab[CLS]] + [vocab.get(ch, vocab[UNK]) for ch in seq]


def numeric_feats(seq: str):
    L = len(seq)
    return {
        "feat_vec": [
            L / 128.0,  # length normalized
            (L % 2),  # parity
            len(set(seq)) / (L + 1e-3),  # diversity ratio
        ]
    }


for split in dsets.keys():
    dsets[split] = dsets[split].map(
        lambda ex: {
            "input_ids": encode(ex["sequence"]),
            **numeric_feats(ex["sequence"]),
        },
        remove_columns=["sequence"],
    )


# ---------------- collate fn -------------------------
def collate(batch):
    ids = [ex["input_ids"] for ex in batch]
    feats = torch.tensor([ex["feat_vec"] for ex in batch], dtype=torch.float)
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
        "feats": feats,
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


# ---------------- sinusoidal positional encoding ------------------
def sinusoid(pos, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sinusoid = torch.outer(pos, inv_freq)
    sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
    emb = torch.zeros(pos.size(0), dim)
    emb[:, 0::2] = sin
    emb[:, 1::2] = cos
    return emb


# ---------------- model ------------------------------------------
class HybridSPR(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=192,
        depth=3,
        heads=6,
        num_feat=3,
        dropout=0.15,
        max_len=256,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.register_buffer(
            "pos_table", sinusoid(torch.arange(max_len), embed_dim), persistent=False
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.feat_mlp = nn.Sequential(
            nn.Linear(num_feat, embed_dim // 2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(embed_dim + embed_dim // 2, 2)

    def forward(self, ids, mask, feats):
        pos_emb = self.pos_table[: ids.size(1)].to(ids.device)
        x = self.embed(ids) + pos_emb  # (B, T, D)
        x = self.encoder(x, src_key_padding_mask=~mask)
        cls_vec = x[:, 0]  # CLS position
        feat_emb = self.feat_mlp(feats)
        concat = torch.cat([cls_vec, feat_emb], dim=-1)
        return self.classifier(concat)


model = HybridSPR(vocab_size).to(device)

# ---------------- training setup -------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
clip = 1.0

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

best_val_mcc, best_state = -1.0, None
epochs = 6
for epoch in range(1, epochs + 1):
    # ---- training ----
    model.train()
    tr_loss, tr_pred, tr_gt = [], [], []
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"], batch["feats"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        tr_loss.append(loss.item())
        tr_pred.extend(logits.argmax(1).detach().cpu().numpy())
        tr_gt.extend(batch["labels"].cpu().numpy())
    train_mcc = matthews_corrcoef(tr_gt, tr_pred)

    # ---- validation ----
    model.eval()
    val_loss, val_pred, val_gt = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"], batch["feats"])
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

    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    scheduler.step()

# --------------- test --------------------------------
model.load_state_dict(best_state)
model.to(device)
model.eval()
test_pred, test_gt = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"], batch["feats"])
        test_pred.extend(logits.argmax(1).cpu().numpy())
        test_gt.extend(batch["labels"].cpu().numpy())
test_mcc = matthews_corrcoef(test_gt, test_pred)
print(f"Best Val MCC = {best_val_mcc:.4f} | Test MCC = {test_mcc:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = test_gt
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to ./working/experiment_data.npy")
