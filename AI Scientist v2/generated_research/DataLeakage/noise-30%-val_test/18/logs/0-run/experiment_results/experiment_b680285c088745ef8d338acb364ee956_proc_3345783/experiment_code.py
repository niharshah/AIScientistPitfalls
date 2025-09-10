import os, pathlib, random, string, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score

# ---------- experiment bookkeeping ----------
experiment_data = {
    "NoFeedForwardLayer": {
        "SPR_BENCH": {
            "metrics": {"train_MCC": [], "val_MCC": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load SPR-BENCH or synth ----------
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


for split in dsets.keys():
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=["sequence"]
    )


# ---------- dataloaders ----------
def collate(batch):
    ids = [ex["input_ids"] for ex in batch]
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), vocab[PAD], dtype=torch.long)
    attn_mask = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attn_mask[i, : len(seq)] = 1
    return {"input_ids": padded, "attention_mask": attn_mask, "labels": labels}


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


# ---------- custom encoder layer without FFN ----------
class EncoderLayerNoFFN(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        # Self-attention block
        attn_out, _ = self.self_attn(
            src, src, src, key_padding_mask=src_key_padding_mask, need_weights=False
        )
        src = self.norm1(src + self.dropout1(attn_out))
        return src


# ---------- transformer with NoFeedForwardLayer ----------
class SPRTransformerNoFFN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_heads=4, depth=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.randn(512, embed_dim))  # max length 512
        self.layers = nn.ModuleList(
            [EncoderLayerNoFFN(embed_dim, n_heads, dropout) for _ in range(depth)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, ids, attn_mask):
        x = self.embed(ids) + self.pos[: ids.shape[1]]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~attn_mask)
        cls_vec = x[:, 0]
        return self.fc(self.dropout(cls_vec))


model = SPRTransformerNoFFN(vocab_size).to(device)

# ---------- training utilities ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)

best_val_mcc, best_state = -1.0, None
epochs = 10
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
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        tr_loss.append(loss.item())
        tr_preds.extend(logits.argmax(1).cpu().numpy())
        tr_gts.extend(batch["labels"].cpu().numpy())
    train_mcc = matthews_corrcoef(tr_gts, tr_preds)

    # ---- validation ----
    model.eval()
    val_loss, val_preds, val_gts = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            val_loss.append(loss.item())
            val_preds.extend(logits.argmax(1).cpu().numpy())
            val_gts.extend(batch["labels"].cpu().numpy())
    val_mcc = matthews_corrcoef(val_gts, val_preds)
    print(f"Epoch {epoch}: val_loss={np.mean(val_loss):.4f}, val_MCC={val_mcc:.4f}")

    # ---- logging ----
    exp = experiment_data["NoFeedForwardLayer"]["SPR_BENCH"]
    exp["losses"]["train"].append(np.mean(tr_loss))
    exp["losses"]["val"].append(np.mean(val_loss))
    exp["metrics"]["train_MCC"].append(train_mcc)
    exp["metrics"]["val_MCC"].append(val_mcc)

    # ---- save best ----
    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# ---------- test ----------
model.load_state_dict(best_state)
model.eval()
test_preds, test_gts = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"])
        test_preds.extend(logits.argmax(1).cpu().numpy())
        test_gts.extend(batch["labels"].cpu().numpy())
test_mcc = matthews_corrcoef(test_gts, test_preds)
test_f1 = f1_score(test_gts, test_preds, average="macro")
print(
    f"Best Val MCC={best_val_mcc:.4f} | Test MCC={test_mcc:.4f} | Test F1={test_f1:.4f}"
)

experiment_data["NoFeedForwardLayer"]["SPR_BENCH"]["predictions"] = test_preds
experiment_data["NoFeedForwardLayer"]["SPR_BENCH"]["ground_truth"] = test_gts

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to ./working/experiment_data.npy")
