import os, pathlib, random, string, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score

# ---------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- load SPR-BENCH or synthetic fallback ----------
def load_spr_bench(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = {}
    for s in ["train", "dev", "test"]:
        dd[s] = _load(f"{s}.csv")
    from datasets import DatasetDict

    return DatasetDict(dd)


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

# ---------- vocab & encoding ----------
PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"
vocab = {PAD: 0, UNK: 1, CLS: 2}
for s in dsets["train"]["sequence"]:
    for ch in s:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)


def encode(seq):
    return [vocab[CLS]] + [vocab.get(ch, vocab[UNK]) for ch in seq]


for sp in dsets.keys():
    dsets[sp] = dsets[sp].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=["sequence"]
    )


# ---------- dataloaders ----------
def collate(batch):
    ids = [ex["input_ids"] for ex in batch]
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    mx = max(map(len, ids))
    padded = torch.full((len(ids), mx), vocab[PAD], dtype=torch.long)
    att = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq)
        att[i, : len(seq)] = 1
    return {"input_ids": padded, "attention_mask": att, "labels": labels}


bs = 128
train_loader = DataLoader(
    dsets["train"], batch_size=bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dsets["dev"], batch_size=bs, shuffle=False, collate_fn=collate)
test_loader = DataLoader(
    dsets["test"], batch_size=bs, shuffle=False, collate_fn=collate
)


# ---------- model (ablation: NoPaddingMask) ----------
class SPRTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, heads=4, depth=2, drop=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.randn(512, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=drop,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.do = nn.Dropout(drop)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, ids, attn_mask):
        x = self.embed(ids) + self.pos[: ids.size(1)]
        # --- NoPaddingMask: allow full attention ---
        x = self.transformer(x, src_key_padding_mask=None)
        cls = x[:, 0]
        return self.fc(self.do(cls))


model = SPRTransformer(vocab_size).to(device)

# ---------- training utilities ----------
criterion = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)

experiment_data = {
    "NoPaddingMask": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}
best_val, best_state = -1.0, None
epochs = 10
for ep in range(1, epochs + 1):
    # train
    model.train()
    tr_loss = []
    tr_pred = []
    tr_gt = []
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optim.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optim.step()
        tr_loss.append(loss.item())
        tr_pred.extend(logits.argmax(1).cpu().numpy())
        tr_gt.extend(batch["labels"].cpu().numpy())
    train_mcc = matthews_corrcoef(tr_gt, tr_pred)
    # val
    model.eval()
    val_loss = []
    val_pred = []
    val_gt = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            val_loss.append(loss.item())
            val_pred.extend(logits.argmax(1).cpu().numpy())
            val_gt.extend(batch["labels"].cpu().numpy())
    val_mcc = matthews_corrcoef(val_gt, val_pred)
    print(f"Epoch {ep}: val_loss={np.mean(val_loss):.4f}, val_MCC={val_mcc:.4f}")
    # log
    exp = experiment_data["NoPaddingMask"]["SPR_BENCH"]
    exp["losses"]["train"].append(np.mean(tr_loss))
    exp["losses"]["val"].append(np.mean(val_loss))
    exp["metrics"]["train"].append(train_mcc)
    exp["metrics"]["val"].append(val_mcc)
    # best
    if val_mcc > best_val:
        best_val = val_mcc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# ---------- test ----------
model.load_state_dict(best_state)
model.eval()
t_pred, t_gt = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"])
        t_pred.extend(logits.argmax(1).cpu().numpy())
        t_gt.extend(batch["labels"].cpu().numpy())
test_mcc = matthews_corrcoef(t_gt, t_pred)
test_f1 = f1_score(t_gt, t_pred, average="macro")
print(f"Best Val MCC={best_val:.4f} | Test MCC={test_mcc:.4f} | Test F1={test_f1:.4f}")

# save predictions
exp = experiment_data["NoPaddingMask"]["SPR_BENCH"]
exp["predictions"] = t_pred
exp["ground_truth"] = t_gt
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to ./working/experiment_data.npy")
