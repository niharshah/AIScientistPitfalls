import os, pathlib, random, string, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score

# ----------------- mandatory dirs / device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- load SPR_BENCH or fallback --------------
def load_spr_bench(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _load(csv):  # treat csv as a single split
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr_bench(data_path) if data_path.exists() else 1 / 0
except Exception:  # synthetic tiny dataset if benchmark absent
    from datasets import Dataset, DatasetDict

    def synth(n):
        seqs, labels = [], []
        for i in range(n):
            L = random.randint(5, 15)
            seq = "".join(random.choices(list(string.ascii_lowercase) + "#@&", k=L))
            labels.append(int(seq.count("#") % 2 == 0))
            seqs.append(seq)
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    dsets = DatasetDict({"train": synth(8000), "dev": synth(2000), "test": synth(2000)})

print({k: len(v) for k, v in dsets.items()})

# ----------------- vocab build & encoding ------------------
PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"
vocab = {PAD: 0, UNK: 1, CLS: 2}
for seq in dsets["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)


def encode(seq):
    return [vocab[CLS]] + [vocab.get(ch, vocab[UNK]) for ch in seq]  # prepend CLS


for split in dsets:
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=["sequence"]
    )


# ----------------- collate fn ------------------------------
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), vocab[PAD], dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return {"input_ids": padded, "labels": labels}


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


# ----------------- Transformer classifier -----------------
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_sz,
        emb_dim=128,
        n_heads=4,
        n_layers=4,
        pf_dim=256,
        dropout=0.1,
        max_len=256,
    ):
        super().__init__()
        self.pad_idx = vocab[PAD]
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=self.pad_idx)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=pf_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(emb_dim, 2)

    def forward(self, x):
        # x: (B, L)
        B, L = x.size()
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        mask = x == self.pad_idx  # (B, L)
        h = self.emb(x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        cls_tokens = h[:, 0, :]  # (<cls> is first)
        return self.fc(cls_tokens)


# ----------------- training loop ---------------------------
model = TransformerClassifier(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

best_val_mcc, best_state = -1.0, None
epochs = 8
for epoch in range(1, epochs + 1):
    # --- train ---
    model.train()
    tr_losses, tr_preds, tr_gts = [], [], []
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.item())
        tr_preds.extend(logits.argmax(1).cpu().numpy())
        tr_gts.extend(batch["labels"].cpu().numpy())
    train_mcc = matthews_corrcoef(tr_gts, tr_preds)

    # --- dev ---
    model.eval()
    dv_losses, dv_preds, dv_gts = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            dv_losses.append(loss.item())
            dv_preds.extend(logits.argmax(1).cpu().numpy())
            dv_gts.extend(batch["labels"].cpu().numpy())
    val_mcc = matthews_corrcoef(dv_gts, dv_preds)

    print(
        f"Epoch {epoch}: train_loss={np.mean(tr_losses):.4f}, val_loss={np.mean(dv_losses):.4f}, "
        f"train_MCC={train_mcc:.4f}, val_MCC={val_mcc:.4f}"
    )

    # log
    experiment_data["SPR_BENCH"]["metrics"]["train_MCC"].append(train_mcc)
    experiment_data["SPR_BENCH"]["metrics"]["val_MCC"].append(val_mcc)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(np.mean(tr_losses))
    experiment_data["SPR_BENCH"]["losses"]["val"].append(np.mean(dv_losses))
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    # save best
    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        best_state = model.state_dict()

# ----------------- test evaluation ------------------------
model.load_state_dict(best_state)
model.eval()
tst_preds, tst_gts = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        tst_preds.extend(logits.argmax(1).cpu().numpy())
        tst_gts.extend(batch["labels"].cpu().numpy())
test_mcc = matthews_corrcoef(tst_gts, tst_preds)
test_f1 = f1_score(tst_gts, tst_preds, average="macro")
print(
    f"Best-Dev MCC={best_val_mcc:.4f} | Test MCC={test_mcc:.4f} | Test Macro-F1={test_f1:.4f}"
)

# ----------------- save experiment data -------------------
experiment_data["SPR_BENCH"]["predictions"] = tst_preds
experiment_data["SPR_BENCH"]["ground_truth"] = tst_gts
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("All experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
