import os, pathlib, random, string, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# optional benchmark loader (from prompt)
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        out[split] = _load(f"{split}.csv")
    return out


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    # synthetic fallback (simple parity rule)
    def synth(n, start_id=0):
        rows = []
        for i in range(start_id, start_id + n):
            L = random.randint(5, 18)
            seq = "".join(random.choices(string.ascii_uppercase[:12], k=L))
            label = int(seq.count("A") % 2 == 0)
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    spr = DatasetDict()
    spr["train"] = load_dataset(
        "json", data_files={"train": synth(4000)}, split="train"
    )
    spr["dev"] = load_dataset(
        "json", data_files={"train": synth(800, 5000)}, split="train"
    )
    spr["test"] = load_dataset(
        "json", data_files={"train": synth(800, 6000)}, split="train"
    )
print({k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------------
# vocabulary
vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)
max_len = min(max(len(ex["sequence"]) for ex in spr["train"]), 120)
print(f"Vocab size={vocab_size}, max_len={max_len}")


def encode(seq):
    ids = [vocab.get(c, 1) for c in seq][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


# ------------------------------------------------------------------
# torch dataset
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds):
        self.data = hf_ds
        self.weight_key = (
            "complexity"
            if "complexity" in hf_ds.column_names
            else (
                "rule_complexity" if "rule_complexity" in hf_ds.column_names else None
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        item = {
            "input_ids": torch.tensor(encode(ex["sequence"]), dtype=torch.long),
            "label": torch.tensor(int(ex["label"]), dtype=torch.long),
            "weight": (
                torch.tensor(float(ex[self.weight_key]))
                if self.weight_key
                else torch.tensor(1.0)
            ),
        }
        return item


def collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    out["labels"] = out.pop("label")
    return out


train_ds, dev_ds, test_ds = (SPRTorchDataset(spr[s]) for s in ["train", "dev", "test"])


# ------------------------------------------------------------------
# Transformer classifier
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab,
        d_model=128,
        nhead=4,
        nlayers=2,
        dim_ff=256,
        n_classes=2,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, nlayers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.embed(x) + self.pos[:, : x.size(1), :]
        h = self.enc(h, src_key_padding_mask=(x == 0))
        h = h.masked_fill((x == 0).unsqueeze(-1), 0)  # zero-out pads
        h = h.sum(1) / (x != 0).sum(1, keepdim=True).clamp(min=1)  # mean pooling
        return self.fc(h)


model = TransformerClassifier(vocab_size).to(device)

# ------------------------------------------------------------------
# training prep
batch_size = 32
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)


# ------------------------------------------------------------------
def complexity_weighted_acc(preds, labels, weights):
    correct = (preds == labels).astype(float)
    return (correct * weights).sum() / weights.sum()


# experiment data dict
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"val_macroF1": [], "val_CWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

best_val_loss, patience, es_counter = float("inf"), 3, 0
epochs = 15
for epoch in range(1, epochs + 1):
    # ----------------- train -----------------
    model.train()
    tot_loss = tot_items = 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        tot_items += batch["labels"].size(0)
    train_loss = tot_loss / tot_items
    # ----------------- validate -----------------
    model.eval()
    v_loss = v_items = 0
    all_preds, all_labels, all_w = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            v_loss += loss.item() * batch["labels"].size(0)
            v_items += batch["labels"].size(0)
            preds = logits.argmax(1).cpu().numpy()
            lbls = batch["labels"].cpu().numpy()
            wts = batch["weight"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls)
            all_w.extend(wts)
    val_loss = v_loss / v_items
    macroF1 = f1_score(all_labels, all_preds, average="macro")
    CWA = complexity_weighted_acc(
        np.array(all_preds), np.array(all_labels), np.array(all_w)
    )
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_macroF1={macroF1:.4f} | CWA={CWA:.4f}"
    )
    # record
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_macroF1"].append(macroF1)
    experiment_data["SPR_BENCH"]["metrics"]["val_CWA"].append(CWA)
    scheduler.step()
    # early stopping
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        es_counter = 0
    else:
        es_counter += 1
        if es_counter >= patience:
            print("Early stopping triggered.")
            break

# store last dev preds for inspection
experiment_data["SPR_BENCH"]["predictions"] = all_preds
experiment_data["SPR_BENCH"]["ground_truth"] = all_labels
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Saved logs to {os.path.join(working_dir, "experiment_data.npy")}')
