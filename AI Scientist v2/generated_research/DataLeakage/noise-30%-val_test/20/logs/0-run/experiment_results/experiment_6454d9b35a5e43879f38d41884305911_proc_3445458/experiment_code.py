import os, math, random, string, pathlib, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------- housekeeping ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data dict ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_cwa": [], "val_cwa": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------- data loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    print("SPR_BENCH not found – using synthetic toy data.")

    def synth(n):
        rows = []
        for i in range(n):
            seq = "".join(
                random.choices(string.ascii_uppercase[:12], k=random.randint(5, 20))
            )
            label = int(seq.count("A") % 2 == 0)
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    def to_hf(rows):
        return load_dataset("json", data_files={"train": rows}, split="train")

    spr = DatasetDict()
    spr["train"] = to_hf(synth(2000))
    spr["dev"] = to_hf(synth(500))
    spr["test"] = to_hf(synth(1000))

print({k: len(v) for k, v in spr.items()})

# ---------- vocab ----------
vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)
max_len = min(120, max(len(ex["sequence"]) for ex in spr["train"]))
print(f"vocab_size={vocab_size}, max_len={max_len}")


def encode(seq):
    ids = [vocab.get(c, 1) for c in seq][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


# ---------- dataset ----------
class SPRDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        ids = torch.tensor(encode(ex["sequence"]), dtype=torch.long)
        lbl = torch.tensor(int(ex["label"]), dtype=torch.long)
        length = torch.tensor(min(len(ex["sequence"]), max_len), dtype=torch.long)
        return {"input_ids": ids, "labels": lbl, "lengths": length}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


train_ds, dev_ds, test_ds = (
    SPRDataset(spr["train"]),
    SPRDataset(spr["dev"]),
    SPRDataset(spr["test"]),
)


# ---------- model ----------
class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab,
        emb_dim=128,
        n_heads=4,
        hidden_dim=256,
        n_layers=2,
        num_classes=2,
        p_drop=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=p_drop,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        b, l = x.shape
        pos_ids = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        h = self.emb(x) + self.pos(pos_ids)
        h = self.transformer(h)
        h = (h.masked_fill((x == 0).unsqueeze(-1), 0.0)).sum(1) / (x != 0).sum(
            1, keepdim=True
        ).clamp(min=1)
        return self.fc(h)


# ---------- hyperparams ----------
batch_size = 32
epochs = 15
patience = 3
lr = 1e-4
warmup_steps = 200
grad_clip = 1.0

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

model = CharTransformer(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda step: min((step + 1) / warmup_steps, 1.0)
)

best_val_cwa, stale = -1, 0


# ---------- metric helper ----------
def compute_metrics(logits_or_preds, labels, lengths):
    """
    Accepts either raw logits (2-D) or already argmax'ed predictions (1-D).
    Returns CWA, macro-F1, preds list.
    """
    if logits_or_preds.dim() == 2:  # logits
        preds = logits_or_preds.argmax(1)
    else:  # already predictions
        preds = logits_or_preds
    correct = (preds == labels).cpu().numpy()
    w = lengths.cpu().numpy().astype(np.float32)
    cwa = (correct * w).sum() / w.sum()
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
    return cwa, f1, preds.cpu().tolist()


# ---------- training ----------
for epoch in range(1, epochs + 1):
    # ----- train -----
    model.train()
    tot_loss, tot_samples = 0.0, 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        bs = batch["labels"].size(0)
        tot_loss += loss.item() * bs
        tot_samples += bs
    train_loss = tot_loss / tot_samples
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ----- validation -----
    model.eval()
    val_loss, val_samples = 0.0, 0
    all_preds, all_labels, all_lengths = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            bs = batch["labels"].size(0)
            val_loss += loss.item() * bs
            val_samples += bs
            _, _, preds = compute_metrics(logits, batch["labels"], batch["lengths"])
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().tolist())
            all_lengths.extend(batch["lengths"].cpu().tolist())
    val_loss /= val_samples
    val_cwa = (
        (np.array(all_preds) == np.array(all_labels)) * np.array(all_lengths)
    ).sum() / np.array(all_lengths).sum()
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | CWA={val_cwa:.4f} | Macro-F1={val_f1:.4f}"
    )

    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_cwa"].append(None)
    experiment_data["SPR_BENCH"]["metrics"]["val_cwa"].append(val_cwa)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    # ----- early stopping -----
    if val_cwa > best_val_cwa:
        best_val_cwa = val_cwa
        stale = 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        stale += 1
        if stale >= patience:
            print("Early stopping.")
            break

# ---------- test ----------
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best_model.pt"), map_location=device)
)
model.eval()
all_preds, all_labels, all_lengths = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())
        all_lengths.extend(batch["lengths"].cpu().tolist())
test_cwa = (
    (np.array(all_preds) == np.array(all_labels)) * np.array(all_lengths)
).sum() / np.array(all_lengths).sum()
test_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"TEST   : CWA={test_cwa:.4f} | Macro-F1={test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = all_preds
experiment_data["SPR_BENCH"]["ground_truth"] = all_labels

# ---------- save artefacts ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
