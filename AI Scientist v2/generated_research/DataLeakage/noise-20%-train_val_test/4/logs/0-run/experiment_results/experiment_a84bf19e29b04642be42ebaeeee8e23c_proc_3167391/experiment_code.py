import os, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- GPU / CPU ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data ----------
experiment_data = {
    "char_count_transformer": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": None},
        "losses": {"train": [], "val": [], "test": None},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- SPR loader ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

num_labels = len(set(spr["train"]["label"]))

# ---------- character vocab ----------
special = ["<PAD>", "<UNK>", "<SOS>"]
char_vocab = {tok: idx for idx, tok in enumerate(special)}
for seq in spr["train"]["sequence"]:
    for ch in seq:
        if ch not in char_vocab:
            char_vocab[ch] = len(char_vocab)
pad_id, unk_id, sos_id = char_vocab["<PAD>"], char_vocab["<UNK>"], char_vocab["<SOS>"]
vocab_size = len(char_vocab)
print("Char vocab size:", vocab_size)


# ---------- encode with count vector ----------
def encode(example):
    seq = example["sequence"]
    char_ids, counts = [], [0] * vocab_size
    prev = "<SOS>"
    for ch in seq:
        idx = char_vocab.get(ch, unk_id)
        char_ids.append(idx)
        counts[idx] += 1
        prev = ch
    return {"char_ids": char_ids, "counts": counts}


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(encode, remove_columns=[])


# ---------- collate ----------
def collate(batch):
    max_len = max(len(b["char_ids"]) for b in batch)
    seq_tensor = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros_like(seq_tensor, dtype=torch.bool)
    count_tensor = torch.tensor([b["counts"] for b in batch], dtype=torch.float)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        l = len(b["char_ids"])
        seq_tensor[i, :l] = torch.tensor(b["char_ids"], dtype=torch.long)
        attn_mask[i, :l] = 1
    return {
        "char_ids": seq_tensor,
        "attention_mask": attn_mask,
        "counts": count_tensor,
        "labels": labels,
    }


batch_size = 128
loaders = {
    split: DataLoader(
        spr[split],
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    for split in ["train", "dev", "test"]
}


# ---------- model ----------
class CountAugTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=256,
        nhead=8,
        nlayers=6,
        dim_ff=512,
        dropout=0.2,
        max_len=512,
    ):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.count_proj = nn.Sequential(
            nn.LayerNorm(vocab_size), nn.Linear(vocab_size, d_model), nn.GELU()
        )
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, char_ids, attention_mask, counts):
        L = char_ids.size(1)
        x = self.char_emb(char_ids) + self.pos_emb[:L]
        enc = self.encoder(x, src_key_padding_mask=~attention_mask)
        pooled = (enc * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        count_vec = self.count_proj(counts)
        fused = pooled + count_vec
        return self.classifier(fused)


model = CountAugTransformer(vocab_size, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


# ---------- helpers ----------
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.set_grad_enabled(train):
            logits = model(batch["char_ids"], batch["attention_mask"], batch["counts"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ---------- training loop ----------
best_val_f1, wait, patience = 0.0, 0, 3
max_epochs = 15
exp = experiment_data["char_count_transformer"]

for epoch in range(1, max_epochs + 1):
    t0 = time.time()
    train_loss, train_f1, _, _ = run_epoch(loaders["train"], train=True)
    val_loss, val_f1, _, _ = run_epoch(loaders["dev"], train=False)
    print(
        f"Epoch {epoch}: val_F1={val_f1:.4f} val_loss={val_loss:.4f} "
        f"train_loss={train_loss:.4f} ({time.time()-t0:.1f}s)"
    )
    exp["epochs"].append(epoch)
    exp["losses"]["train"].append(train_loss)
    exp["losses"]["val"].append(val_loss)
    exp["metrics"]["train_f1"].append(train_f1)
    exp["metrics"]["val_f1"].append(val_f1)
    if val_f1 > best_val_f1:
        best_val_f1, wait = val_f1, 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ---------- test evaluation ----------
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
test_loss, test_f1, preds, gts = run_epoch(loaders["test"], train=False)
print(f"Test Macro F1: {test_f1:.4f}")
exp["losses"]["test"] = test_loss
exp["metrics"]["test_f1"] = test_f1
exp["predictions"] = preds
exp["ground_truth"] = gts

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
