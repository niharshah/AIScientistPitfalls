"""
No-Bigram-Embedding Pathway (C + C Transformer) – ablation study.
Trains a character-only transformer with a parallel count-vector pathway on SPR-BENCH.
All plottable data are stored in experiment_data.npy under key
'no_bigram_char_count' → 'spr_bench'.
"""

import os, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------
# experiment data dict (follow required format)
# -------------------------------------------------------
experiment_data = {
    "no_bigram_char_count": {
        "spr_bench": {
            "epochs": [],
            "metrics": {"train_f1": [], "val_f1": [], "test_f1": None},
            "losses": {"train": [], "val": [], "test": None},
            "predictions": [],
            "ground_truth": [],
        }
    }
}
ed = experiment_data["no_bigram_char_count"]["spr_bench"]

# -------------------------------------------------------
# working dir & device
# -------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------
# data loading helper (same as baseline)
# -------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
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

# -------------------------------------------------------
# build vocabularies (identical to baseline)
# -------------------------------------------------------
special_tokens = ["<PAD>", "<UNK>", "<SOS>"]
char_vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
for seq in spr["train"]["sequence"]:
    for ch in seq:
        if ch not in char_vocab:
            char_vocab[ch] = len(char_vocab)
pad_id, unk_id, sos_id = char_vocab["<PAD>"], char_vocab["<UNK>"], char_vocab["<SOS>"]

bigram_vocab = {tok: idx for idx, tok in enumerate(["<PAD>", "<UNK>"])}
for seq in spr["train"]["sequence"]:
    prev = "<SOS>"
    for ch in seq:
        bg = prev + ch
        if bg not in bigram_vocab:
            bigram_vocab[bg] = len(bigram_vocab)
        prev = ch
print(f"Char vocab size {len(char_vocab)}, Bigram vocab size {len(bigram_vocab)}")


# -------------------------------------------------------
# encode samples
# -------------------------------------------------------
def encode(example):
    seq = example["sequence"]
    char_ids, bigram_ids = [], []
    prev = "<SOS>"
    for ch in seq:
        char_ids.append(char_vocab.get(ch, unk_id))
        bg = prev + ch
        bigram_ids.append(bigram_vocab.get(bg, bigram_vocab["<UNK>"]))
        prev = ch
    counts = np.zeros(len(char_vocab), dtype=np.int16)
    for idx in char_ids:
        counts[idx] += 1
    return {
        "char_ids": char_ids,
        "bigram_ids": bigram_ids,
        "count_vec": counts.tolist(),
    }


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(encode, remove_columns=[])


# -------------------------------------------------------
# collate fn
# -------------------------------------------------------
def collate(batch):
    max_len = max(len(b["char_ids"]) for b in batch)
    B = len(batch)
    char_tensor = torch.full((B, max_len), pad_id, dtype=torch.long)
    bigram_tensor = torch.full_like(char_tensor, bigram_vocab["<PAD>"])
    attn_mask = torch.zeros_like(char_tensor, dtype=torch.bool)
    counts_tensor = torch.stack(
        [torch.tensor(b["count_vec"], dtype=torch.float32) for b in batch]
    )
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["char_ids"])
        char_tensor[i, :L] = torch.tensor(b["char_ids"], dtype=torch.long)
        bigram_tensor[i, :L] = torch.tensor(b["bigram_ids"], dtype=torch.long)
        attn_mask[i, :L] = 1
    return {
        "char_ids": char_tensor,
        "bigram_ids": bigram_tensor,  # kept for compatibility, ignored by model
        "attention_mask": attn_mask,
        "count_vec": counts_tensor,
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


# -------------------------------------------------------
# model (Char + Count, NO bigram embedding)
# -------------------------------------------------------
class CCTransformer(nn.Module):
    """
    Character-only token embedding + count-vector pathway transformer.
    """

    def __init__(
        self,
        char_vocab,
        num_labels,
        d_model=256,
        nhead=8,
        nlayers=4,
        dim_feedforward=512,
        dropout=0.2,
        max_len=512,
    ):
        super().__init__()
        self.char_emb = nn.Embedding(len(char_vocab), d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        # count pathway
        self.count_proj = nn.Sequential(
            nn.Linear(len(char_vocab), d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_model * 2, num_labels)

    def forward(self, char_ids, bigram_ids, attention_mask, count_vec):
        # bigram_ids is ignored by design
        L = char_ids.shape[1]
        tok_emb = self.char_emb(char_ids) + self.pos_emb[:L]
        enc_out = self.encoder(tok_emb, src_key_padding_mask=~attention_mask)
        pooled = (enc_out * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        count_rep = self.count_proj(count_vec)
        features = torch.cat([pooled, count_rep], dim=-1)
        return self.classifier(features)


model = CCTransformer(char_vocab, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


# -------------------------------------------------------
# helpers
# -------------------------------------------------------
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.set_grad_enabled(train):
            logits = model(
                batch["char_ids"],
                batch["bigram_ids"],
                batch["attention_mask"],
                batch["count_vec"],
            )
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
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# -------------------------------------------------------
# training loop with early stopping
# -------------------------------------------------------
best_val_f1, patience, wait = 0.0, 3, 0
max_epochs = 15
save_path = os.path.join(working_dir, "cc_best.pt")

for epoch in range(1, max_epochs + 1):
    tic = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(loaders["train"], train=True)
    val_loss, val_f1, _, _ = run_epoch(loaders["dev"], train=False)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_F1={val_f1:.4f}")
    # record
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_f1"].append(tr_f1)
    ed["metrics"]["val_f1"].append(val_f1)
    # early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), save_path)
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break
    print(f"  epoch time {time.time()-tic:.1f}s  best_val_F1={best_val_f1:.4f}")

# -------------------------------------------------------
# test evaluation
# -------------------------------------------------------
model.load_state_dict(torch.load(save_path))
test_loss, test_f1, test_preds, test_gts = run_epoch(loaders["test"], train=False)
print(f"Test Macro F1: {test_f1:.4f}")
ed["losses"]["test"] = test_loss
ed["metrics"]["test_f1"] = test_f1
ed["predictions"] = test_preds
ed["ground_truth"] = test_gts

# -------------------------------------------------------
# save experiment data
# -------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
