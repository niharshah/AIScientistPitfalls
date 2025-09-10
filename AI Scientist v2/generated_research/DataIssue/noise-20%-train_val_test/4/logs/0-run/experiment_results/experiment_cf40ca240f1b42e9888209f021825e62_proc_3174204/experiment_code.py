import os, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# --------------------------- directories / device ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------- experiment store --------------------------------
experiment_data = {
    "char_bigram_only": {  # ablation key
        "SPR_BENCH": {
            "epochs": [],
            "metrics": {"train_f1": [], "val_f1": [], "test_f1": None},
            "losses": {"train": [], "val": [], "test": None},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# ---------------------------- data loading -----------------------------------
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

# ----------------------------- vocab build -----------------------------------
special_tokens = ["<PAD>", "<UNK>", "<SOS>"]
char_vocab = {t: i for i, t in enumerate(special_tokens)}
for seq in spr["train"]["sequence"]:
    for ch in seq:
        if ch not in char_vocab:
            char_vocab[ch] = len(char_vocab)
pad_id, unk_id, sos_id = char_vocab["<PAD>"], char_vocab["<UNK>"], char_vocab["<SOS>"]

bigram_vocab = {t: i for i, t in enumerate(["<PAD>", "<UNK>"])}
for seq in spr["train"]["sequence"]:
    prev = "<SOS>"
    for ch in seq:
        bg = prev + ch
        if bg not in bigram_vocab:
            bigram_vocab[bg] = len(bigram_vocab)
        prev = ch
print(f"Char vocab {len(char_vocab)}, Bigram vocab {len(bigram_vocab)}")


# ----------------------------- encoding --------------------------------------
def encode(example):
    seq = example["sequence"]
    char_ids, bigram_ids = [], []
    prev = "<SOS>"
    for ch in seq:
        char_ids.append(char_vocab.get(ch, unk_id))
        bg = prev + ch
        bigram_ids.append(bigram_vocab.get(bg, bigram_vocab["<UNK>"]))
        prev = ch
    # count vector retained for compatibility although unused in ablation
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


# ----------------------------- collate ---------------------------------------
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
        "bigram_ids": bigram_tensor,
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
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )
    for split in ["train", "dev", "test"]
}


# ------------------------------ Model ----------------------------------------
class CBTransformer(nn.Module):
    """
    Character + Bigram Transformer WITHOUT count-vector branch.
    """

    def __init__(
        self,
        char_vocab,
        bigram_vocab,
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
        self.bigram_emb = nn.Embedding(
            len(bigram_vocab), d_model, padding_idx=bigram_vocab["<PAD>"]
        )
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, char_ids, bigram_ids, attention_mask, count_vec=None):
        L = char_ids.size(1)
        tok_emb = (
            self.char_emb(char_ids) + self.bigram_emb(bigram_ids) + self.pos_emb[:L]
        )
        enc_out = self.encoder(tok_emb, src_key_padding_mask=~attention_mask)
        pooled = (enc_out * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(pooled)


model = CBTransformer(char_vocab, bigram_vocab, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


# ------------------------------ helpers --------------------------------------
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
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
            )  # ignored
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# --------------------------- train / early stop ------------------------------
best_val, patience, wait = 0.0, 3, 0
max_epochs = 15
save_path = os.path.join(working_dir, "cb_only_best.pt")

for epoch in range(1, max_epochs + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(loaders["train"], train=True)
    val_loss, val_f1, _, _ = run_epoch(loaders["dev"])
    print(f"Epoch {epoch}: val_loss {val_loss:.4f}  val_F1 {val_f1:.4f}")
    ed = experiment_data["char_bigram_only"]["SPR_BENCH"]
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_f1"].append(tr_f1)
    ed["metrics"]["val_f1"].append(val_f1)
    if val_f1 > best_val:
        best_val, wait = val_f1, 0
        torch.save(model.state_dict(), save_path)
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break
    print(f"  epoch time {(time.time()-t0):.1f}s  best_val_F1={best_val:.4f}")

# ------------------------------ test -----------------------------------------
model.load_state_dict(torch.load(save_path))
test_loss, test_f1, preds, gts = run_epoch(loaders["test"])
print(f"Test Macro F1: {test_f1:.4f}")
ed = experiment_data["char_bigram_only"]["SPR_BENCH"]
ed["losses"]["test"] = test_loss
ed["metrics"]["test_f1"] = test_f1
ed["predictions"] = preds
ed["ground_truth"] = gts

# ------------------------------ save -----------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Results saved to", os.path.join(working_dir, "experiment_data.npy"))
