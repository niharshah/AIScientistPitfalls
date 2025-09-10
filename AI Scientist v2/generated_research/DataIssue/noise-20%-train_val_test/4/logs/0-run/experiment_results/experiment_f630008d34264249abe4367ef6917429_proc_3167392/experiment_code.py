import os, pathlib, time, math, numpy as np, torch, torch.nn.functional as F
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
    "count_token_transformer": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": None},
        "losses": {"train": [], "val": [], "test": None},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- load SPR_BENCH ----------
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

# ---------- build vocabularies ----------
special_tokens = ["<PAD>", "<UNK>", "<SOS>"]
char_vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
for seq in spr["train"]["sequence"]:
    for ch in seq:
        if ch not in char_vocab:
            char_vocab[ch] = len(char_vocab)
pad_id, unk_id, sos_id = [char_vocab[t] for t in ["<PAD>", "<UNK>", "<SOS>"]]

bigram_vocab = {tok: idx for idx, tok in enumerate(["<PAD>", "<UNK>"])}
for seq in spr["train"]["sequence"]:
    prev = "<SOS>"
    for ch in seq:
        bg = prev + ch
        if bg not in bigram_vocab:
            bigram_vocab[bg] = len(bigram_vocab)
        prev = ch

print(f"Char vocab size={len(char_vocab)}, Bigram vocab size={len(bigram_vocab)}")


# ---------- encode sequences ----------
def encode(example):
    seq = example["sequence"]
    char_ids, bigram_ids = [], []
    prev = "<SOS>"
    for ch in seq:
        char_ids.append(char_vocab.get(ch, unk_id))
        bigram_ids.append(bigram_vocab.get(prev + ch, bigram_vocab["<UNK>"]))
        prev = ch
    return {"char_ids": char_ids, "bigram_ids": bigram_ids}


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(encode, remove_columns=[])


# ---------- collate ----------
def collate_fn(batch):
    max_len = max(len(b["char_ids"]) for b in batch)
    char_tensor = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    bigram_tensor = torch.full_like(char_tensor, bigram_vocab["<PAD>"])
    attn_mask = torch.zeros_like(char_tensor, dtype=torch.bool)
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
        "labels": labels,
    }


batch_size = 128
loaders = {
    split: DataLoader(
        spr[split],
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    for split in ["train", "dev", "test"]
}


# ---------- model ----------
class CountTokenTransformer(nn.Module):
    def __init__(
        self,
        char_vocab_size,
        bigram_vocab_size,
        num_labels,
        d_model=256,
        nhead=8,
        nlayers=8,
        dim_ff=512,
        dropout=0.2,
        max_len=512,
    ):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, d_model, padding_idx=pad_id)
        self.bigram_emb = nn.Embedding(
            bigram_vocab_size, d_model, padding_idx=bigram_vocab["<PAD>"]
        )
        self.pos_emb = nn.Parameter(torch.randn(max_len + 2, d_model) * 0.02)
        self.count_proj = nn.Linear(char_vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, char_ids, bigram_ids, attention_mask):
        B, L = char_ids.size()
        tok_emb = self.char_emb(char_ids) + self.bigram_emb(bigram_ids)
        tok_emb = tok_emb + self.pos_emb[:L]

        # ----- global count token -----
        one_hot = F.one_hot(
            char_ids.clamp(max=self.char_emb.num_embeddings - 1),
            num_classes=self.char_emb.num_embeddings,
        ).float()
        one_hot[:, :, pad_id] = 0.0
        counts = one_hot.sum(1)  # (B, V)
        count_tok = self.count_proj(counts)  # (B, d_model)
        count_tok = count_tok + self.pos_emb[L]  # positional encoding
        count_tok = count_tok.unsqueeze(1)  # (B,1,d_model)

        emb_ext = torch.cat([count_tok, tok_emb], dim=1)  # (B, L+1, d)
        attn_ext = torch.cat(
            [
                torch.ones(B, 1, device=attention_mask.device, dtype=torch.bool),
                attention_mask,
            ],
            dim=1,
        )

        enc_out = self.encoder(emb_ext, src_key_padding_mask=~attn_ext)
        cls_rep = enc_out[:, 0]  # representation of count token
        logits = self.cls(cls_rep)
        return logits


model = CountTokenTransformer(len(char_vocab), len(bigram_vocab), num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


# ---------- helpers ----------
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
                batch["char_ids"], batch["bigram_ids"], batch["attention_mask"]
            )
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ---------- training loop ----------
best_val_f1, patience, wait = 0.0, 3, 0
max_epochs = 15
best_state = os.path.join(working_dir, "best_counttok.pt")

for epoch in range(1, max_epochs + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(loaders["train"], train=True)
    val_loss, val_f1, _, _ = run_epoch(loaders["dev"], train=False)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}; MacroF1 = {val_f1:.4f}")
    ed = experiment_data["count_token_transformer"]
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_f1"].append(tr_f1)
    ed["metrics"]["val_f1"].append(val_f1)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_state)
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ---------- evaluation ----------
model.load_state_dict(torch.load(best_state))
test_loss, test_f1, preds, gts = run_epoch(loaders["test"], train=False)
print(f"Test Macro F1: {test_f1:.4f}")

ed = experiment_data["count_token_transformer"]
ed["losses"]["test"] = test_loss
ed["metrics"]["test_f1"] = test_f1
ed["predictions"] = preds
ed["ground_truth"] = gts

# ---------- save experiment data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
