# No-Transformer-Context ablation for SPR-BENCH
import os, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------
# experiment bookkeeping
# -------------------------------------------------------
experiment_data = {
    "no_transformer_context": {
        "spr_bench": {
            "epochs": [],
            "metrics": {"train_f1": [], "val_f1": [], "test_f1": None},
            "losses": {"train": [], "val": [], "test": None},
            "predictions": [],
            "ground_truth": [],
        }
    }
}
ed = experiment_data["no_transformer_context"]["spr_bench"]

# -------------------------------------------------------
# misc
# -------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -------------------------------------------------------
# data helpers (same as baseline)
# -------------------------------------------------------
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

# -------------------------------------------------------
# vocab construction
# -------------------------------------------------------
special_tokens = ["<PAD>", "<UNK>", "<SOS>"]
char_vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
for seq in spr["train"]["sequence"]:
    for ch in seq:
        char_vocab.setdefault(ch, len(char_vocab))
pad_id, unk_id, sos_id = char_vocab["<PAD>"], char_vocab["<UNK>"], char_vocab["<SOS>"]

bigram_vocab = {"<PAD>": 0, "<UNK>": 1}
for seq in spr["train"]["sequence"]:
    prev = "<SOS>"
    for ch in seq:
        bg = prev + ch
        if bg not in bigram_vocab:
            bigram_vocab[bg] = len(bigram_vocab)
        prev = ch
print(f"Char vocab: {len(char_vocab)}, Bigram vocab: {len(bigram_vocab)}")


# -------------------------------------------------------
# encoding
# -------------------------------------------------------
def encode(example):
    seq = example["sequence"]
    char_ids, bigram_ids = [], []
    prev = "<SOS>"
    for ch in seq:
        char_ids.append(char_vocab.get(ch, unk_id))
        bigram_ids.append(bigram_vocab.get(prev + ch, bigram_vocab["<UNK>"]))
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
# collator
# -------------------------------------------------------
def collate(batch):
    max_len = max(len(b["char_ids"]) for b in batch)
    B = len(batch)
    char_tensor = torch.full((B, max_len), pad_id, dtype=torch.long)
    bigram_tensor = torch.full_like(char_tensor, bigram_vocab["<PAD>"])
    mask = torch.zeros_like(char_tensor, dtype=torch.bool)
    counts = torch.stack(
        [torch.tensor(b["count_vec"], dtype=torch.float32) for b in batch]
    )
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["char_ids"])
        char_tensor[i, :L] = torch.tensor(b["char_ids"])
        bigram_tensor[i, :L] = torch.tensor(b["bigram_ids"])
        mask[i, :L] = 1
    return {
        "char_ids": char_tensor,
        "bigram_ids": bigram_tensor,
        "mask": mask,
        "count_vec": counts,
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
# Model without Transformer
# -------------------------------------------------------
class CBCMeanPool(nn.Module):
    def __init__(
        self,
        char_vocab_size,
        bigram_vocab_size,
        num_labels,
        d_model=256,
        dropout=0.2,
        max_len=512,
    ):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, d_model, padding_idx=pad_id)
        self.bigram_emb = nn.Embedding(
            bigram_vocab_size, d_model, padding_idx=bigram_vocab["<PAD>"]
        )
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.count_proj = nn.Sequential(
            nn.Linear(char_vocab_size, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_model * 2, num_labels)

    def forward(self, char_ids, bigram_ids, mask, count_vec):
        L = char_ids.size(1)
        tok_repr = (
            self.char_emb(char_ids) + self.bigram_emb(bigram_ids) + self.pos_emb[:L]
        )
        masked_sum = (tok_repr * mask.unsqueeze(-1)).sum(1)
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        seq_repr = masked_sum / denom
        count_repr = self.count_proj(count_vec)
        cat = torch.cat([seq_repr, count_repr], dim=-1)
        return self.classifier(cat)


model = CBCMeanPool(len(char_vocab), len(bigram_vocab), num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


# -------------------------------------------------------
# epoch runner
# -------------------------------------------------------
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
                batch["mask"],
                batch["count_vec"],
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
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# -------------------------------------------------------
# training loop
# -------------------------------------------------------
best_val, wait, patience = 0.0, 0, 3
max_epochs = 15
save_path = os.path.join(working_dir, "cbc_meanpool_best.pt")

for epoch in range(1, max_epochs + 1):
    st = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(loaders["train"], train=True)
    val_loss, val_f1, _, _ = run_epoch(loaders["dev"])
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}  val_F1={val_f1:.4f}")
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
            print("Early stopping.")
            break
    print(f"  epoch time {time.time()-st:.1f}s  best_val_F1={best_val:.4f}")

# -------------------------------------------------------
# test evaluation
# -------------------------------------------------------
model.load_state_dict(torch.load(save_path))
test_loss, test_f1, test_preds, test_gts = run_epoch(loaders["test"])
print("Test Macro F1:", test_f1)
ed["losses"]["test"] = test_loss
ed["metrics"]["test_f1"] = test_f1
ed["predictions"] = test_preds
ed["ground_truth"] = test_gts

# -------------------------------------------------------
# save artifacts
# -------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
