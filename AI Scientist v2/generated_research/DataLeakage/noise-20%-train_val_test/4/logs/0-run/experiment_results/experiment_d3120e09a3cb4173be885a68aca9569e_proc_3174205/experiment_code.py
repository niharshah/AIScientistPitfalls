# Bigram-Only Transformer (B-Only) ablation
import os, pathlib, time, numpy as np, torch, random
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# reproducibility
# ------------------------------------------------------------------
seed = 37
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------------
# experiment bookkeeping ------------------------------------------------
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "bigram_only": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": None},
        "losses": {"train": [], "val": [], "test": None},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ------------------------------------------------------------------
# data -----------------------------------------------------------------
# ------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

num_labels = len(set(spr["train"]["label"]))

# ------------------------------------------------------------------
# vocab -----------------------------------------------------------------
# ------------------------------------------------------------------
special_tokens = ["<PAD>", "<UNK>", "<SOS>"]
char_vocab = {tok: i for i, tok in enumerate(special_tokens)}
for seq in spr["train"]["sequence"]:
    for ch in seq:
        if ch not in char_vocab:
            char_vocab[ch] = len(char_vocab)
pad_id, unk_id, sos_id = [char_vocab[tok] for tok in ["<PAD>", "<UNK>", "<SOS>"]]

bigram_vocab = {"<PAD>": 0, "<UNK>": 1}
for seq in spr["train"]["sequence"]:
    prev = "<SOS>"
    for ch in seq:
        bg = prev + ch
        if bg not in bigram_vocab:
            bigram_vocab[bg] = len(bigram_vocab)
        prev = ch
print(f"Char vocab {len(char_vocab)} | Bigram vocab {len(bigram_vocab)}")


# ------------------------------------------------------------------
# encode ----------------------------------------------------------------
# ------------------------------------------------------------------
def encode(example):
    seq = example["sequence"]
    bigram_ids = []
    prev = "<SOS>"
    for ch in seq:
        bg = prev + ch
        bigram_ids.append(bigram_vocab.get(bg, bigram_vocab["<UNK>"]))
        prev = ch
    return {"bigram_ids": bigram_ids}


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(encode, remove_columns=[])


# ------------------------------------------------------------------
# collate --------------------------------------------------------------
# ------------------------------------------------------------------
def collate(batch):
    max_len = max(len(b["bigram_ids"]) for b in batch)
    B = len(batch)
    bigram_t = torch.full((B, max_len), bigram_vocab["<PAD>"], dtype=torch.long)
    attn_mask = torch.zeros_like(bigram_t, dtype=torch.bool)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["bigram_ids"])
        bigram_t[i, :L] = torch.tensor(b["bigram_ids"], dtype=torch.long)
        attn_mask[i, :L] = 1
    return {"bigram_ids": bigram_t, "attention_mask": attn_mask, "labels": labels}


batch_size = 128
loaders = {
    split: DataLoader(
        spr[split],
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    for split in ["train", "dev", "test"]
}


# ------------------------------------------------------------------
# model ----------------------------------------------------------------
# ------------------------------------------------------------------
class BigramOnlyTransformer(nn.Module):
    def __init__(
        self,
        bigram_vocab_size,
        num_labels,
        d_model=256,
        nhead=8,
        nlayers=4,
        dim_ff=512,
        dropout=0.2,
        max_len=512,
    ):
        super().__init__()
        self.bigram_emb = nn.Embedding(
            bigram_vocab_size, d_model, padding_idx=bigram_vocab["<PAD>"]
        )
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, bigram_ids, attention_mask):
        L = bigram_ids.size(1)
        x = self.bigram_emb(bigram_ids) + self.pos_emb[:L]
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(pooled)


model = BigramOnlyTransformer(len(bigram_vocab), num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


# ------------------------------------------------------------------
# helpers -------------------------------------------------------------
# ------------------------------------------------------------------
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.set_grad_enabled(train):
            logits = model(batch["bigram_ids"], batch["attention_mask"])
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


# ------------------------------------------------------------------
# training loop ------------------------------------------------------
# ------------------------------------------------------------------
best_val_f1, patience, wait = 0.0, 3, 0
max_epochs = 15
save_path = os.path.join(working_dir, "b_only_best.pt")

for epoch in range(1, max_epochs + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(loaders["train"], train=True)
    val_loss, val_f1, _, _ = run_epoch(loaders["dev"], train=False)
    print(f"Epoch {epoch:2d}  val_loss {val_loss:.4f}  val_F1 {val_f1:.4f}")

    ed = experiment_data["bigram_only"]
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_f1"].append(tr_f1)
    ed["metrics"]["val_f1"].append(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), save_path)
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break
    print(f"  time {time.time()-t0:.1f}s  best_val_F1 {best_val_f1:.4f}")

# ------------------------------------------------------------------
# test ----------------------------------------------------------------
# ------------------------------------------------------------------
model.load_state_dict(torch.load(save_path))
test_loss, test_f1, test_preds, test_gts = run_epoch(loaders["test"], train=False)
print(f"Test Macro-F1: {test_f1:.4f}")

ed = experiment_data["bigram_only"]
ed["losses"]["test"] = test_loss
ed["metrics"]["test_f1"] = test_f1
ed["predictions"] = test_preds
ed["ground_truth"] = test_gts

# ------------------------------------------------------------------
# save ----------------------------------------------------------------
# ------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
