import os, pathlib, random, math, time, numpy as np, torch
from collections import Counter
from typing import Dict, List
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ---------- set up working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load SPR_BENCH or synthetic fallback ----------
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


dataset_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if dataset_path.exists():
    spr = load_spr_bench(dataset_path)
else:  # tiny synthetic fallback
    print("SPR_BENCH not found – generating synthetic data.")

    def synth_split(n, n_labels=5, max_len=20):
        data = {"id": [], "sequence": [], "label": []}
        vocab = list("ABCDEFXYZUV")
        for i in range(n):
            seq = "".join(random.choices(vocab, k=random.randint(5, max_len)))
            data["id"].append(str(i))
            data["sequence"].append(seq)
            data["label"].append(random.randint(0, n_labels - 1))
        return load_dataset("json", data_files={"train": [data]}, split="train")

    spr = DatasetDict(
        {"train": synth_split(500), "dev": synth_split(100), "test": synth_split(100)}
    )
num_labels = len(set(spr["train"]["label"]))
print(f"Dataset loaded with {num_labels} labels.")

# ---------- build char vocabulary ----------
PAD_ID = 0


def build_vocab(ds) -> Dict[str, int]:
    chars = set()
    for s in ds["sequence"]:
        chars.update(s)
    v = {c: i + 1 for i, c in enumerate(sorted(chars))}
    v["<PAD>"] = PAD_ID
    return v


vocab = build_vocab(spr["train"])
vocab_size = len(vocab)


# ---------- n-gram vocabulary (bigrams & trigrams) ----------
def top_ngrams(ds, top_k=256):
    cnt = Counter()
    for seq in ds["sequence"]:
        for n in (2, 3):
            cnt.update(seq[i : i + n] for i in range(len(seq) - n + 1))
    most = [ng for ng, _ in cnt.most_common(top_k)]
    return {ng: i for i, ng in enumerate(most)}


ng_vocab = top_ngrams(spr["train"], top_k=256)
ng_dim = len(ng_vocab)
print(f"Character vocab={vocab_size}, n-gram vocab={ng_dim}")


# ---------- helpers ----------
def encode_seq(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(ch, PAD_ID) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    return ids


def encode_ngrams(seq: str) -> np.ndarray:
    vec = np.zeros(ng_dim, dtype=np.float32)
    for n in (2, 3):
        for i in range(len(seq) - n + 1):
            ng = seq[i : i + n]
            if ng in ng_vocab:
                vec[ng_vocab[ng]] += 1.0
    # optional normalisation:
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec


MAX_LEN = min(64, max(len(s) for s in spr["train"]["sequence"]))


# ---------- torch dataset ----------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, max_len):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.max_len = max_len
        # pre-compute n-gram vectors for speed
        self.ngram_feats = [encode_ngrams(s) for s in self.seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = torch.tensor(encode_seq(self.seqs[idx], self.max_len), dtype=torch.long)
        attn = (ids != PAD_ID).long()
        ng = torch.tensor(self.ngram_feats[idx], dtype=torch.float)
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "ngram_feats": ng,
            "labels": lbl,
        }


def collate(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


train_ds = SPRTorchDataset(spr["train"], MAX_LEN)
dev_ds = SPRTorchDataset(spr["dev"], MAX_LEN)
test_ds = SPRTorchDataset(spr["test"], MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------- model ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HybridModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        ngram_size,
        num_labels,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        drop=0.2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, drop, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ng_proj = nn.Sequential(
            nn.Linear(ngram_size, 64), nn.ReLU(), nn.Dropout(drop)
        )
        self.classifier = nn.Linear(d_model + 64, num_labels)

    def forward(self, input_ids, attention_mask, ngram_feats):
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=(attention_mask == 0))
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        ng = self.ng_proj(ngram_feats)
        logits = self.classifier(torch.cat([x, ng], dim=-1))
        return logits


model = HybridModel(vocab_size, ng_dim, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ---------- experiment tracking ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------- training / evaluation ----------
def run_epoch(loader, train_flag: bool):
    model.train() if train_flag else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train_flag):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["ngram_feats"]
            )
            loss = criterion(logits, batch["labels"])
            if train_flag:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


EPOCHS = 7
best_val = -1
patience = 2
stalls = 0
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f}"
    )
    # simple early stopping
    if val_f1 > best_val:
        best_val = val_f1
        stalls = 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
    else:
        stalls += 1
        if stalls >= patience:
            print("Early stopping.")
            break

# ---------- test evaluation ----------
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best.pt"), map_location=device)
)
test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, False)
print(f"Test : loss={test_loss:.4f}  MacroF1={test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
experiment_data["SPR_BENCH"]["test_loss"] = test_loss
experiment_data["SPR_BENCH"]["test_macroF1"] = test_f1
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy.")
