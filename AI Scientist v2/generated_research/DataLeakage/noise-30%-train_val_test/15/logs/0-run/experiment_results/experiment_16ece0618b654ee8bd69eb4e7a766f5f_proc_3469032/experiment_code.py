import os, pathlib, random, math, numpy as np, torch, time
from collections import Counter
from typing import List, Dict
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------
# mandatory working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# 1. Load SPR_BENCH or synthetic fallback
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


dataset_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if dataset_path.exists():
    spr = load_spr_bench(dataset_path)
else:
    print("SPR_BENCH not found, generating tiny synthetic data …")

    def synth(n=500):
        data = {"id": [], "sequence": [], "label": []}
        alphabet = list("ABCDELMNOPQR")
        for i in range(n):
            seq_len = random.randint(5, 18)
            seq = "".join(random.choices(alphabet, k=seq_len))
            data["id"].append(str(i))
            data["sequence"].append(seq)
            data["label"].append(random.randint(0, 4))
        return load_dataset("json", data_files={"train": [data]}, split="train")

    spr = DatasetDict({"train": synth(800), "dev": synth(200), "test": synth(200)})

num_labels = len(set(spr["train"]["label"]))
print(f"n_labels = {num_labels}")

# ---------------------------------------------------------------------
# 2. Build char vocab
PAD_ID = 0


def build_vocab(ds) -> Dict[str, int]:
    chars = set()
    for s in ds["sequence"]:
        chars.update(s)
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
    vocab["<PAD>"] = PAD_ID
    return vocab


vocab = build_vocab(spr["train"])
vocab_size = len(vocab)
MAX_LEN = min(128, max(len(s) for s in spr["train"]["sequence"]))
print(f"vocab_size={vocab_size}, MAX_LEN={MAX_LEN}")


def encode(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(ch, PAD_ID) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    return ids


# ---------------------------------------------------------------------
# 3. Torch dataset with enriched symbolic features
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, max_len):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.max_len = max_len
        self.count_dim = vocab_size - 1

    def _count_vec(self, seq: str) -> torch.Tensor:
        cnt = Counter(seq)
        vec = torch.zeros(self.count_dim, dtype=torch.float32)
        for ch, n in cnt.items():
            idx = vocab.get(ch, 0)
            if idx > 0:
                vec[idx - 1] = n / len(seq)
        return vec

    def _extra_feats(self, seq: str) -> torch.Tensor:
        length = len(seq) / self.max_len
        uniq = len(set(seq)) / self.count_dim
        # Shannon entropy scaled to [0,1]
        p = np.array(list(Counter(seq).values())) / len(seq)
        entropy = -(p * np.log2(p + 1e-9)).sum() / math.log2(self.count_dim + 1e-9)
        return torch.tensor([length, uniq, entropy], dtype=torch.float32)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(encode(seq, self.max_len), dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if i != PAD_ID else 0 for i in encode(seq, self.max_len)],
                dtype=torch.long,
            ),
            "symbol_counts": self._count_vec(seq),
            "extra_feats": self._extra_feats(seq),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr[s], MAX_LEN) for s in ["train", "dev", "test"]
)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = torch.utils.data.DataLoader(
    dev_ds, batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=256, shuffle=False, collate_fn=collate
)


# ---------------------------------------------------------------------
# 4. Model: Transformer + gated symbolic vector
class PositionalEncoding(torch.nn.Module):
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


class Model(torch.nn.Module):
    def __init__(
        self, vocab_size, num_labels, d_model=128, nhead=4, n_layers=2, ff=256, sym_h=64
    ):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout=0.1, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, n_layers)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        # symbolic branch
        self.sym_mlp = torch.nn.Sequential(
            torch.nn.Linear(vocab_size - 1 + 3, sym_h),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        # gating
        self.gate = torch.nn.Linear(d_model + sym_h, 1)
        self.classifier = torch.nn.Linear(d_model + sym_h, num_labels)

    def forward(self, input_ids, attention_mask, symbol_counts, extra_feats):
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=(attention_mask == 0))
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)  # (B,d_model)
        sym = torch.cat([symbol_counts, extra_feats], dim=-1)
        sym = self.sym_mlp(sym)  # (B,sym_h)
        concat = torch.cat([x, sym], dim=-1)  # (B,d_model+sym_h)
        g = torch.sigmoid(self.gate(concat))  # (B,1)
        mixed = torch.cat([x * g, sym * (1 - g)], dim=-1)  # simple gated fusion
        return self.classifier(mixed)  # logits


model = Model(vocab_size, num_labels).to(device)


# ---------------------------------------------------------------------
# 5. Loss, optimiser, scheduler
class LabelSmoothingCE(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


criterion = LabelSmoothingCE()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

# ---------------------------------------------------------------------
# 6. experiment tracking dict
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------------------------------------------------------------------
# 7. Train / eval loops
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["symbol_counts"],
            batch["extra_feats"],
        )
        loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


best_val = -1
patience, wait = 3, 0
EPOCHS = 15
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, False)
    scheduler.step()
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f}  MacroF1={val_f1:.4f}  time={time.time()-t0:.1f}s"
    )
    if val_f1 > best_val:
        best_val, wait = val_f1, 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ---------------------------------------------------------------------
# 8. Test evaluation
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best.pt"), map_location=device)
)
test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, False)
print(f"Test: loss={test_loss:.4f}  MacroF1={test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
experiment_data["SPR_BENCH"]["test_loss"] = test_loss
experiment_data["SPR_BENCH"]["test_macroF1"] = test_f1

# ---------------------------------------------------------------------
# 9. Save experiment data
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
