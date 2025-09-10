import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, random, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, DatasetDict

# ---------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------- reproducibility ----------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------- load data ----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(DATA_PATH)


# --------------- vocab --------------------
def build_vocab(dsets):
    chars = set()
    for split in dsets.values():
        for s in split["sequence"]:
            chars.update(s)
    return {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 = PAD


vocab = build_vocab(spr)
vocab_size = len(vocab) + 1


def enc(seq):
    return [vocab[c] for c in seq]


# --------------- dataset ------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def collate(batch):
    seqs, labels = zip(*batch)
    enc_seqs = [torch.tensor(enc(s), dtype=torch.long) for s in seqs]
    max_len = max(map(len, enc_seqs))
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attn_mask = torch.zeros_like(input_ids)
    stats = torch.zeros(len(batch), vocab_size, dtype=torch.float32)
    for i, seq in enumerate(enc_seqs):
        L = len(seq)
        input_ids[i, :L] = seq
        attn_mask[i, :L] = 1
        counts = torch.bincount(seq, minlength=vocab_size).float() / L
        stats[i] = counts
    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "stats": stats,
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


bs = 256
train_loader = DataLoader(
    SPRDataset(spr["train"]), bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(SPRDataset(spr["dev"]), bs, collate_fn=collate)
test_loader = DataLoader(SPRDataset(spr["test"]), bs, collate_fn=collate)


# --------------- model --------------------
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class StatsTransformerSPR(nn.Module):
    def __init__(self, vocab, stats_dim, d_model=128, nhead=4, layers=2, stats_proj=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pe = PosEnc(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.stats_proj = nn.Sequential(
            nn.LayerNorm(stats_dim), nn.Linear(stats_dim, stats_proj), nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(d_model + stats_proj, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, stats):
        x = self.embed(input_ids) * math.sqrt(self.embed.embedding_dim)
        x = self.pe(x)
        key_padding = attention_mask == 0
        enc = self.encoder(x, src_key_padding_mask=key_padding)
        pooled = (enc * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        ).clamp(1e-9)
        s = self.stats_proj(stats)
        concat = torch.cat([pooled, s], dim=-1)
        return self.out(concat).squeeze(1)


# -------------- imbalance handling ---------
train_labels = np.array(spr["train"]["label"])
pos_weight = torch.tensor(
    (len(train_labels) - train_labels.sum()) / train_labels.sum(), dtype=torch.float32
).to(device)


# -------------- utils ----------------------
def evaluate(model, loader):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    tot, preds, labs = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"], batch["stats"])
            loss = crit(logits, batch["labels"])
            tot += loss.item() * batch["labels"].size(0)
            preds.append((logits.sigmoid() > 0.5).cpu().numpy())
            labs.append(batch["labels"].cpu().numpy())
    preds, labs = np.concatenate(preds), np.concatenate(labs)
    return (
        tot / len(loader.dataset),
        matthews_corrcoef(labs, preds),
        f1_score(labs, preds, average="macro"),
        preds,
        labs,
    )


class EarlyStop:
    def __init__(self, patience=4, delta=1e-4):
        self.best = None
        self.p = patience
        self.d = delta
        self.c = 0

    def step(self, val):
        if self.best is None or val > self.best + self.d:
            self.best = val
            self.c = 0
            return False
        self.c += 1
        return self.c >= self.p


# -------------- experiment log -------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------- training loop --------------
def train(epochs=15, lr=2e-3):
    model = StatsTransformerSPR(vocab_size, vocab_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    es = EarlyStop(3)
    best, best_mcc = None, -1
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optim.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"], batch["stats"])
            loss = crit(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item() * batch["labels"].size(0)
        sched.step()
        tr_loss = running / len(train_loader.dataset)
        _, tr_mcc, _, _, _ = evaluate(model, train_loader)
        val_loss, val_mcc, _, _, _ = evaluate(model, dev_loader)
        print(f"Epoch {ep}: validation_loss = {val_loss:.4f}, val_MCC = {val_mcc:.4f}")
        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_mcc)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_mcc)
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best = model.state_dict()
        if es.step(val_mcc):
            print("Early stopping")
            break
    model.load_state_dict(best)
    tloss, tmcc, tf1, preds, labs = evaluate(model, test_loader)
    print(f"Test MCC = {tmcc:.4f} | Test Macro-F1 = {tf1:.4f}")
    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(labs)


train()

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
