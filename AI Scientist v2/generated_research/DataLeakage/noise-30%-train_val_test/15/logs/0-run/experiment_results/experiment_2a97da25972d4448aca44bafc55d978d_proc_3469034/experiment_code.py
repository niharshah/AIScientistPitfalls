import os, pathlib, random, math, numpy as np, torch, itertools
from collections import Counter, defaultdict
from typing import Dict, List
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ------------------------------------------------------------
# working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# dataset helper ------------------------------------------------
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
else:
    print("SPR_BENCH not found – using synthetic toy data")

    def synth(n_rows=500, max_len=18, n_labels=6):
        data = {"id": [], "sequence": [], "label": []}
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for i in range(n_rows):
            seq = "".join(random.choices(alphabet, k=random.randint(5, max_len)))
            data["id"].append(str(i))
            data["sequence"].append(seq)
            data["label"].append(random.randint(0, n_labels - 1))
        return load_dataset("json", data_files={"train": [data]}, split="train")

    spr = DatasetDict({"train": synth(1500), "dev": synth(300), "test": synth(300)})

num_labels = len(set(spr["train"]["label"]))
print(f"labels: {num_labels}")

# ------------------------ vocabulation -----------------------
PAD_ID = 0


def build_char_vocab(ds) -> Dict[str, int]:
    chars = set()
    for s in ds["sequence"]:
        chars |= set(s)
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
    vocab["<PAD>"] = PAD_ID
    return vocab


vocab = build_char_vocab(spr["train"])
vocab_size = len(vocab)
id2char = {i: c for c, i in vocab.items()}
print("Char vocab:", vocab_size)


# --------------- build top-K bigram vocabulary ---------------
def get_top_bigrams(seqs: List[str], k: int = 200):
    cnt = Counter()
    for s in seqs:
        cnt.update(["".join(pair) for pair in zip(s, s[1:])])
    bigrams = [b for b, _ in cnt.most_common(k)]
    return {bg: i for i, bg in enumerate(bigrams)}


BIGRAM_K = 200
bigram2idx = get_top_bigrams(spr["train"]["sequence"], BIGRAM_K)
bigram_dim = len(bigram2idx)
print("Bigram dim:", bigram_dim)

# -------------------- encode utilities -----------------------
MAX_LEN = min(128, max(len(s) for s in spr["train"]["sequence"]))


def encode_chars(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(ch, PAD_ID) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    return ids


# -------------------- torch dataset --------------------------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, max_len):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.max_len = max_len
        self.char_dim = vocab_size - 1
        self.bigram_dim = bigram_dim

    def __len__(self):
        return len(self.seqs)

    def _sym_vec(self, seq: str) -> torch.Tensor:
        char_cnt = Counter(seq)
        bg_cnt = Counter(["".join(p) for p in zip(seq, seq[1:])])
        vec = torch.zeros(self.char_dim + self.bigram_dim, dtype=torch.float)
        for ch, n in char_cnt.items():
            idx = vocab.get(ch, 0)
            if idx > 0:
                vec[idx - 1] = float(n)
        for bg, n in bg_cnt.items():
            idx = bigram2idx.get(bg, None)
            if idx is not None:
                vec[self.char_dim + idx] = float(n)
        vec = vec / len(seq)  # normalise
        return vec

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = torch.tensor(encode_chars(seq, self.max_len), dtype=torch.long)
        attn = (ids != PAD_ID).long()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        sym = self._sym_vec(seq)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "symbolic": sym,
            "labels": label,
        }


def collate(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


train_ds = SPRTorchDataset(spr["train"], MAX_LEN)
dev_ds = SPRTorchDataset(spr["dev"], MAX_LEN)
test_ds = SPRTorchDataset(spr["test"], MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# -------------------- model ----------------------------------
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
        return x + self.pe[:, : x.size(1), :]


class GatedHybrid(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=128,
        nhead=4,
        n_layers=2,
        ff_dim=256,
        sym_hidden=128,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff_dim, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # symbolic branch
        self.sym_mlp = nn.Sequential(
            nn.Linear((vocab_size - 1) + bigram_dim, sym_hidden),
            nn.ReLU(),
            nn.LayerNorm(sym_hidden),
        )
        self.sym_proj = nn.Linear(sym_hidden, d_model)
        # gating
        self.gate_fc = nn.Linear(d_model * 2, 1)
        # classifier
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask, symbolic):
        # transformer branch
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=(attention_mask == 0))
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B,d_model)
        # symbolic branch
        s = self.sym_mlp(symbolic)
        s = self.sym_proj(s)  # (B,d_model)
        # gating
        g = torch.sigmoid(self.gate_fc(torch.cat([x, s], dim=-1)))  # (B,1)
        fused = g * x + (1 - g) * s
        logits = self.cls(fused)
        return logits


model = GatedHybrid(vocab_size, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

# -------------------- experiment logging ---------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------- training / eval funcs ------------------
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["symbolic"]
            )
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# -------------------- training loop --------------------------
EPOCHS = 6
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, train=False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  MacroF1 = {val_f1:.4f}")

# -------------------- final test -----------------------------
test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, train=False)
print(f"Test: loss = {test_loss:.4f}  MacroF1 = {test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
experiment_data["SPR_BENCH"]["test_loss"] = test_loss
experiment_data["SPR_BENCH"]["test_macroF1"] = test_f1

# -------------------- save log -------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Metrics saved at", os.path.join(working_dir, "experiment_data.npy"))
