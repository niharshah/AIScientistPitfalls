import os, pathlib, random, math, numpy as np, torch
from collections import Counter
from typing import Dict, List
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ------------------- boiler-plate & device ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------- data loading ------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


dataset_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if dataset_path.exists():
    spr = load_spr_bench(dataset_path)
else:  # tiny synthetic fallback
    print("SPR_BENCH not found, creating synthetic toy data …")

    def synth(n_rows=400, max_len=18, n_labels=5):
        data = {"id": [], "sequence": [], "label": []}
        alphabet = list("ABCDEFGHIJKL")
        for i in range(n_rows):
            seq = "".join(random.choices(alphabet, k=random.randint(5, max_len)))
            data["id"].append(str(i))
            data["sequence"].append(seq)
            data["label"].append(random.randint(0, n_labels - 1))
        return load_dataset("json", data_files={"train": [data]}, split="train")

    spr = DatasetDict({"train": synth(2000), "dev": synth(400), "test": synth(400)})

num_labels = len(set(spr["train"]["label"]))
print(f"Dataset has {num_labels} labels.")

# ------------------- vocab & helpers ---------------------------------
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
print("Vocab size:", vocab_size)
id2char = {i: c for c, i in vocab.items()}


def encode(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(ch, PAD_ID) for ch in seq[:max_len]]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


MAX_LEN = min(128, max(len(s) for s in spr["train"]["sequence"]))


# ------------------- torch Datasets ----------------------------------
class SPRDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, max_len):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.max_len = max_len
        self.count_dim = vocab_size - 1

    def __len__(self):
        return len(self.seqs)

    def _count_vec(self, seq: str) -> torch.Tensor:
        cnt = Counter(seq)
        vec = torch.zeros(self.count_dim)
        for ch, n in cnt.items():
            idx = vocab.get(ch, 0)
            if idx > 0:
                vec[idx - 1] = float(n)
        return vec / len(seq)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(encode(seq, self.max_len), dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if i != PAD_ID else 0 for i in encode(seq, self.max_len)],
                dtype=torch.long,
            ),
            "symbol_counts": self._count_vec(seq).float(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


train_loader = DataLoader(
    SPRDataset(spr["train"], MAX_LEN), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"], MAX_LEN), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(spr["test"], MAX_LEN), batch_size=256, shuffle=False, collate_fn=collate
)


# ------------------- model -------------------------------------------
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


class HybridReconTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=160,
        nhead=4,
        layers=3,
        ff=256,
        sym_hidden=80,
        drop=0.15,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout=drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sym_mlp = nn.Sequential(
            nn.Linear(vocab_size - 1, sym_hidden), nn.ReLU(), nn.LayerNorm(sym_hidden)
        )
        self.classifier = nn.Linear(d_model + sym_hidden, num_labels)
        # auxiliary decoder
        self.recon = nn.Linear(d_model, vocab_size - 1)

    def forward(self, input_ids, attention_mask, symbol_counts):
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B,d)
        logits_cls = self.classifier(
            torch.cat([pooled, self.sym_mlp(symbol_counts)], dim=-1)
        )
        logits_recon = self.recon(pooled)
        return logits_cls, logits_recon


model = HybridReconTransformer(vocab_size, num_labels).to(device)
ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# ------------------- experiment store --------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ------------------- training utils ----------------------------------
def run_epoch(loader, train_mode=False, lambda_aux=0.3):
    model.train() if train_mode else model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out_cls, out_recon = model(
                batch["input_ids"], batch["attention_mask"], batch["symbol_counts"]
            )
            loss_cls = ce_loss(out_cls, batch["labels"])
            loss_recon = mse_loss(torch.sigmoid(out_recon), batch["symbol_counts"])
            loss = loss_cls + lambda_aux * loss_recon
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(out_cls, dim=-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macroF1 = f1_score(gts, preds, average="macro")
    return avg_loss, macroF1, preds, gts


# ------------------- main training loop ------------------------------
EPOCHS = 15
best_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, False)
    scheduler.step()
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  MacroF1 = {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))

# ------------------- evaluation on test ------------------------------
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best.pt"), map_location=device)
)
test_loss, test_f1, preds, gts = run_epoch(test_loader, False)
print(f"Test: loss = {test_loss:.4f}  MacroF1 = {test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
experiment_data["SPR_BENCH"]["test_loss"] = test_loss
experiment_data["SPR_BENCH"]["test_macroF1"] = test_f1
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
