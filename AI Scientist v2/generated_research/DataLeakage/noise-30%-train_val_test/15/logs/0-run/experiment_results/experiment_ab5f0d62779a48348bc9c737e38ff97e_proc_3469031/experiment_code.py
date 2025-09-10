import os, pathlib, random, math, time, json, numpy as np, torch
from typing import Dict, List
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ------------------- working dir & device ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------- load SPR_BENCH (with fallback) -----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(name):  # helper
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _l(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    print("SPR_BENCH not found – generating small synthetic dataset as fallback.")

    def synth(n, max_len=20, n_labels=5):
        data = {"id": [], "sequence": [], "label": []}
        alpha = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for i in range(n):
            seq_len = random.randint(5, max_len)
            data["id"].append(str(i))
            data["sequence"].append("".join(random.choices(alpha, k=seq_len)))
            data["label"].append(random.randrange(n_labels))
        return load_dataset("json", data_files={"train": [data]}, split="train")

    spr = DatasetDict({"train": synth(800), "dev": synth(200), "test": synth(200)})

num_labels = len(set(spr["train"]["label"]))
print(f"Loaded data. #labels = {num_labels}")

# ------------------- build vocabulary ------------------------------------
PAD_ID = 0


def build_vocab(ds) -> Dict[str, int]:
    chars = set()
    for s in ds["sequence"]:
        chars.update(s)
    vocab = {ch: i + 1 for i, ch in enumerate(sorted(chars))}
    vocab["<PAD>"] = PAD_ID
    return vocab


vocab = build_vocab(spr["train"])
id2char = {i: c for c, i in vocab.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ------------------- helpers ---------------------------------------------
def encode(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(c, PAD_ID) for c in seq[:max_len]]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    return ids


MAX_LEN = min(64, max(len(s) for s in spr["train"]["sequence"]))


# ------------------- Dataset wrapper -------------------------------------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, max_len):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor(encode(self.seqs[idx], self.max_len), dtype=torch.long)
        attn = (ids != PAD_ID).long()
        # symbolic count vector (ignore PAD=0)
        counts = (
            torch.bincount(ids[ids > 0], minlength=vocab_size).clamp_max(255).float()
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "counts": counts,
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


# ------------------- Model -----------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class HybridSPRModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=128,
        nhead=4,
        num_layers=2,
        symb_dim=64,
        dim_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.trf = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.symb_mlp = nn.Sequential(
            nn.Linear(vocab_size, symb_dim), nn.ReLU(), nn.LayerNorm(symb_dim)
        )

        self.classifier = nn.Sequential(nn.Linear(d_model + symb_dim, num_labels))

    def forward(self, input_ids, attention_mask, counts):
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.trf(x, src_key_padding_mask=(attention_mask == 0))
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        s = self.symb_mlp(counts)
        feats = torch.cat([x, s], dim=-1)
        return self.classifier(feats)


model = HybridSPRModel(vocab_size, num_labels).to(device)

# ------------------- optimisation & loss ---------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ------------------- experiment data dict --------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ------------------- training / eval loops -------------------------------
def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"], batch["counts"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(logits, -1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


EPOCHS = 8
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, True)
    val_loss, val_f1, vp, vg = run_epoch(dev_loader, False)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
        f"MacroF1 train={tr_f1:.4f} val={val_f1:.4f}"
    )

# ------------------- final test ------------------------------------------
test_loss, test_f1, tp, tg = run_epoch(test_loader, False)
print(f"Test : loss={test_loss:.4f}  MacroF1={test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = tp
experiment_data["SPR_BENCH"]["ground_truth"] = tg
experiment_data["SPR_BENCH"]["test_loss"] = test_loss
experiment_data["SPR_BENCH"]["test_macroF1"] = test_f1

# ------------------- save -------------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy to", working_dir)
