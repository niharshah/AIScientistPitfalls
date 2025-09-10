import os, pathlib, numpy as np, torch, matplotlib

matplotlib.use("Agg")  # headless plotting if desired later
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

# --------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", {k: len(v) for k, v in spr.items()})

# ----------------- vocabulary -------------------------------------------------
PAD, UNK = "<pad>", "<unk>"
all_chars = set(ch for ex in spr["train"] for ch in ex["sequence"])
itos = [PAD, UNK] + sorted(list(all_chars))
stoi = {c: i for i, c in enumerate(itos)}
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
max_len = 128


def encode(seq):
    ids = [stoi.get(c, stoi[UNK]) for c in seq[:max_len]]
    if len(ids) < max_len:
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.d = hf_dataset

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        row = self.d[idx]
        ids = torch.tensor(encode(row["sequence"]), dtype=torch.long)
        mask = (ids != stoi[PAD]).long()
        label = torch.tensor(row["label"], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": label}


# ----------------- models -----------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, ids, mask):
        x = self.embed(ids) + self.pos[:, : ids.size(1), :]
        x = self.enc(x, src_key_padding_mask=~mask.bool())
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.cls(pooled)


class NeuroSymbolicTransformer(nn.Module):
    def __init__(self, base: TinyTransformer, sym_hidden=128):
        super().__init__()
        self.base = base
        self.sym_head = nn.Sequential(
            nn.Linear(vocab_size, sym_hidden),
            nn.ReLU(),
            nn.Linear(sym_hidden, num_classes),
        )

    def forward(self, ids, mask):
        logits_neural = self.base(ids, mask)
        # symbolic counts (exclude PAD=0)
        counts = torch.bincount(ids.view(-1), minlength=vocab_size)
        counts = counts.unsqueeze(0).repeat(ids.size(0), 1)  # naive but quick
        # The above is CPU-heavy; vectorised alternative:
        counts = torch.zeros(ids.size(0), vocab_size, device=ids.device)
        counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float))
        counts[:, stoi[PAD]] = 0.0
        logits_sym = self.sym_head(counts)
        return logits_neural + logits_sym


# ----------------- training utils -------------------------------------------
def run_epoch(model, loader, crit, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out = model(batch["input_ids"], batch["attention_mask"])
        loss = crit(out, batch["labels"])
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(-1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# ----------------- experiment loop ------------------------------------------
experiment_data = {
    "baseline": {
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": []},
    },
    "hybrid": {"losses": {"train": [], "val": []}, "metrics": {"train": [], "val": []}},
}

batch_size = 128
epochs = 5
crit = nn.CrossEntropyLoss()

train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=batch_size)


def train_model(model_key, model):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for ep in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, crit, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_loader, crit)
        experiment_data[model_key]["losses"]["train"].append(tr_loss)
        experiment_data[model_key]["losses"]["val"].append(val_loss)
        experiment_data[model_key]["metrics"]["train"].append(tr_f1)
        experiment_data[model_key]["metrics"]["val"].append(val_f1)
        print(
            f"[{model_key}] Epoch {ep}: val_loss={val_loss:.4f} val_MacroF1={val_f1:.4f}"
        )


# baseline
baseline_model = TinyTransformer()
train_model("baseline", baseline_model)

# hybrid
hybrid_model = NeuroSymbolicTransformer(TinyTransformer())
train_model("hybrid", hybrid_model)

# ----------------- save ------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
