import os, math, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- workspace ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- locate SPR_BENCH directory ----------
def find_spr_bench_dir() -> pathlib.Path:
    # 1) environment variable has priority
    env_path = os.getenv("SPR_BENCH_DIR")
    if env_path and (pathlib.Path(env_path) / "train.csv").exists():
        return pathlib.Path(env_path)
    # 2) walk up from cwd looking for SPR_BENCH/train.csv
    cur = pathlib.Path(os.getcwd()).resolve()
    for parent in [cur] + list(cur.parents):
        candidate = parent / "SPR_BENCH"
        if (candidate / "train.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Cannot locate SPR_BENCH dataset. "
        "Set SPR_BENCH_DIR env var or place SPR_BENCH folder in/above the working directory."
    )


DATA_PATH = find_spr_bench_dir()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ---------- load SPR benchmark ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",  # csv already contains only one split
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


spr = load_spr_bench(DATA_PATH)
print({split: len(ds) for split, ds in spr.items()})

# ---------- vocabulary ----------
PAD, UNK = 0, 1
vocab = {
    char: idx
    for idx, char in enumerate(
        sorted({c for s in spr["train"]["sequence"] for c in s}), start=2
    )
}
vocab_size = len(vocab) + 2


def encode(seq: str):
    return [vocab.get(c, UNK) for c in seq]


max_len = max(len(seq) for seq in spr["train"]["sequence"])
print(f"vocab_size={vocab_size}, max_len={max_len}")


# ---------- datasets ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return encode(self.seqs[idx]), self.labels[idx]


def collate(batch):
    seqs, labels = zip(*batch)
    lens = [len(s) for s in seqs]
    max_l = max(lens)
    padded = torch.full((len(seqs), max_l), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return {"input_ids": padded, "labels": torch.tensor(labels, dtype=torch.long)}


train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SPRModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, nlayers=2, num_classes=None):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be provided")
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len=max_len + 10)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 256, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        h = self.transformer(x)
        return self.cls(h[:, 0])  # first token as CLS


num_classes = len(set(spr["train"]["label"]))
model = SPRModel(vocab_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------- helpers ----------
def run_epoch(loader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        preds = logits.argmax(-1)
        total += preds.size(0)
        correct += (preds == batch["labels"]).sum().item()
        loss_sum += loss.item() * preds.size(0)
    return loss_sum / total, correct / total


# ---------- training loop ----------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(dev_loader, train=False)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} "
        f"train_acc={train_acc*100:.2f}% | val_loss={val_loss:.4f} "
        f"val_acc={val_acc*100:.2f}% | time={time.time()-t0:.1f}s"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

# ---------- test evaluation ----------
test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"\nTest set: loss={test_loss:.4f} accuracy={test_acc*100:.2f}%")

# gather predictions
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        preds = logits.argmax(-1).cpu().numpy()
        experiment_data["SPR_BENCH"]["predictions"].extend(preds.tolist())
        experiment_data["SPR_BENCH"]["ground_truth"].extend(
            batch["labels"].cpu().numpy().tolist()
        )

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {working_dir}/experiment_data.npy")
