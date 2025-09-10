import os, pathlib, random, math, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

# ---------- experiment data skeleton ----------
experiment_data = {
    "batch_size": {"SPR_BENCH": {}}  # each batch size will be inserted here
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- deterministic helpers ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()

# ---------- dataset loader ----------
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper to share cache
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

# ---------- vocabulary ----------
special_tokens = ["<PAD>"]
chars = set(ch for seq in spr["train"]["sequence"] for ch in seq)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id = stoi["<PAD>"]
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size {vocab_size} , classes {num_classes}")


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[ch] for ch in self.seqs[idx]], dtype=torch.long)
        return {"input_ids": ids, "label": torch.tensor(self.labels[idx])}


def collate_fn(batch):
    inputs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    return {"input_ids": padded, "label": labels}


train_ds, dev_ds, test_ds = (SPRTorchDataset(spr[s]) for s in ["train", "dev", "test"])


# ---------- model ----------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, nclass, pad):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.cls = nn.Linear(d_model, nclass)

    def forward(self, x, pad_mask):
        x = self.embed(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).type_as(x)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.cls(pooled)


# ---------- training & evaluation helpers ----------
def evaluate(model, loader, loss_fn):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch["label"]
            logits = model(batch["input_ids"], batch["input_ids"] == pad_id)
            loss = loss_fn(logits, y)
            tot_loss += loss.item() * y.size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(y.cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


def train_one_setting(bs, epochs=5, lr=1e-3, d_model=128):
    print(f"\n--- Training with batch_size={bs} ---")
    loaders = {
        "train": DataLoader(train_ds, bs, True, collate_fn=collate_fn),
        "dev": DataLoader(dev_ds, bs, False, collate_fn=collate_fn),
        "test": DataLoader(test_ds, bs, False, collate_fn=collate_fn),
    }
    model = SimpleTransformer(vocab_size, d_model, 4, 2, num_classes, pad_id).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    logs = {"metrics": {"train_loss": [], "val_loss": [], "val_f1": []}, "epochs": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in loaders["train"]:
            batch = {k: v.to(device) for k, v in batch.items()}
            pad_mask = batch["input_ids"] == pad_id
            opt.zero_grad()
            out = model(batch["input_ids"], pad_mask)
            loss = loss_fn(out, batch["label"])
            loss.backward()
            opt.step()
            running += loss.item() * batch["label"].size(0)
        tr_loss = running / len(loaders["train"].dataset)
        val_loss, val_f1, *_ = evaluate(model, loaders["dev"], loss_fn)
        print(
            f"Epoch {epoch} | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | val_f1 {val_f1:.4f}"
        )
        logs["metrics"]["train_loss"].append(tr_loss)
        logs["metrics"]["val_loss"].append(val_loss)
        logs["metrics"]["val_f1"].append(val_f1)
        logs["epochs"].append(epoch)

    test_loss, test_f1, preds, gts = evaluate(model, loaders["test"], loss_fn)
    print(f"Test: loss {test_loss:.4f} | macro_f1 {test_f1:.4f}")
    logs.update(
        {
            "test_loss": test_loss,
            "test_f1": test_f1,
            "predictions": preds,
            "ground_truth": gts,
        }
    )
    return logs


# ---------- hyperparameter sweep ----------
batch_sizes = [32, 64, 128, 256]
for bs in batch_sizes:
    set_seed()  # reset seed for fairness
    logs = train_one_setting(bs)
    experiment_data["batch_size"]["SPR_BENCH"][str(bs)] = logs

# ---------- save ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to", os.path.join(working_dir, "experiment_data.npy"))
