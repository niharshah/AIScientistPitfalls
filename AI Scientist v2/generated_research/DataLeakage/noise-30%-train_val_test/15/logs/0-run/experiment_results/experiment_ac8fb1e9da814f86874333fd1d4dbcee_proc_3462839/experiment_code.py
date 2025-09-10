import os, pathlib, random, math, time, json, numpy as np, torch
from typing import List, Dict
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ---------------- housekeeping & GPU -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- load SPR_BENCH or fallback ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


dataset_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if dataset_path.exists():
    spr = load_spr_bench(dataset_path)
else:
    print("SPR_BENCH not found, generating synthetic data …")

    def synth_split(n_rows, n_labels=5, max_len=20):
        data = {"id": [], "sequence": [], "label": []}
        alphabet = list("ABCDEXYZUVW")
        for i in range(n_rows):
            seq = "".join(random.choices(alphabet, k=random.randint(5, max_len)))
            data["id"].append(str(i))
            data["sequence"].append(seq)
            data["label"].append(random.randint(0, n_labels - 1))
        return load_dataset("json", data_files={"train": [data]}, split="train")

    spr = DatasetDict(
        {"train": synth_split(500), "dev": synth_split(100), "test": synth_split(100)}
    )

num_labels = len(set(spr["train"]["label"]))
print("Loaded dataset with", num_labels, "labels.")

# ---------------- vocab & encoding -------------------
PAD_ID = 0


def build_vocab(ds) -> Dict[str, int]:
    chars = set(ch for seq in ds["sequence"] for ch in seq)
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
    vocab["<PAD>"] = PAD_ID
    return vocab


vocab = build_vocab(spr["train"])
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq, max_len):
    ids = [vocab.get(ch, PAD_ID) for ch in seq[:max_len]]
    return ids + [PAD_ID] * (max_len - len(ids))


MAX_LEN = min(64, max(len(s) for s in spr["train"]["sequence"]))


# ---------------- dataset wrapper --------------------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, max_len):
        self.seqs, self.labels, self.max_len = (
            hf_ds["sequence"],
            hf_ds["label"],
            max_len,
        )

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor(encode(self.seqs[idx], self.max_len), dtype=torch.long)
        attn = (ids != PAD_ID).long()
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": torch.tensor(self.labels[idx]),
        }


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr[s], MAX_LEN) for s in ["train", "dev", "test"]
)


def collate(b):
    return {k: torch.stack([d[k] for d in b]) for k in b[0]}


train_loader = DataLoader(train_ds, 128, True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, 256, False, collate_fn=collate)
test_loader = DataLoader(test_ds, 256, False, collate_fn=collate)


# ---------------- model ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div_term), torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SPRTransformer(nn.Module):
    def __init__(
        self, vocab, labels, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cls = nn.Linear(d_model, labels)

    def forward(self, ids, attn_mask):
        x = self.embed(ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=(attn_mask == 0))
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.cls(x)


# ---------------- training helpers ------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    return (
        total / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ---------------- hyperparameter sweep ---------------
HEAD_VALUES = [2, 4, 8, 16]
EPOCHS = 5
experiment_data = {"nhead": {}}

for nhead in HEAD_VALUES:
    print(f"\n===== Training with nhead={nhead} =====")
    model = SPRTransformer(vocab_size, num_labels, d_model=128, nhead=nhead).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    exp_entry = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_loader, criterion)
        exp_entry["losses"]["train"].append(tr_loss)
        exp_entry["losses"]["val"].append(val_loss)
        exp_entry["metrics"]["train"].append(tr_f1)
        exp_entry["metrics"]["val"].append(val_f1)
        exp_entry["epochs"].append(epoch)
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f}"
        )

    test_loss, test_f1, test_preds, test_gts = run_epoch(model, test_loader, criterion)
    exp_entry.update(
        {
            "test_loss": test_loss,
            "test_macroF1": test_f1,
            "predictions": test_preds,
            "ground_truth": test_gts,
        }
    )
    print(f"Test : loss={test_loss:.4f} MacroF1={test_f1:.4f}")

    # store under experiment_data
    if "SPR_BENCH" not in experiment_data["nhead"]:
        experiment_data["nhead"]["SPR_BENCH"] = {}
    experiment_data["nhead"]["SPR_BENCH"][f"nhead_{nhead}"] = exp_entry

    # free GPU memory
    del model
    torch.cuda.empty_cache()

# --------------- save experiment data ---------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nSaved experiment_data.npy to", working_dir)
