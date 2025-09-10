import os, pathlib, random, math, time, json, numpy as np, torch
from typing import List, Dict
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ---------------- housekeeping & reproducibility ------------------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------- dataset utils --------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


dataset_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if dataset_path.exists():
    spr = load_spr_bench(dataset_path)
else:
    print("SPR_BENCH not found, generating synthetic fallback …")

    def synth_split(n_rows, n_labels=5, max_len=20):
        data = {"id": [], "sequence": [], "label": []}
        abc = list("ABCDEXYZUVW")
        for i in range(n_rows):
            ln = random.randint(5, max_len)
            data["id"].append(str(i))
            data["sequence"].append("".join(random.choices(abc, k=ln)))
            data["label"].append(random.randint(0, n_labels - 1))
        return load_dataset("json", data_files={"train": [data]}, split="train")

    spr = DatasetDict(
        {"train": synth_split(500), "dev": synth_split(100), "test": synth_split(100)}
    )

num_labels = len(set(spr["train"]["label"]))
print(f"Num labels: {num_labels}")

# ------------------- vocab & encoding -----------------------------------
PAD_ID = 0


def build_vocab(ds) -> Dict[str, int]:
    chars = set()
    for seq in ds["sequence"]:
        chars.update(seq)
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
    vocab["<PAD>"] = PAD_ID
    return vocab


vocab = build_vocab(spr["train"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


def encode(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(ch, PAD_ID) for ch in seq[:max_len]]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


MAX_LEN = min(64, max(len(s) for s in spr["train"]["sequence"]))


class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, max_len):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor(encode(self.seqs[idx], self.max_len))
        attn = (ids != PAD_ID).long()
        lbl = torch.tensor(self.labels[idx])
        return {"input_ids": ids, "attention_mask": attn, "labels": lbl}


def collate(b):
    return {k: torch.stack([d[k] for d in b]) for k in b[0]}


train_ds, dev_ds, test_ds = (
    (SPRtorch := SPRTorchDataset)(spr["train"], MAX_LEN),
    (SPRtorch := SPRTorchDataset)(spr["dev"], MAX_LEN),
    (SPRtorch := SPRTorchDataset)(spr["test"], MAX_LEN),
)  # noqa
train_loader = DataLoader(train_ds, 128, True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, 256, False, collate_fn=collate)
test_loader = DataLoader(test_ds, 256, False, collate_fn=collate)


# ------------------- model definition -----------------------------------
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


class SPRTransformer(nn.Module):
    def __init__(self, vocab_size, num_labels, dropout):
        super().__init__()
        d_model, nhead, num_layers, dim_ff = 128, 4, 2, 256
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(x)


# ------------------- train / eval helpers --------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(logits, -1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ------------------- hyperparameter sweep -------------------------------
dropout_grid = [0.0, 0.05, 0.1, 0.2, 0.3]
EPOCHS = 5
experiment_data = {
    "dropout_tuning": {
        "SPR_BENCH": {"results": []}  # each entry stores all info for one dropout value
    }
}

best_val_f1, best_cfg = -1.0, None
for dp in dropout_grid:
    print(f"\n=== Training with dropout={dp:.2f} ===")
    model = SPRTransformer(vocab_size, num_labels, dropout=dp).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tr_losses, tr_f1s, val_losses, val_f1s = [], [], [], []
    for epoch in range(1, EPOCHS + 1):
        tl, tf1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl, vf1, _, _ = run_epoch(model, dev_loader, criterion)
        tr_losses.append(tl)
        tr_f1s.append(tf1)
        val_losses.append(vl)
        val_f1s.append(vf1)
        print(
            f"Epoch {epoch}: train_loss={tl:.4f} val_loss={vl:.4f} "
            f"train_F1={tf1:.4f} val_F1={vf1:.4f}"
        )

    # final test evaluation
    test_loss, test_f1, preds, gts = run_epoch(model, test_loader, criterion)
    print(f"Test  : loss={test_loss:.4f} MacroF1={test_f1:.4f}")

    # store
    result = {
        "dropout": dp,
        "metrics": {"train": tr_f1s, "val": val_f1s},
        "losses": {"train": tr_losses, "val": val_losses},
        "predictions": preds,
        "ground_truth": gts,
        "test_loss": test_loss,
        "test_macroF1": test_f1,
        "epochs": list(range(1, EPOCHS + 1)),
    }
    experiment_data["dropout_tuning"]["SPR_BENCH"]["results"].append(result)

    # track best
    if max(val_f1s) > best_val_f1:
        best_val_f1 = max(val_f1s)
        best_cfg = dp

print(
    f"\nBest dropout value based on dev Macro-F1: {best_cfg} "
    f"with F1={best_val_f1:.4f}"
)

# ------------------- save experiment data -------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy to", working_dir)
