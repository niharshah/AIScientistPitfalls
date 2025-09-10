import os, pathlib, random, time, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------- seed ----------
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ---------- experiment dict ----------
experiment_data = {
    "dropout_tuning": {
        "SPR_BENCH": {
            "by_dropout": {},  # metrics per dropout
            "best_dropout": None,  # chosen by highest val F1
        }
    }
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_DIR", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- vocabulary ----------
special_tokens = ["<PAD>"]
chars = set(ch for s in spr["train"]["sequence"] for ch in s)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id = stoi["<PAD>"]
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size: {vocab_size}, Num classes: {num_classes}")


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [stoi[ch] for ch in self.seqs[idx]]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    padded = pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
    )
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": padded, "label": labels}


train_ds, dev_ds, test_ds = (SPRTorchDataset(spr[s]) for s in ["train", "dev", "test"])
batch_size = 128
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)


# ---------- model ----------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, n_classes, pad, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.pre_cls_dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x, pad_mask):
        x = self.embed(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).type_as(x)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.pre_cls_dropout(pooled)
        return self.cls(pooled)


# ---------- helper ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pad_mask = batch["input_ids"] == pad_id
            logits = model(batch["input_ids"], pad_mask)
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


def train_one_dropout(dropout_rate, epochs=5, lr=1e-3):
    model = SimpleTransformer(
        vocab_size, 128, 4, 2, num_classes, pad_id, dropout_rate
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    info = {"train_loss": [], "val_loss": [], "val_f1": [], "epochs": []}
    best_f1, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            pad_mask = batch["input_ids"] == pad_id
            logits = model(batch["input_ids"], pad_mask)
            loss = criterion(logits, batch["label"])
            loss.backward()
            opt.step()
            running += loss.item() * batch["label"].size(0)
        train_loss = running / len(train_loader.dataset)
        val_loss, val_f1, _, _ = evaluate(model, dev_loader)
        print(
            f"dropout={dropout_rate:.1f} | Epoch {ep} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
        info["train_loss"].append(train_loss)
        info["val_loss"].append(val_loss)
        info["val_f1"].append(val_f1)
        info["epochs"].append(ep)
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {
                k: v.cpu() for k, v in model.state_dict().items()
            }
    # save per-dropout info
    experiment_data["dropout_tuning"]["SPR_BENCH"]["by_dropout"][dropout_rate] = info
    # load best state for final evaluation
    model.load_state_dict(best_state)
    return model, best_f1


# ---------- hyperparameter search ----------
search_space = [0.0, 0.1, 0.2, 0.3]
best_model, best_drop, best_val = None, None, -1.0
for dr in search_space:
    m, f1_val = train_one_dropout(dr)
    if f1_val > best_val:
        best_model, best_val, best_drop = m, f1_val, dr

experiment_data["dropout_tuning"]["SPR_BENCH"]["best_dropout"] = best_drop
print(f"Best dropout: {best_drop} (val_macroF1={best_val:.4f})")

# ---------- test ----------
test_loss, test_f1, preds, gts = evaluate(best_model, test_loader)
print(f"Test:  loss={test_loss:.4f}  macro_f1={test_f1:.4f}")

exp_spr = experiment_data["dropout_tuning"]["SPR_BENCH"]
exp_spr["predictions"] = preds
exp_spr["ground_truth"] = gts
exp_spr.setdefault("test_metrics", {})["loss"] = test_loss
exp_spr["test_metrics"]["macro_f1"] = test_f1

# ---------- save ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
