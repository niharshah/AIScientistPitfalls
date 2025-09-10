import os, pathlib, math, time, random, json
import torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset

# -------------------- utility & bookkeeping --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "learning_rate_sweep": {
        "SPR_BENCH": {
            "lr_vals": [],
            "metrics": {"train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- load SPR_BENCH --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):  # helper to load csv files
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


for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder missing.")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------- vocab & label --------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = set(ch for seq in spr["train"]["sequence"] for ch in seq)
vocab = [PAD, UNK] + sorted(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(vocab)

labels = sorted(list(set(spr["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)

MAX_LEN = 64


def encode_seq(seq):
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:MAX_LEN]]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def encode_label(l):
    return label2id[l]


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labs = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_seq(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(encode_label(self.labs[idx]), dtype=torch.long),
        }


batch_size = 128
train_dl = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=batch_size, shuffle=True
)
val_dl = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_dl = DataLoader(SPRTorchDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# -------------------- model --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, layers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 256, 0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        mask = ids == 0
        x = self.pos(self.embed(ids))
        x = self.enc(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(x)


# -------------------- train/eval helpers --------------------
def run_epoch(model, dl, criterion, opt=None, scheduler=None):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss, preds, labels = 0.0, [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.append(logits.argmax(-1).cpu())
        labels.append(batch["labels"].cpu())
    if scheduler and train:
        scheduler.step()
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return total_loss / len(dl.dataset), f1_score(labels, preds, average="macro")


def train_with_lr(lr, epochs=8, use_scheduler=True):
    model = TransformerClassifier(vocab_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        if use_scheduler
        else None
    )
    lr_train_losses, lr_val_losses, lr_train_f1s, lr_val_f1s = [], [], [], []
    for ep in range(1, epochs + 1):
        tr_loss, tr_f1 = run_epoch(model, train_dl, criterion, optimizer, scheduler)
        val_loss, val_f1 = run_epoch(model, val_dl, criterion)
        lr_train_losses.append(tr_loss)
        lr_val_losses.append(val_loss)
        lr_train_f1s.append(tr_f1)
        lr_val_f1s.append(val_f1)
        print(f"LR={lr:.1e}  Epoch {ep}/{epochs}  valF1={val_f1:.4f}")
    return model, {
        "train_loss": lr_train_losses,
        "val_loss": lr_val_losses,
        "train_f1": lr_train_f1s,
        "val_f1": lr_val_f1s,
    }


# -------------------- hyper-parameter sweep --------------------
lr_candidates = [3e-4, 5e-4, 1e-3, 2e-3]
best_val, best_model, best_lr = -1, None, None

for lr in lr_candidates:
    model, res = train_with_lr(lr)
    experiment_data["learning_rate_sweep"]["SPR_BENCH"]["lr_vals"].append(lr)
    experiment_data["learning_rate_sweep"]["SPR_BENCH"]["losses"]["train"].append(
        res["train_loss"]
    )
    experiment_data["learning_rate_sweep"]["SPR_BENCH"]["losses"]["val"].append(
        res["val_loss"]
    )
    experiment_data["learning_rate_sweep"]["SPR_BENCH"]["metrics"]["train_f1"].append(
        res["train_f1"]
    )
    experiment_data["learning_rate_sweep"]["SPR_BENCH"]["metrics"]["val_f1"].append(
        res["val_f1"]
    )
    experiment_data["learning_rate_sweep"]["SPR_BENCH"]["epochs"].append(
        list(range(1, len(res["val_f1"]) + 1))
    )
    if res["val_f1"][-1] > best_val:
        best_val = res["val_f1"][-1]
        best_model = model
        best_lr = lr

print(f"Best LR={best_lr:.1e} with validation F1={best_val:.4f}")

# -------------------- final test eval --------------------
criterion = nn.CrossEntropyLoss()
test_loss, test_f1 = run_epoch(best_model, test_dl, criterion)
print(f"Test macro-F1 with best LR: {test_f1:.4f}")

# collect test predictions
best_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = best_model(batch["input_ids"])
        all_preds.append(logits.argmax(-1).cpu())
        all_labels.append(batch["labels"].cpu())
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

experiment_data["learning_rate_sweep"]["SPR_BENCH"]["predictions"] = all_preds.tolist()
experiment_data["learning_rate_sweep"]["SPR_BENCH"][
    "ground_truth"
] = all_labels.tolist()

# -------------------- save --------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
