# -------------------------------------------------------------
# Learning-rate hyper-parameter sweep on SPR_BENCH
# -------------------------------------------------------------
import os, pathlib, time, json, numpy as np, torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from collections import Counter
from typing import List, Dict
from datasets import load_dataset, DatasetDict

# ---------------- Experiment-data container ------------------
experiment_data = {"learning_rate": {"SPR_BENCH": {}}}

# -------------- Basic set-up ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------- Dataset utilities ----------------------------
def resolve_spr_path() -> pathlib.Path:
    cand = []
    if "SPR_BENCH_PATH" in os.environ:
        cand.append(os.environ["SPR_BENCH_PATH"])
    cwd = pathlib.Path.cwd()
    cand += [
        cwd / "SPR_BENCH",
        cwd.parent / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for c in cand:
        p = pathlib.Path(c)
        if (p / "train.csv").exists():
            print(f"Found SPR_BENCH dataset at {p.resolve()}")
            return p.resolve()
    raise FileNotFoundError(
        "SPR_BENCH not found. Set SPR_BENCH_PATH or place csvs in ./SPR_BENCH"
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    ds = DatasetDict()
    ds["train"] = _load("train.csv")
    ds["dev"] = _load("dev.csv")
    ds["test"] = _load("test.csv")
    return ds


spr_root = resolve_spr_path()
spr = load_spr_bench(spr_root)
print("Loaded splits:", {k: len(v) for k, v in spr.items()})


# -------------------- Vocab -----------------------------------
def tokenize(seq: str) -> List[str]:
    return seq.strip().split()


all_tokens = [tok for seq in spr["train"]["sequence"] for tok in tokenize(seq)]
vocab_counter = Counter(all_tokens)
vocab = ["<PAD>", "<UNK>"] + sorted(vocab_counter)
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = stoi["<PAD>"], stoi["<UNK>"]

all_labels = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(all_labels)}


def encode(seq: str) -> List[int]:
    return [stoi.get(tok, unk_idx) for tok in tokenize(seq)]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lengths = [len(x["input_ids"]) for x in batch]
    maxlen = max(lengths)
    input_ids = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, : len(seq)] = seq
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "label": labels}


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# -------------------- Model -----------------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_labels, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(emb_dim, num_labels)
        self.pad = pad_idx

    def forward(self, x):
        mask = (x != self.pad).unsqueeze(-1)
        emb = self.emb(x)
        mean = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(mean))


criterion = nn.CrossEntropyLoss()


# ------------- Optional auxiliary metrics ---------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [w0 if t == p else 0 for w0, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [w0 if t == p else 0 for w0, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0


# ------------- Train for one LR --------------------------------
def train_with_lr(lr: float, num_epochs: int = 5):
    lr_key = f"lr_{lr}"
    exp_slot = {
        "metrics": {"train_macroF1": [], "val_macroF1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    model = MeanPoolClassifier(len(vocab), 64, len(all_labels), pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        # training
        model.train()
        tr_loss, tr_preds, tr_trues = 0.0, [], []
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * batch["label"].size(0)
            tr_preds.extend(logits.argmax(1).cpu().numpy())
            tr_trues.extend(batch["label"].cpu().numpy())
        tr_loss /= len(train_loader.dataset)
        tr_macro = f1_score(tr_trues, tr_preds, average="macro")
        exp_slot["losses"]["train"].append(tr_loss)
        exp_slot["metrics"]["train_macroF1"].append(tr_macro)

        # validation
        model.eval()
        v_loss, v_preds, v_trues = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                v_loss += loss.item() * batch["label"].size(0)
                v_preds.extend(logits.argmax(1).cpu().numpy())
                v_trues.extend(batch["label"].cpu().numpy())
        v_loss /= len(val_loader.dataset)
        v_macro = f1_score(v_trues, v_preds, average="macro")
        exp_slot["losses"]["val"].append(v_loss)
        exp_slot["metrics"]["val_macroF1"].append(v_macro)
        exp_slot["timestamps"].append(time.time())
        print(
            f"[LR {lr}] Epoch {epoch}: ValLoss={v_loss:.4f}  ValMacroF1={v_macro:.4f}"
        )
    # store preds/gt of last epoch
    exp_slot["predictions"] = v_preds
    exp_slot["ground_truth"] = v_trues
    # optional weighted accuracies
    swa = shape_weighted_accuracy(spr["dev"]["sequence"], v_trues, v_preds)
    cwa = color_weighted_accuracy(spr["dev"]["sequence"], v_trues, v_preds)
    exp_slot["shape_weighted_acc"] = swa
    exp_slot["color_weighted_acc"] = cwa
    print(f"[LR {lr}] Dev SWA={swa:.4f} | CWA={cwa:.4f}")
    experiment_data["learning_rate"]["SPR_BENCH"][lr_key] = exp_slot


# ------------------ Hyper-parameter sweep ----------------------
lrs_to_try = [5e-4, 1e-3, 2e-3]
for lr in lrs_to_try:
    train_with_lr(lr, num_epochs=5)

# ------------------ Save everything ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all results to", os.path.join(working_dir, "experiment_data.npy"))
