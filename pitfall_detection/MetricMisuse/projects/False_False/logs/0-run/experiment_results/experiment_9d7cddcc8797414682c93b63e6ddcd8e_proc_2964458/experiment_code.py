import os, pathlib, time, json, numpy as np, torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from collections import Counter
from typing import List
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------ #
#              EXPERIMENT BOOK-KEEPING DICT                    #
# ------------------------------------------------------------ #
experiment_data = {
    "batch_size": {  # hyperparameter tuning type
        "SPR_BENCH": {}  # we will add one sub-dict per batch size tried
    }
}

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------ #
#                       DEVICE                                 #
# ------------------------------------------------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------ #
#                 DATASET LOADING HELPERS                      #
# ------------------------------------------------------------ #
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
    for p in cand:
        p = pathlib.Path(p)
        if (p / "train.csv").exists():
            print("Found SPR_BENCH at", p.resolve())
            return p.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = _l("train.csv"), _l("dev.csv"), _l("test.csv")
    return d


spr_root = resolve_spr_path()
spr = load_spr_bench(spr_root)
print("Loaded splits:", {k: len(v) for k, v in spr.items()})


# ------------------------------------------------------------ #
#                 VOCAB & ENCODING                             #
# ------------------------------------------------------------ #
def tokenize(seq: str) -> List[str]:
    return seq.strip().split()


all_tokens = [tok for seq in spr["train"]["sequence"] for tok in tokenize(seq)]
vocab = ["<PAD>", "<UNK>"] + sorted(Counter(all_tokens))
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
    lens = [len(b["input_ids"]) for b in batch]
    maxlen = max(lens)
    x = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : len(b["input_ids"])] = b["input_ids"]
    y = torch.stack([b["label"] for b in batch])
    return {"input_ids": x, "label": y}


# ------------------------------------------------------------ #
#                       MODEL                                  #
# ------------------------------------------------------------ #
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_labels, pad):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(emb_dim, n_labels)
        self.pad = pad

    def forward(self, x):
        mask = (x != self.pad).unsqueeze(-1)
        em = self.emb(x)
        mean = (em * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(mean))


# ------------------------------------------------------------ #
#          SHAPE/COLOR WEIGHTED HELPERS (OPTIONAL)             #
# ------------------------------------------------------------ #
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [w0 if t == p else 0 for w0, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [w0 if t == p else 0 for w0, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0


# ------------------------------------------------------------ #
#                MAIN HYPERPARAMETER LOOP                      #
# ------------------------------------------------------------ #
candidate_batch_sizes = [32, 64, 128, 256]
num_epochs = 5
val_batch_size = 256  # fixed for speed

for bs in candidate_batch_sizes:
    print(f"\n===== Training with train batch_size={bs} =====")
    subdict = {
        "metrics": {"train_macroF1": [], "val_macroF1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    # loaders
    train_loader = DataLoader(
        SPRDataset(spr["train"]), batch_size=bs, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        SPRDataset(spr["dev"]),
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    # fresh model
    model = MeanPoolClassifier(len(vocab), 64, len(all_labels), pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        tr_loss, tr_pred, tr_true = 0.0, [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * batch["label"].size(0)
            tr_pred.extend(logits.argmax(1).cpu().numpy())
            tr_true.extend(batch["label"].cpu().numpy())
        tr_loss /= len(train_loader.dataset)
        tr_f1 = f1_score(tr_true, tr_pred, average="macro")
        subdict["losses"]["train"].append(tr_loss)
        subdict["metrics"]["train_macroF1"].append(tr_f1)

        # ---- validation ----
        model.eval()
        v_loss, v_pred, v_true = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                v_loss += loss.item() * batch["label"].size(0)
                v_pred.extend(logits.argmax(1).cpu().numpy())
                v_true.extend(batch["label"].cpu().numpy())
        v_loss /= len(val_loader.dataset)
        v_f1 = f1_score(v_true, v_pred, average="macro")
        subdict["losses"]["val"].append(v_loss)
        subdict["metrics"]["val_macroF1"].append(v_f1)
        subdict["timestamps"].append(time.time())
        print(f"Epoch {epoch}/{num_epochs} | val_loss={v_loss:.4f}  val_F1={v_f1:.4f}")

    # store final-epoch preds / gts & optional metrics
    subdict["predictions"] = v_pred
    subdict["ground_truth"] = v_true
    subdict["SWA"] = shape_weighted_accuracy(spr["dev"]["sequence"], v_true, v_pred)
    subdict["CWA"] = color_weighted_accuracy(spr["dev"]["sequence"], v_true, v_pred)
    print(f"Final Dev SWA={subdict['SWA']:.4f}  CWA={subdict['CWA']:.4f}")

    experiment_data["batch_size"]["SPR_BENCH"][str(bs)] = subdict

# ------------------------------------------------------------ #
#                    SAVE EVERYTHING                           #
# ------------------------------------------------------------ #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all results to", os.path.join(working_dir, "experiment_data.npy"))
