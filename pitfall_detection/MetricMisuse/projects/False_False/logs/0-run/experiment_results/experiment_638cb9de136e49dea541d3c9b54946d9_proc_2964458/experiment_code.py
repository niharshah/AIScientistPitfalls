# ----------------- SET-UP ----------------------------------------------------
import os, pathlib, time, math, numpy as np, torch
from collections import Counter
from typing import List, Dict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# create working directory for any artefacts
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device handling (CRITICAL GPU REQUIREMENT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- EXPERIMENT DATA STRUCTURE ---------------------------------
experiment_data: Dict = {
    "SPR_BENCH": {
        "metrics": {
            "train_macroF1": [],
            "val_macroF1": [],
            "val_SWA": [],
            "val_CWA": [],
            "val_SC_Gmean": [],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ----------------- DATA LOADING ----------------------------------------------
def resolve_spr_path() -> pathlib.Path:
    """
    Try several sensible locations, then raise if nothing valid is found.
    """
    # 1) environment variable
    cands: List[pathlib.Path] = []
    if "SPR_BENCH_PATH" in os.environ:
        cands.append(pathlib.Path(os.environ["SPR_BENCH_PATH"]))
    # 2) common relative paths
    cwd = pathlib.Path.cwd()
    cands += [
        cwd / "SPR_BENCH",
        cwd.parent / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
    ]
    # 3) absolute path used by earlier baseline code
    cands.append(pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"))
    for p in cands:
        if (p / "train.csv").exists() and (p / "dev.csv").exists():
            print("Found SPR_BENCH at", p.resolve())
            return p.resolve()
    raise FileNotFoundError(
        "Could not locate SPR_BENCH. Place csvs locally or set $SPR_BENCH_PATH."
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    """
    Return DatasetDict with train/dev/test splits, each loaded from its csv.
    """

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",  # treat csv as a single split
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


spr_root = resolve_spr_path()
spr = load_spr_bench(spr_root)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})


# ----------------- TOKENISATION & VOCAB --------------------------------------
def tokenize(seq: str) -> List[str]:
    return seq.strip().split()


all_train_tokens = [tok for seq in spr["train"]["sequence"] for tok in tokenize(seq)]
vocab = ["<PAD>", "<UNK>"] + sorted(Counter(all_train_tokens))
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = stoi["<PAD>"], stoi["<UNK>"]

labels = sorted({lab for lab in spr["train"]["label"]})
ltoi = {l: i for i, l in enumerate(labels)}


def encode(seq: str) -> List[int]:
    return [stoi.get(tok, unk_idx) for tok in tokenize(seq)]


# ----------------- DATASET / DATALOADER --------------------------------------
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


def collate_fn(batch):
    lengths = [len(b["input_ids"]) for b in batch]
    max_len = max(lengths)
    x = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : lengths[i]] = b["input_ids"]
    y = torch.stack([b["label"] for b in batch])
    return {"input_ids": x, "label": y}


batch_size = 256  # tuned
train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=512, shuffle=False, collate_fn=collate_fn
)


# ----------------- MODEL ------------------------------------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_sz: int, emb_dim: int, num_labels: int, pad: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad)
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(emb_dim, num_labels)
        self.pad = pad

    def forward(self, x):
        mask = (x != self.pad).unsqueeze(-1)  # (B, T, 1)
        emb = self.emb(x)  # (B, T, D)
        summed = (emb * mask).sum(1)
        denom = mask.sum(1).clamp(min=1)
        mean = summed / denom
        return self.fc(self.drop(mean))


model = MeanPoolClassifier(len(vocab), 128, len(labels), pad_idx).to(device)

# ----------------- OPTIMISER / LOSS ------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)  # tuned lr


# ----------------- METRIC HELPERS --------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len({tok[0] for tok in sequence.split() if tok})


def count_color_variety(sequence: str) -> int:
    return len({tok[1] for tok in sequence.split() if len(tok) > 1})


def weighted_accuracy(seqs: List[str], y_t, y_p, fn) -> float:
    weights = [fn(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_t, y_p)]
    tot = sum(weights)
    return sum(correct) / tot if tot else 0.0


# ----------------- EVALUATION -------------------------------------------------
def evaluate(mdl, loader):
    mdl.eval()
    tot_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = mdl(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(batch["label"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(trues, preds, average="macro")
    return avg_loss, trues, preds, macro_f1


# ----------------- TRAINING LOOP ---------------------------------------------
max_epochs, patience = 25, 4
best_sc_gmean, epochs_since_improve = -1, 0
best_state, best_preds, best_trues = None, None, None

for epoch in range(1, max_epochs + 1):
    # ---- training -----------------------------------------------------------
    model.train()
    epoch_loss, epoch_preds, epoch_trues = 0.0, [], []
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["label"].size(0)
        epoch_preds.extend(logits.argmax(1).cpu().tolist())
        epoch_trues.extend(batch["label"].cpu().tolist())
    train_loss = epoch_loss / len(train_loader.dataset)
    train_macro = f1_score(epoch_trues, epoch_preds, average="macro")
    # ---- validation ---------------------------------------------------------
    val_loss, val_trues, val_preds, val_macro = evaluate(model, val_loader)
    val_swa = weighted_accuracy(
        spr["dev"]["sequence"], val_trues, val_preds, count_shape_variety
    )
    val_cwa = weighted_accuracy(
        spr["dev"]["sequence"], val_trues, val_preds, count_color_variety
    )
    val_sc_gmean = math.sqrt(val_swa * val_cwa)

    # ---- log & save metrics --------------------------------------------------
    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_macroF1"].append(train_macro)
    ed["metrics"]["val_macroF1"].append(val_macro)
    ed["metrics"]["val_SWA"].append(val_swa)
    ed["metrics"]["val_CWA"].append(val_cwa)
    ed["metrics"]["val_SC_Gmean"].append(val_sc_gmean)
    ed["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d} | "
        f"val_loss={val_loss:.4f} | "
        f"macroF1={val_macro:.4f} | "
        f"SWA={val_swa:.4f} | CWA={val_cwa:.4f} | "
        f"SC-Gmean={val_sc_gmean:.4f}"
    )

    # ---- early stopping based on SC-Gmean -----------------------------------
    if val_sc_gmean > best_sc_gmean:
        best_sc_gmean = val_sc_gmean
        epochs_since_improve = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        best_preds, best_trues = val_preds, val_trues
    else:
        epochs_since_improve += 1
        if epochs_since_improve >= patience:
            print("Early stopping triggered.")
            break

# ----------------- RESTORE BEST & FINAL METRICS ------------------------------
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
experiment_data["SPR_BENCH"]["predictions"] = best_preds
experiment_data["SPR_BENCH"]["ground_truth"] = best_trues

print(f"Best Dev SC-Gmean: {best_sc_gmean:.4f}")

# ----------------- SAVE ALL RESULTS ------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
