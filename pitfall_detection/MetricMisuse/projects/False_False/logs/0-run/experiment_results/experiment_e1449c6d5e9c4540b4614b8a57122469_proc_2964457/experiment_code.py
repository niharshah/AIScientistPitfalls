import os, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from collections import Counter
from typing import Dict, List
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
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

# ----------------------------- DEVICE ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- DATA LOCATION HELPERS ----------------------
def resolve_spr_path() -> pathlib.Path:
    """
    Attempt to locate SPR_BENCH folder.
    Order of preference:
        1) env var SPR_BENCH_PATH
        2) ./SPR_BENCH or ../SPR_BENCH or ~/SPR_BENCH
        3) baseline absolute path from earlier code
    """
    candidate_paths = []
    if "SPR_BENCH_PATH" in os.environ:
        candidate_paths.append(pathlib.Path(os.environ["SPR_BENCH_PATH"]))
    cwd = pathlib.Path.cwd()
    candidate_paths += [
        cwd / "SPR_BENCH",
        cwd.parent / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]
    for p in candidate_paths:
        if (p / "train.csv").exists():
            print("Found SPR_BENCH at", p.resolve())
            return p.resolve()
    raise FileNotFoundError(
        "Could not find SPR_BENCH dataset. Place 'train.csv/dev.csv/test.csv' in one "
        "of the standard locations or export SPR_BENCH_PATH pointing to the folder."
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr_root = resolve_spr_path()
spr = load_spr_bench(spr_root)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})


# --------------------------- VOCAB ---------------------------------
def tokenize(seq: str) -> List[str]:
    return seq.strip().split()


all_tokens = [tok for seq in spr["train"]["sequence"] for tok in tokenize(seq)]
vocab = ["<PAD>", "<UNK>"] + sorted(Counter(all_tokens))
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = stoi["<PAD>"], stoi["<UNK>"]

all_labels = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(all_labels)}
itos = {i: l for l, i in ltoi.items()}


def encode(seq: str) -> List[int]:
    return [stoi.get(tok, unk_idx) for tok in tokenize(seq)]


# ------------------------- DATASET / DATALOADER --------------------
class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [ltoi[lbl] for lbl in split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lengths = [len(b["input_ids"]) for b in batch]
    max_len = max(lengths)
    input_mat = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        input_mat[i, : lengths[i]] = b["input_ids"]
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_mat, "label": labels}


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ------------------------------ MODEL ------------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_sz: int, emb_dim: int, n_labels: int, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(emb_dim, n_labels)
        self.pad_id = pad_id

    def forward(self, x):
        mask = (x != self.pad_id).unsqueeze(-1)
        embs = self.emb(x) * mask
        avg = embs.sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.dropout(avg))


model = MeanPoolClassifier(len(vocab), 96, len(all_labels), pad_idx).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# --------------------- METRIC HELPERS ------------------------------
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int], fn
) -> float:
    weights = [fn(s) for s in seqs]
    correct_w = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct_w) / sum(weights) if sum(weights) else 0.0


# ----------------------- TRAIN / EVAL LOOPS ------------------------
def evaluate(net: nn.Module, loader) -> Dict[str, float]:
    net.eval()
    loss_tot, preds, trues = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = net(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss_tot += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(batch["label"].cpu().tolist())
    loss_avg = loss_tot / len(loader.dataset)
    macro_f1 = f1_score(trues, preds, average="macro")
    return {"loss": loss_avg, "preds": preds, "trues": trues, "macro_f1": macro_f1}


max_epochs = 25
patience = 4
best_metric = -1.0
epochs_since_improve = 0
best_state = None
best_preds, best_trues = None, None

for epoch in range(1, max_epochs + 1):
    # ---- train ----
    model.train()
    train_loss_tot, train_preds, train_trues = 0.0, [], []
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
        train_loss_tot += loss.item() * batch["label"].size(0)
        train_preds.extend(logits.argmax(1).cpu().tolist())
        train_trues.extend(batch["label"].cpu().tolist())

    train_loss = train_loss_tot / len(train_loader.dataset)
    train_macro = f1_score(train_trues, train_preds, average="macro")

    # ---- validation ----
    val_stats = evaluate(model, val_loader)
    val_loss = val_stats["loss"]
    val_macro = val_stats["macro_f1"]

    # shape/color metrics
    val_swa = weighted_accuracy(
        spr["dev"]["sequence"],
        val_stats["trues"],
        val_stats["preds"],
        count_shape_variety,
    )
    val_cwa = weighted_accuracy(
        spr["dev"]["sequence"],
        val_stats["trues"],
        val_stats["preds"],
        count_color_variety,
    )
    sc_gmean = (val_swa * val_cwa) ** 0.5 if val_swa > 0 and val_cwa > 0 else 0.0

    # ---- bookkeeping ----
    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_macroF1"].append(train_macro)
    ed["metrics"]["val_macroF1"].append(val_macro)
    ed["metrics"]["val_SWA"].append(val_swa)
    ed["metrics"]["val_CWA"].append(val_cwa)
    ed["metrics"]["val_SC_Gmean"].append(sc_gmean)
    ed["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
        f"val_F1={val_macro:.4f} SWA={val_swa:.4f} CWA={val_cwa:.4f} SC-G={sc_gmean:.4f}"
    )

    # ---- early stopping based on SC-Gmean ----
    if sc_gmean > best_metric:
        best_metric = sc_gmean
        epochs_since_improve = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        best_preds, best_trues = val_stats["preds"], val_stats["trues"]
    else:
        epochs_since_improve += 1
        if epochs_since_improve >= patience:
            print("Early stopping (no SC-Gmean improvement).")
            break

# ------------------------ SAVE FINAL RESULTS -----------------------
if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

experiment_data["SPR_BENCH"]["predictions"] = best_preds
experiment_data["SPR_BENCH"]["ground_truth"] = best_trues

# persist results
save_path = os.path.join(working_dir, "experiment_data.npy")
np.save(save_path, experiment_data)
print("Experiment data saved to", save_path)
