import os, pathlib, numpy as np, torch, random
from typing import List, Dict
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ---------------- paths / dirs ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- reproducibility -------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- device ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- data utilities --------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 0 if (swa + cwa) == 0 else 2 * swa * cwa / (swa + cwa)


# ---------------- vocab -----------------------
class Vocab:
    def __init__(self, tokens: List[str]):
        self.itos = ["<pad>"] + sorted(set(tokens))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens: List[str]) -> List[int]:
        return [self.stoi[t] for t in tokens]


# ---------------- model -----------------------
class BagClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_cls: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, n_cls)

    def forward(self, text, offsets):
        return self.fc(self.embedding(text, offsets))


# ---------------- dataset loading -------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"SPR_BENCH not found at {DATA_PATH}")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------------- vocab / label map -----------
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


# ---------------- collate ---------------------
def collate_batch(batch):
    token_ids, offsets, label_ids = [], [0], []
    for ex in batch:
        ids = vocab(ex["sequence"].split())
        token_ids.extend(ids)
        offsets.append(offsets[-1] + len(ids))
        label_ids.append(label2id[ex["label"]])
    text = torch.tensor(token_ids, dtype=torch.long)
    offsets = torch.tensor(offsets[:-1], dtype=torch.long)
    labels_t = torch.tensor(label_ids, dtype=torch.long)
    sequences = [ex["sequence"] for ex in batch]
    return text.to(device), offsets.to(device), labels_t.to(device), sequences


# ---------------- loaders ---------------------
batch_size = 128
train_loader = DataLoader(
    spr["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_batch
)
dev_loader = DataLoader(
    spr["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate_batch
)
test_loader = DataLoader(
    spr["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_batch
)

# ---------------- experiment store ------------
experiment_data = {
    "weight_decay": {
        "SPR_BENCH": {
            "decay_values": [],
            "losses": {"train": [], "val": []},
            "metrics": {"val": []},
            "best_val_hwa": [],
            "test_metrics": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------------- evaluation helper ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, data_loader):
    model.eval()
    y_true, y_pred, sequences = [], [], []
    tot_loss = 0.0
    with torch.no_grad():
        for text, offsets, labels_t, seqs in data_loader:
            outputs = model(text, offsets)
            loss = criterion(outputs, labels_t)
            tot_loss += loss.item() * labels_t.size(0)
            preds = outputs.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[i] for i in labels_t.cpu().tolist()])
            sequences.extend(seqs)
    avg_loss = tot_loss / len(y_true)
    swa = shape_weighted_accuracy(sequences, y_true, y_pred)
    cwa = color_weighted_accuracy(sequences, y_true, y_pred)
    hwa = harmonic_weighted_accuracy(sequences, y_true, y_pred)
    return avg_loss, swa, cwa, hwa, y_true, y_pred


# ---------------- training loop --------------
def run_training(weight_decay: float, epochs: int = 5, embed_dim: int = 64):
    model = BagClassifier(len(vocab), embed_dim, len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    tr_losses, val_losses, val_metrics = [], [], []

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for text, offsets, labels_t, _ in train_loader:
            optimizer.zero_grad()
            out = model(text, offsets)
            loss = criterion(out, labels_t)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels_t.size(0)
        tr_loss = running / len(spr["train"])
        val_loss, swa, cwa, hwa, _, _ = evaluate(model, dev_loader)
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_metrics.append({"SWA": swa, "CWA": cwa, "HWA": hwa})
        print(
            f"decay={weight_decay} | epoch={ep} | tr_loss={tr_loss:.4f} | "
            f"val_loss={val_loss:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | HWA={hwa:.4f}"
        )
    return model, tr_losses, val_losses, val_metrics


weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
best_hwa, best_idx = -1, -1
stored_models: Dict[int, torch.nn.Module] = {}

for idx, wd in enumerate(weight_decays):
    model, tr_l, val_l, val_m = run_training(wd)
    hwa_last = val_m[-1]["HWA"]
    # store experiment data
    exp = experiment_data["weight_decay"]["SPR_BENCH"]
    exp["decay_values"].append(wd)
    exp["losses"]["train"].append(tr_l)
    exp["losses"]["val"].append(val_l)
    exp["metrics"]["val"].append(val_m)
    exp["best_val_hwa"].append(hwa_last)
    # keep model for potential best selection
    stored_models[idx] = model
    if hwa_last > best_hwa:
        best_hwa, best_idx = hwa_last, idx

print(f"Best weight_decay={weight_decays[best_idx]} with dev HWA={best_hwa:.4f}")

# --------------- final test evaluation -------
best_model = stored_models[best_idx]
test_loss, swa_t, cwa_t, hwa_t, y_true_t, y_pred_t = evaluate(best_model, test_loader)
print(
    f"Test (best model): loss={test_loss:.4f} | SWA={swa_t:.4f} | "
    f"CWA={cwa_t:.4f} | HWA={hwa_t:.4f}"
)

exp = experiment_data["weight_decay"]["SPR_BENCH"]
exp["test_metrics"].append(
    {"loss": test_loss, "SWA": swa_t, "CWA": cwa_t, "HWA": hwa_t}
)
exp["predictions"] = y_pred_t
exp["ground_truth"] = y_true_t

# --------------- save experiment -------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
