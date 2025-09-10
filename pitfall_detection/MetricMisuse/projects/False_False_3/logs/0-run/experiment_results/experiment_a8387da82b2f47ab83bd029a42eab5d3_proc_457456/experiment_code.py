import os, pathlib, numpy as np, torch
from typing import List
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ---------------- basic setup / paths ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"SPR_BENCH not found at {DATA_PATH}")


# ---------------- data utilities ---------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 0.0 if swa + cwa == 0 else 2 * swa * cwa / (swa + cwa)


# ---------------- vocabulary -------------------------
class Vocab:
    def __init__(self, tokens: List[str]):
        self.itos = ["<pad>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens: List[str]):
        return [self.stoi[t] for t in tokens]


# ---------------- model ------------------------------
class BagClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean")
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, text, offsets):
        return self.fc(self.embedding(text, offsets))


# ---------------- data load -------------------------
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

batch_size = 128


def collate_fn(batch):
    token_ids, offsets, label_ids = [], [0], []
    for ex in batch:
        ids = vocab(ex["sequence"].split())
        token_ids.extend(ids)
        offsets.append(offsets[-1] + len(ids))
        label_ids.append(label2id[ex["label"]])
    return (
        torch.tensor(token_ids, dtype=torch.long).to(device),
        torch.tensor(offsets[:-1], dtype=torch.long).to(device),
        torch.tensor(label_ids, dtype=torch.long).to(device),
    )


train_loader = DataLoader(
    spr["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    spr["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    spr["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

# ---------------- evaluation helper -----------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total_loss, y_true, y_pred, seqs = 0.0, [], [], []
    with torch.no_grad():
        for bidx, (text, offs, labs) in enumerate(loader):
            out = model(text, offs)
            total_loss += criterion(out, labs).item() * labs.size(0)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[i] for i in labs.cpu().tolist()])
            start, end = bidx * batch_size, bidx * batch_size + labs.size(0)
            seqs.extend(loader.dataset["sequence"][start:end])
    loss = total_loss / len(y_true)
    swa, cwa = shape_weighted_accuracy(seqs, y_true, y_pred), color_weighted_accuracy(
        seqs, y_true, y_pred
    )
    hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)
    return loss, swa, cwa, hwa, y_true, y_pred


# ---------------- experiment container ---------------
experiment_data = {"embed_dim": {"SPR_BENCH": {}}}

# ---------------- hyper-parameter loop ---------------
embed_dims = [32, 64, 128, 256]
epochs = 5
for emb_dim in embed_dims:
    print(f"\n====== Training with embed_dim={emb_dim} ======")
    model = BagClassifier(len(vocab), emb_dim, len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses_tr, losses_val, metrics_val = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for text, offs, labs in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(text, offs), labs)
            loss.backward()
            optimizer.step()
            running += loss.item() * labs.size(0)
        tr_loss = running / len(spr["train"])
        val_loss, swa, cwa, hwa, _, _ = evaluate(model, dev_loader)
        losses_tr.append(tr_loss)
        losses_val.append(val_loss)
        metrics_val.append({"SWA": swa, "CWA": cwa, "HWA": hwa})
        print(
            f"epoch {epoch}: train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} "
            f"| SWA={swa:.3f} CWA={cwa:.3f} HWA={hwa:.3f}"
        )

    # final test evaluation
    test_loss, swa_t, cwa_t, hwa_t, y_true_t, y_pred_t = evaluate(model, test_loader)
    print(
        f"Test | loss={test_loss:.4f} SWA={swa_t:.3f} CWA={cwa_t:.3f} HWA={hwa_t:.3f}"
    )

    # store
    experiment_data["embed_dim"]["SPR_BENCH"][str(emb_dim)] = {
        "metrics": {"train": None, "val": metrics_val},
        "losses": {"train": losses_tr, "val": losses_val},
        "predictions": y_pred_t,
        "ground_truth": y_true_t,
        "test_metrics": {"loss": test_loss, "SWA": swa_t, "CWA": cwa_t, "HWA": hwa_t},
    }

# ---------------- save -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
