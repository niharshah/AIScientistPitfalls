import os, pathlib, numpy as np, torch, random
from typing import List
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------- reproducibility ----------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- data loader utilities ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper to load a split
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa, cwa = shape_weighted_accuracy(seqs, y_true, y_pred), color_weighted_accuracy(
        seqs, y_true, y_pred
    )
    return 0 if (swa + cwa) == 0 else 2 * swa * cwa / (swa + cwa)


# ---------- vocab ----------
class Vocab:
    def __init__(self, tokens: List[str]):
        self.itos = ["<pad>"] + sorted(set(tokens))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, toks: List[str]):
        return [self.stoi[t] for t in toks]


# ---------- model ----------
class BagClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, num_classes: int, dropout_rate: float
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text, offsets):
        pooled = self.embedding(text, offsets)
        dropped = self.dropout(pooled)
        return self.fc(dropped)


# ---------- data path ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"SPR_BENCH not found at {DATA_PATH}")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- build vocab / label maps ----------
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


# ---------- collate ----------
def collate_batch(batch):
    token_ids, offsets, label_ids = [], [0], []
    for ex in batch:
        ids = vocab(ex["sequence"].split())
        token_ids.extend(ids)
        offsets.append(offsets[-1] + len(ids))
        label_ids.append(label2id[ex["label"]])
    offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
    text = torch.tensor(token_ids, dtype=torch.long, device=device)
    labels_t = torch.tensor(label_ids, dtype=torch.long, device=device)
    return text, offsets, labels_t


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

# ---------- experiment container ----------
experiment_data = {}


# ---------- evaluation helper ----------
def evaluate(model, data_loader, criterion):
    model.eval()
    y_true, y_pred, seqs, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for b_idx, (text, offsets, labels_t) in enumerate(data_loader):
            out = model(text, offsets)
            loss = criterion(out, labels_t)
            total_loss += loss.item() * labels_t.size(0)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[i] for i in labels_t.cpu().tolist()])
            start = b_idx * batch_size
            seqs.extend(
                data_loader.dataset["sequence"][start : start + labels_t.size(0)]
            )
    avg_loss = total_loss / len(y_true)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, swa, cwa, hwa, y_true, y_pred


# ---------- hyperparameter tuning ----------
embed_dim, epochs, lr = 64, 5, 1e-3
dropout_rates = [0.0, 0.1, 0.3, 0.5]

for p in dropout_rates:
    key = f"dropout_{p}"
    experiment_data[key] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "config": {"dropout_rate": p},
        }
    }
    model = BagClassifier(len(vocab), embed_dim, len(labels), p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for text, offsets, labels_t in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(text, offsets), labels_t)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels_t.size(0)
        train_loss = running_loss / len(spr["train"])
        val_loss, swa, cwa, hwa, _, _ = evaluate(model, dev_loader, criterion)

        experiment_data[key]["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data[key]["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data[key]["SPR_BENCH"]["metrics"]["train"].append(None)
        experiment_data[key]["SPR_BENCH"]["metrics"]["val"].append(
            {"SWA": swa, "CWA": cwa, "HWA": hwa}
        )

        print(
            f"[p={p}] Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | HWA={hwa:.4f}"
        )

    # final test evaluation
    test_loss, swa_t, cwa_t, hwa_t, y_true_t, y_pred_t = evaluate(
        model, test_loader, criterion
    )
    print(
        f"[p={p}] Test: loss={test_loss:.4f} | SWA={swa_t:.4f} | CWA={cwa_t:.4f} | HWA={hwa_t:.4f}"
    )

    experiment_data[key]["SPR_BENCH"]["predictions"] = y_pred_t
    experiment_data[key]["SPR_BENCH"]["ground_truth"] = y_true_t

# ---------- save ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
