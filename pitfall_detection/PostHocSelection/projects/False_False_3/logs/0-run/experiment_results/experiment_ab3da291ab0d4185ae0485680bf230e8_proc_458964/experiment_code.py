import os, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from typing import List

# ------------- basic working dir -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- device -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------- SPR utilities (copied for self-containment) -------------
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


def count_shape_variety(sequence: str) -> int:
    return len({tok[0] for tok in sequence.strip().split() if tok})


def count_color_variety(sequence: str) -> int:
    return len({tok[1] for tok in sequence.strip().split() if len(tok) > 1})


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


# ------------- vocabulary -------------
class Vocab:
    def __init__(self, tokens: List[str]):
        uniq = sorted(set(tokens))
        self.itos = ["<pad>"] + uniq
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __call__(self, toks: List[str]):
        return [self.stoi[t] for t in toks]

    def __len__(self):
        return len(self.itos)


# ------------- model -------------
class NeuralSymbolicClassifier(nn.Module):
    def __init__(self, vocab_sz: int, embed_dim: int, n_cls: int, n_sym: int = 3):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_sz, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim + n_sym, n_cls)

    def forward(self, text, offsets, sym_feats):
        emb = self.embedding(text, offsets)
        x = torch.cat([emb, sym_feats], dim=1)
        return self.fc(x)


# ------------- data path -------------
DATA_PATH = pathlib.Path(
    os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
if not DATA_PATH.exists():
    raise FileNotFoundError(f"SPR_BENCH not found at {DATA_PATH}")

spr = load_spr_bench(DATA_PATH)
print({split: len(ds) for split, ds in spr.items()})

# build vocab & label maps
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


# ------------- collate -------------
def symbolic_feats(seq: str):
    return [count_shape_variety(seq), count_color_variety(seq), len(seq.split())]


def collate(batch):
    tok_ids, offsets, labs, feats = [], [0], [], []
    for ex in batch:
        tids = vocab(ex["sequence"].split())
        tok_ids.extend(tids)
        offsets.append(offsets[-1] + len(tids))
        labs.append(label2id[ex["label"]])
        feats.append(symbolic_feats(ex["sequence"]))
    text = torch.tensor(tok_ids, dtype=torch.long).to(device)
    offsets = torch.tensor(offsets[:-1], dtype=torch.long).to(device)
    labs = torch.tensor(labs, dtype=torch.long).to(device)
    feats = torch.tensor(feats, dtype=torch.float32).to(device)
    return text, offsets, feats, labs


batch_size = 128
train_loader = DataLoader(
    spr["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    spr["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    spr["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)

# ------------- training utilities -------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader, store_seqs=False):
    model.eval()
    y_true, y_pred, seqs, loss_sum = [], [], [], 0.0
    with torch.no_grad():
        for (text, off, feats, labs), idx in zip(loader, range(len(loader))):
            out = model(text, off, feats)
            loss_sum += criterion(out, labs).item() * labs.size(0)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[i] for i in labs.cpu().tolist()])
            if store_seqs:
                start = idx * batch_size
                seqs.extend(loader.dataset["sequence"][start : start + labs.size(0)])
    avg_loss = loss_sum / len(y_true)
    swa = shape_weighted_accuracy(
        loader.dataset["sequence"] if not seqs else seqs, y_true, y_pred
    )
    return avg_loss, swa, y_true, y_pred


# ------------- experiment container -------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------- training loop -------------
embed_dim = 64
lr = 1e-3
epochs = 15

model = NeuralSymbolicClassifier(len(vocab), embed_dim, len(labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, epochs + 1):
    model.train()
    run_loss = 0.0
    for text, off, feats, labs in train_loader:
        optimizer.zero_grad()
        out = model(text, off, feats)
        loss = criterion(out, labs)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * labs.size(0)
    tr_loss = run_loss / len(spr["train"])
    val_loss, val_swa, _, _ = evaluate(model, dev_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append({"SWA": val_swa})

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA = {val_swa:.4f}")

# ------------- final test evaluation -------------
test_loss, test_swa, y_true, y_pred = evaluate(model, test_loader, store_seqs=True)
print(f"Test | loss={test_loss:.4f} | SWA={test_swa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
experiment_data["SPR_BENCH"]["test_metrics"] = {"loss": test_loss, "SWA": test_swa}

# ------------- save everything -------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {working_dir}")
