import os, pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
# basic setup, working dir, device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# deterministic behaviour
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------
# helpers for SPR benchmark
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
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


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


# ---------------------------------------------------------------------
# Vocab
class Vocab:
    def __init__(self, tokens):
        self.itos = ["<pad>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, tokens):
        return [self.stoi[t] for t in tokens]

    def __len__(self):
        return len(self.itos)


# ---------------------------------------------------------------------
# Neural-Symbolic model
class HybridClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim, n_cls):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_sz, embed_dim, mode="mean")
        self.count_proj = nn.Linear(2, embed_dim)  # symbolic branch
        self.fc = nn.Linear(embed_dim * 2, n_cls)

    def forward(self, text, offsets, counts):
        em = self.embedding(text, offsets)  # [B, D]
        cs = self.count_proj(counts)  # [B, D]
        z = torch.cat([em, cs], dim=1)
        return self.fc(z)


# ---------------------------------------------------------------------
# load data
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# vocab build
all_toks = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_toks)

labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}


# ---------------------------------------------------------------------
# collate fn with symbolic counts
def collate(batch):
    tok_ids, offs, labs, cnts = [], [0], [], []
    for ex in batch:
        tokens = ex["sequence"].split()
        tok_ids.extend(vocab.encode(tokens))
        offs.append(offs[-1] + len(tokens))
        labs.append(lab2id[ex["label"]])
        cnts.append(
            [count_shape_variety(ex["sequence"]), count_color_variety(ex["sequence"])]
        )
    text = torch.tensor(tok_ids, dtype=torch.long)
    offs = torch.tensor(offs[:-1], dtype=torch.long)
    labs = torch.tensor(labs, dtype=torch.long)
    cnts = torch.tensor(cnts, dtype=torch.float)
    return (text.to(device), offs.to(device), cnts.to(device), labs.to(device))


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

# ---------------------------------------------------------------------
# containers for logging
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    total_loss, y_true, y_pred, seqs = 0.0, [], [], []
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (txt, off, cnt, lab) in enumerate(loader):
            out = model(txt, off, cnt)
            loss = ce(out, lab)
            total_loss += loss.item() * lab.size(0)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend([id2lab[p] for p in preds])
            y_true.extend([id2lab[i] for i in lab.cpu().tolist()])
            start = batch_idx * batch_size
            seqs.extend(loader.dataset["sequence"][start : start + lab.size(0)])
    sw_acc = shape_weighted_accuracy(seqs, y_true, y_pred)
    return total_loss / len(y_true), sw_acc, y_true, y_pred


# ---------------------------------------------------------------------
# training loop with early stopping
embed_dim, lr, max_epochs, patience = 64, 1e-3, 30, 3
model = HybridClassifier(len(vocab), embed_dim, len(labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

best_val_swa, wait = 0.0, 0
for epoch in range(1, max_epochs + 1):
    model.train()
    run_loss = 0.0
    for txt, off, cnt, lab in train_loader:
        optimizer.zero_grad()
        out = model(txt, off, cnt)
        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * lab.size(0)
    tr_loss = run_loss / len(spr["train"])

    val_loss, val_swa, _, _ = evaluate(model, dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA = {val_swa:.4f}")

    # log
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)

    # early stopping
    if val_swa > best_val_swa:
        best_val_swa, wait = val_swa, 0
        best_state = model.state_dict()
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# ---------------------------------------------------------------------
# evaluation on test split using best model
model.load_state_dict(best_state)
test_loss, test_swa, y_true, y_pred = evaluate(model, test_loader)
print(
    f"Test  | loss = {test_loss:.4f} | Shape-Weighted Accuracy (SWA) = {test_swa:.4f}"
)

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
experiment_data["SPR_BENCH"]["test_metrics"] = {"loss": test_loss, "SWA": test_swa}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
