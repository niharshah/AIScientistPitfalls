import os, pathlib, numpy as np, torch, random
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# --------------------- folders & RNG ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# --------------------- device ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


# --------------------- dataset helpers -------------------
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
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa, cwa = shape_weighted_accuracy(seqs, y_true, y_pred), color_weighted_accuracy(
        seqs, y_true, y_pred
    )
    return 0.0 if (swa + cwa) == 0 else 2 * swa * cwa / (swa + cwa)


# --------------------- vocab -----------------------------
class Vocab:
    def __init__(self, tokens):
        self.itos = ["<pad>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, toks):
        return [self.stoi[t] for t in toks]


# --------------------- model -----------------------------
class BagClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, n_cls):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_sz, emb_dim, mode="mean")
        self.fc = nn.Linear(emb_dim, n_cls)

    def forward(self, text, offsets):
        return self.fc(self.embedding(text, offsets))


# --------------------- data ------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"SPR_BENCH not found at {DATA_PATH}")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# vocab & labels
all_toks = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_toks)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


# collate
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


batch_size = 128
train_loader = DataLoader(spr["train"], batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(spr["dev"], batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(spr["test"], batch_size, shuffle=False, collate_fn=collate_fn)


# --------------------- evaluation ------------------------
def evaluate(model, criterion, dloader):
    model.eval()
    y_true, y_pred, seqs, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for b_idx, (text, off, labs) in enumerate(dloader):
            out = model(text, off)
            loss = criterion(out, labs)
            total_loss += loss.item() * labs.size(0)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[t] for t in labs.cpu().tolist()])
            start, end = b_idx * batch_size, b_idx * batch_size + labs.size(0)
            seqs.extend(dloader.dataset["sequence"][start:end])
    avg_loss = total_loss / len(y_true)
    swa, cwa, hwa = (
        shape_weighted_accuracy(seqs, y_true, y_pred),
        color_weighted_accuracy(seqs, y_true, y_pred),
        harmonic_weighted_accuracy(seqs, y_true, y_pred),
    )
    return avg_loss, swa, cwa, hwa, y_true, y_pred


# --------------------- experiment tracking ---------------
experiment_data = {
    "label_smoothing": {
        "SPR_BENCH": {
            "smoothing_values": [],
            "losses": {"val": []},
            "metrics": {"val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# --------------------- hyperparameter sweep --------------
smooth_vals = [0.00, 0.05, 0.10, 0.15, 0.20]
epochs = 5
embed_dim = 64

best_hwa, best_state, best_smooth = -1.0, None, None

for sm in smooth_vals:
    print(f"\n===== Training with label_smoothing={sm:.2f} =====")
    model = BagClassifier(len(vocab), embed_dim, len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=sm)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for text, off, labs in train_loader:
            optimizer.zero_grad()
            out = model(text, off)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labs.size(0)
        train_loss = run_loss / len(spr["train"])

        val_loss, swa, cwa, hwa, _, _ = evaluate(model, criterion, dev_loader)
        print(f"Epoch {ep}/{epochs}: val_loss={val_loss:.4f} | HWA={hwa:.4f}")

    # store per-smoothing summary (last epoch metrics)
    experiment_data["label_smoothing"]["SPR_BENCH"]["smoothing_values"].append(sm)
    experiment_data["label_smoothing"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["label_smoothing"]["SPR_BENCH"]["metrics"]["val"].append(
        {"SWA": swa, "CWA": cwa, "HWA": hwa}
    )

    # keep best on dev
    if hwa > best_hwa:
        best_hwa, best_state, best_smooth = hwa, model.state_dict(), sm

print(f"\nBest smoothing={best_smooth:.2f} with dev HWA={best_hwa:.4f}")

# --------------------- final test evaluation -------------
best_model = BagClassifier(len(vocab), embed_dim, len(labels)).to(device)
best_model.load_state_dict(best_state)
best_criterion = nn.CrossEntropyLoss(label_smoothing=best_smooth)
test_loss, swa_t, cwa_t, hwa_t, y_true_t, y_pred_t = evaluate(
    best_model, best_criterion, test_loader
)

print(
    f"TEST | loss={test_loss:.4f} | SWA={swa_t:.4f} | CWA={cwa_t:.4f} | HWA={hwa_t:.4f}"
)

exp_section = experiment_data["label_smoothing"]["SPR_BENCH"]
exp_section["predictions"] = y_pred_t
exp_section["ground_truth"] = y_true_t
exp_section["best_smoothing"] = best_smooth
exp_section["test_metrics"] = {"SWA": swa_t, "CWA": cwa_t, "HWA": hwa_t}
exp_section["test_loss"] = test_loss

# --------------------- save ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
