import os, pathlib, random, time, math, json, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict


# --------------------------------------------------------------------------
# misc util
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(0)
# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------- helpers copied from SPR.py -----------------------------
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


def count_shape_variety(seq):
    return len({t[0] for t in seq.split() if t})


def count_color_variety(seq):
    return len({t[1] for t in seq.split() if len(t) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / sum(w)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / sum(w)


def seq_to_tokens(s):
    return s.strip().split()


# ----------------------- LOAD DATA ----------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print("Loaded sizes:", {k: len(v) for k, v in spr.items()})

# -------------------- VOCAB & LABEL MAP -----------------------------------
vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for tok in seq_to_tokens(ex["sequence"]):
        if tok not in vocab:
            vocab[tok] = len(vocab)
labels = sorted({ex["label"] for ex in spr["train"]})
label2id = {l: i for i, l in enumerate(labels)}
vocab_size = len(vocab)
num_classes = len(labels)
print("Vocab size:", vocab_size, "Num classes:", num_classes)


# ---------------------- DATASET WRAPPER -----------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab, label2id):
        self.data = split
        self.vocab = vocab
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = [
            self.vocab.get(t, self.vocab["<unk>"])
            for t in seq_to_tokens(row["sequence"])
        ]
        return {
            "ids": torch.tensor(ids),
            "label": torch.tensor(self.label2id[row["label"]]),
            "raw_seq": row["sequence"],
        }


def collate(batch):
    ids = [b["ids"] for b in batch]
    lens = [len(x) for x in ids]
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    labels = torch.stack([b["label"] for b in batch])
    raws = [b["raw_seq"] for b in batch]
    return {
        "ids": padded,
        "lengths": torch.tensor(lens),
        "label": labels,
        "raw_seq": raws,
    }


batch_size = 256
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], vocab, label2id),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"], vocab, label2id),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"], vocab, label2id),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)

# ---------- RULE GENERALIZATION MASK (tokens unseen in train) -------------
train_tokens_set = set()
for seq in spr["train"]["sequence"]:
    train_tokens_set.update(seq_to_tokens(seq))


def compute_rgs_mask(seqs):
    return np.array(
        [any(tok not in train_tokens_set for tok in seq_to_tokens(s)) for s in seqs],
        dtype=bool,
    )


# -------------------------- MODEL -----------------------------------------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lin = nn.Linear(embed_dim, num_classes)

    def forward(self, ids):
        emb = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        avg = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.lin(avg)


# ----------------------- EXPERIMENT STORAGE -------------------------------
experiment_data = {
    "weight_decay": {
        "SPR_BENCH": {
            "values": [],
            "metrics": {
                "train_acc": [],
                "dev_acc": [],
                "test_acc": [],
                "dev_rgs": [],
                "test_rgs": [],
            },
            "losses": {"train": [], "dev": []},
            "predictions": {"dev": [], "test": []},
            "ground_truth": {
                "dev": [label2id[l] for l in spr["dev"]["label"]],
                "test": [label2id[l] for l in spr["test"]["label"]],
            },
        }
    }
}

# ---------------------- TRAIN & EVAL FUNCTIONS ----------------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    preds, seqs, trues = [], [], []
    with torch.no_grad():
        for b in loader:
            ids = b["ids"].to(device)
            labels = b["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            p = logits.argmax(-1)
            preds.extend(p.cpu().tolist())
            seqs.extend(b["raw_seq"])
            trues.extend(labels.cpu().tolist())
            correct += (p == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total, np.array(preds), seqs, np.array(trues)


# ------------------------- HYPERPARAM SEARCH ------------------------------
embed_dim = 64
num_epochs = 5
weight_decays = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
best_idx = -1
best_dev_acc = -1.0
for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    experiment_data["weight_decay"]["SPR_BENCH"]["values"].append(wd)
    model = AvgEmbedClassifier(vocab_size, embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    train_losses = []
    dev_losses = []
    for ep in range(1, num_epochs + 1):
        model.train()
        tot = 0
        korr = 0
        run_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
            korr += (logits.argmax(-1) == labels).sum().item()
            tot += labels.size(0)
        train_loss = run_loss / tot
        train_acc = korr / tot
        dev_loss, dev_acc, dev_pred, dev_seq, dev_true = evaluate(model, dev_loader)
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        print(f"  Ep{ep}: train_loss={train_loss:.4f} dev_acc={dev_acc:.3f}")
    # final dev metrics
    dev_mask = compute_rgs_mask(dev_seq)
    dev_rgs = (
        (dev_pred[dev_mask] == dev_true[dev_mask]).mean() if dev_mask.any() else 0.0
    )
    # test metrics
    test_loss, test_acc, test_pred, test_seq, test_true = evaluate(model, test_loader)
    test_mask = compute_rgs_mask(test_seq)
    test_rgs = (
        (test_pred[test_mask] == test_true[test_mask]).mean()
        if test_mask.any()
        else 0.0
    )
    # store
    ed = experiment_data["weight_decay"]["SPR_BENCH"]
    ed["metrics"]["train_acc"].append(train_acc)
    ed["metrics"]["dev_acc"].append(dev_acc)
    ed["metrics"]["test_acc"].append(test_acc)
    ed["metrics"]["dev_rgs"].append(dev_rgs)
    ed["metrics"]["test_rgs"].append(test_rgs)
    ed["losses"]["train"].append(train_losses)
    ed["losses"]["dev"].append(dev_losses)
    ed["predictions"]["dev"].append(dev_pred.tolist())
    ed["predictions"]["test"].append(test_pred.tolist())
    # best tracking
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_idx = len(weight_decays) - len(ed["metrics"]["dev_acc"])
        best_idx = len(ed["metrics"]["dev_acc"]) - 1
        best_losses = (train_losses, dev_losses)
        best_wd = wd
        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        best_test_pred, best_test_seq, best_test_true = test_pred, test_seq, test_true
        best_test_rgs = test_rgs
        best_test_acc = test_acc

print(f"\nBest weight_decay={best_wd} with dev_acc={best_dev_acc:.3f}")
# ------------------- PLOT BEST LOSS CURVE ---------------------------------
plt.figure()
plt.plot(best_losses[0], label="train")
plt.plot(best_losses[1], label="dev")
plt.legend()
plt.title(f"Loss curve (weight_decay={best_wd})")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy")
plt.savefig(os.path.join(working_dir, "SPR_loss_curve.png"))

# Extra metrics on best run
swa = shape_weighted_accuracy(best_test_seq, best_test_true, best_test_pred)
cwa = color_weighted_accuracy(best_test_seq, best_test_true, best_test_pred)
print(
    f"TEST (best) acc={best_test_acc:.3f} RGS={best_test_rgs:.3f} | "
    f"SWA={swa:.3f} CWA={cwa:.3f}"
)

# ------------------- SAVE METRICS ----------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Artifacts saved to", working_dir)
