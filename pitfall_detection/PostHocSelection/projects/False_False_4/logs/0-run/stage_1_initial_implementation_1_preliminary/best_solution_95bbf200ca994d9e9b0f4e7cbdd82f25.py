import os, pathlib, random, time, math, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# mandatory working directory & device handling
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --------------------------------------------------------------------------

# ---------- helper from supplied SPR.py (inlined for self-containment) -----
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights)


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights)


# --------------------------------------------------------------------------

# ----------------------- LOAD DATA ----------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    # fallback for local testing – user may symlink dataset here
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print("Loaded sizes:", {k: len(v) for k, v in spr.items()})


# -------------------- SYMBOLIC VOCABULARY ---------------------------------
def seq_to_tokens(seq):
    return seq.strip().split()  # tokens are like "Sg" (shape S, color g)


# Build vocabulary from training set
vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for tok in seq_to_tokens(ex["sequence"]):
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

# Label mapping
labels = sorted({ex["label"] for ex in spr["train"]})
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print("Num classes:", num_classes)


# ---------------------- DATASET WRAPPER -----------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, label2id):
        self.data = hf_split
        self.vocab = vocab
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        toks = [
            self.vocab.get(t, self.vocab["<unk>"])
            for t in seq_to_tokens(row["sequence"])
        ]
        return {
            "ids": torch.tensor(toks, dtype=torch.long),
            "label": torch.tensor(self.label2id[row["label"]], dtype=torch.long),
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
train_ds = SPRTorchDataset(spr["train"], vocab, label2id)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id)
test_ds = SPRTorchDataset(spr["test"], vocab, label2id)
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
)

# ---------- RULE GENERALIZATION MASK (tokens unseen in train) -------------
train_tokens_set = set()
for seq in spr["train"]["sequence"]:
    train_tokens_set.update(seq_to_tokens(seq))


def compute_rgs_mask(seqs):
    mask = []
    for s in seqs:
        mask.append(any(tok not in train_tokens_set for tok in seq_to_tokens(s)))
    return np.array(mask, dtype=bool)


# -------------------------- MODEL -----------------------------------------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, ids):
        emb = self.emb(ids)  # [B, T, D]
        mask = (ids != 0).unsqueeze(-1)  # 1 for real tokens
        summed = (emb * mask).sum(dim=1)
        lens = mask.sum(dim=1).clamp(min=1)
        avg = summed / lens
        return self.classifier(avg)


embed_dim = 64
model = AvgEmbedClassifier(vocab_size, embed_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------ EXPERIMENT TRACKING STRUCTURE -------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "dev_acc": [], "train_rgs": [], "dev_rgs": []},
        "losses": {"train": [], "dev": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {
            "dev": [label2id[l] for l in spr["dev"]["label"]],
            "test": [label2id[l] for l in spr["test"]["label"]],
        },
    }
}


# -------------------------- TRAIN LOOP ------------------------------------
def evaluate(dloader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_pred, all_seq, all_true = [], [], []
    with torch.no_grad():
        for batch in dloader:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_pred.extend(preds.cpu().tolist())
            all_seq.extend(batch["raw_seq"])
            all_true.extend(labels.cpu().tolist())
    avg_loss = loss_sum / total
    acc = correct / total
    return avg_loss, acc, np.array(all_pred), all_seq, np.array(all_true)


num_epochs = 5
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss, correct, tot = 0.0, 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        ids = batch["ids"].to(device)
        labels = batch["label"].to(device)
        logits = model(ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(-1)
        correct += (preds == labels).sum().item()
        tot += labels.size(0)
    train_loss = running_loss / tot
    train_acc = correct / tot

    dev_loss, dev_acc, dev_pred, dev_seq, dev_true = evaluate(dev_loader)

    # RGS computation
    dev_mask = compute_rgs_mask(dev_seq)
    if dev_mask.sum() > 0:
        dev_rgs = (dev_pred[dev_mask] == dev_true[dev_mask]).mean()
    else:
        dev_rgs = 0.0

    train_mask = compute_rgs_mask(spr["train"]["sequence"])
    train_rgs = np.array([0])  # meaningless on train, keep 0
    train_rgs = 0.0

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["dev_acc"].append(dev_acc)
    experiment_data["SPR_BENCH"]["metrics"]["train_rgs"].append(train_rgs)
    experiment_data["SPR_BENCH"]["metrics"]["dev_rgs"].append(dev_rgs)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["dev"].append(dev_loss)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
        f"acc={dev_acc:.3f} RGS={dev_rgs:.3f}"
    )

# ------------------------- FINAL TEST EVAL --------------------------------
test_loss, test_acc, test_pred, test_seq, test_true = evaluate(test_loader)
test_mask = compute_rgs_mask(test_seq)
test_rgs = (
    (test_pred[test_mask] == test_true[test_mask]).mean()
    if test_mask.sum() > 0
    else 0.0
)
print(f"\nTEST  – loss={test_loss:.4f} acc={test_acc:.3f} RGS={test_rgs:.3f}")

# additional metrics SWA / CWA
swa = shape_weighted_accuracy(test_seq, test_true, test_pred)
cwa = color_weighted_accuracy(test_seq, test_true, test_pred)
print(f"Shape-Weighted Accuracy: {swa:.3f} | Color-Weighted Accuracy: {cwa:.3f}")

experiment_data["SPR_BENCH"]["predictions"]["dev"] = dev_pred.tolist()
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_pred.tolist()

# ------------------- SAVE METRICS & PLOT LOSSES ---------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["dev"], label="dev")
plt.legend()
plt.title("Loss curves")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy")
plt.savefig(os.path.join(working_dir, "SPR_loss_curve.png"))

print("All artifacts saved to", working_dir)
