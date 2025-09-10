# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, time, math, json, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# -------------- deterministic behaviour -----------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# --------------------------------------------------------

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- helper functions originally from SPR.py -----------------
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
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w)


# -------------------------- Data ------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print("Loaded sizes:", {k: len(v) for k, v in spr.items()})


def seq_to_tokens(seq):
    return seq.strip().split()


vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for tok in seq_to_tokens(ex["sequence"]):
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

labels = sorted({ex["label"] for ex in spr["train"]})
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print("Num classes:", num_classes)


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, label2id):
        self.data = hf_split
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
            "ids": torch.tensor(ids, dtype=torch.long),
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

# -------- rule generalization mask -----------------------
train_tokens_set = set()
for seq in spr["train"]["sequence"]:
    train_tokens_set.update(seq_to_tokens(seq))


def compute_rgs_mask(seqs):
    return np.array(
        [any(tok not in train_tokens_set for tok in seq_to_tokens(s)) for s in seqs],
        dtype=bool,
    )


# ------------------ model --------------------------------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, ids):
        emb = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        avg = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.cls(avg)


criterion = nn.CrossEntropyLoss()


# --------------- training / evaluation helpers -----------
def evaluate(model, dloader):
    model.eval()
    tot, correct, lsum = 0, 0, 0.0
    preds, seqs, trues = [], [], []
    with torch.no_grad():
        for b in dloader:
            ids = b["ids"].to(device)
            labels = b["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            lsum += loss.item() * labels.size(0)
            p = logits.argmax(-1)
            correct += (p == labels).sum().item()
            tot += labels.size(0)
            preds.extend(p.cpu().tolist())
            seqs.extend(b["raw_seq"])
            trues.extend(labels.cpu().tolist())
    return lsum / tot, correct / tot, np.array(preds), seqs, np.array(trues)


# --------------- hyper-parameter sweep -------------------
learning_rates = [3e-4, 5e-4, 1e-3, 2e-3]
num_epochs = 5
experiment_data = {"learning_rate": {}}

for lr in learning_rates:
    print(f"\n=== Training with lr={lr} ===")
    model = AvgEmbedClassifier(vocab_size, 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    exp_entry = {
        "metrics": {"train": [], "val": [], "train_rgs": [], "val_rgs": []},
        "losses": {"train": [], "val": []},
        "predictions": {"val": [], "test": []},
        "ground_truth": {
            "val": [label2id[l] for l in spr["dev"]["label"]],
            "test": [label2id[l] for l in spr["test"]["label"]],
        },
    }
    for epoch in range(1, num_epochs + 1):
        model.train()
        run_loss, correct, tot = 0.0, 0, 0
        for b in train_loader:
            optimizer.zero_grad()
            ids = b["ids"].to(device)
            labels = b["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            tot += labels.size(0)
        tr_loss = run_loss / tot
        tr_acc = correct / tot

        val_loss, val_acc, val_pred, val_seq, val_true = evaluate(model, dev_loader)
        val_mask = compute_rgs_mask(val_seq)
        val_rgs = (
            (val_pred[val_mask] == val_true[val_mask]).mean()
            if val_mask.sum() > 0
            else 0.0
        )
        tr_rgs = 0.0

        exp_entry["metrics"]["train"].append(tr_acc)
        exp_entry["metrics"]["val"].append(val_acc)
        exp_entry["metrics"]["train_rgs"].append(tr_rgs)
        exp_entry["metrics"]["val_rgs"].append(val_rgs)
        exp_entry["losses"]["train"].append(tr_loss)
        exp_entry["losses"]["val"].append(val_loss)
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} val_RGS={val_rgs:.3f}"
        )
    # dev predictions final
    exp_entry["predictions"]["val"] = val_pred.tolist()

    # test evaluation
    test_loss, test_acc, test_pred, test_seq, test_true = evaluate(model, test_loader)
    test_mask = compute_rgs_mask(test_seq)
    test_rgs = (
        (test_pred[test_mask] == test_true[test_mask]).mean()
        if test_mask.sum() > 0
        else 0.0
    )
    swa = shape_weighted_accuracy(test_seq, test_true, test_pred)
    cwa = color_weighted_accuracy(test_seq, test_true, test_pred)
    print(
        f"TEST lr={lr}: loss={test_loss:.4f} acc={test_acc:.3f} RGS={test_rgs:.3f} "
        f"SWA={swa:.3f} CWA={cwa:.3f}"
    )
    exp_entry["predictions"]["test"] = test_pred.tolist()
    exp_entry["test_metrics"] = {
        "loss": test_loss,
        "acc": test_acc,
        "rgs": test_rgs,
        "swa": swa,
        "cwa": cwa,
    }

    # plot losses
    plt.figure()
    plt.plot(exp_entry["losses"]["train"], label="train")
    plt.plot(exp_entry["losses"]["val"], label="val")
    plt.legend()
    plt.title(f"Loss lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("CE")
    plt.savefig(os.path.join(working_dir, f"loss_curve_lr{lr}.png"))
    plt.close()

    experiment_data["learning_rate"][str(lr)] = exp_entry

# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All artifacts saved to", working_dir)
