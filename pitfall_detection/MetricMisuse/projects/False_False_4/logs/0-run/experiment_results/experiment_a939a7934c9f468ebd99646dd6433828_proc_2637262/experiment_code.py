import os, pathlib, random, time, math, json, gc
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# --------------------------------------------------------------------------
# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------------- data helpers (inlined SPR.py parts) --------------
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


def seq_to_tokens(seq):  # token = "Sg"
    return seq.strip().split()


def count_shape_variety(sequence: str):  # for SWA
    return len(set(tok[0] for tok in sequence.strip().split()))


def count_color_variety(sequence: str):  # for CWA
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / sum(w)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / sum(w)


# ----------------------------- load data ----------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print("Loaded sizes:", {k: len(v) for k, v in spr.items()})

# vocab
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


# -------------------------- dataset wrapper ------------------------------
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
    labs = torch.stack([b["label"] for b in batch])
    raws = [b["raw_seq"] for b in batch]
    return {
        "ids": padded,
        "lengths": torch.tensor(lens),
        "label": labs,
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

# RGS mask utilities
train_tokens_set = set()
for seq in spr["train"]["sequence"]:
    train_tokens_set.update(seq_to_tokens(seq))


def compute_rgs_mask(seqs):
    return np.array(
        [any(tok not in train_tokens_set for tok in seq_to_tokens(s)) for s in seqs],
        dtype=bool,
    )


# ------------------------------ model ------------------------------------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, ids):
        e = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        summed = (e * mask).sum(1)
        lens = mask.sum(1).clamp(min=1)
        avg = summed / lens
        return self.classifier(avg)


# ------------------------- training routine ------------------------------
def run_experiment(embed_dim, num_epochs=5, lr=1e-3):
    model = AvgEmbedClassifier(vocab_size, embed_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    metrics = {"train_acc": [], "dev_acc": [], "train_rgs": [], "dev_rgs": []}
    losses = {"train": [], "dev": []}

    def evaluate(loader):
        model.eval()
        tot, correct, loss_sum = 0, 0, 0.0
        all_pred, all_seq, all_true = [], [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch["ids"].to(device)
                labels = batch["label"].to(device)
                logits = model(ids)
                loss = criterion(logits, labels)
                loss_sum += loss.item() * labels.size(0)
                preds = logits.argmax(-1)
                correct += (preds == labels).sum().item()
                tot += labels.size(0)
                all_pred.extend(preds.cpu().tolist())
                all_seq.extend(batch["raw_seq"])
                all_true.extend(labels.cpu().tolist())
        return (
            loss_sum / tot,
            correct / tot,
            np.array(all_pred),
            all_seq,
            np.array(all_true),
        )

    for ep in range(1, num_epochs + 1):
        # train
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for batch in train_loader:
            optim.zero_grad()
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * labels.size(0)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            tot += labels.size(0)
        train_loss = loss_sum / tot
        train_acc = correct / tot

        dev_loss, dev_acc, dev_pred, dev_seq, dev_true = evaluate(dev_loader)

        # RGS
        dev_mask = compute_rgs_mask(dev_seq)
        dev_rgs = (
            (dev_pred[dev_mask] == dev_true[dev_mask]).mean()
            if dev_mask.sum() > 0
            else 0.0
        )

        metrics["train_acc"].append(train_acc)
        metrics["dev_acc"].append(dev_acc)
        metrics["train_rgs"].append(0.0)
        metrics["dev_rgs"].append(dev_rgs)
        losses["train"].append(train_loss)
        losses["dev"].append(dev_loss)

        print(
            f"[emb {embed_dim}] Epoch {ep}: "
            f"train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
            f"dev_acc={dev_acc:.3f} dev_RGS={dev_rgs:.3f}"
        )

    # final test eval
    test_loss, test_acc, test_pred, test_seq, test_true = evaluate(test_loader)
    test_mask = compute_rgs_mask(test_seq)
    test_rgs = (
        (test_pred[test_mask] == test_true[test_mask]).mean()
        if test_mask.sum() > 0
        else 0.0
    )
    swa = shape_weighted_accuracy(test_seq, test_true, test_pred)
    cwa = color_weighted_accuracy(test_seq, test_true, test_pred)
    print(
        f"[emb {embed_dim}] TEST acc={test_acc:.3f} RGS={test_rgs:.3f} "
        f"SWA={swa:.3f} CWA={cwa:.3f}"
    )

    # save curve
    plt.figure()
    plt.plot(losses["train"], label="train")
    plt.plot(losses["dev"], label="dev")
    plt.legend()
    plt.title(f"Loss curves (embed={embed_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.savefig(os.path.join(working_dir, f"SPR_loss_curve_{embed_dim}.png"))
    plt.close()

    result = {
        "metrics": metrics,
        "losses": losses,
        "predictions": {"dev": dev_pred.tolist(), "test": test_pred.tolist()},
        "final": {
            "dev_acc": dev_acc,
            "dev_rgs": dev_rgs,
            "test_acc": test_acc,
            "test_rgs": test_rgs,
            "SWA": swa,
            "CWA": cwa,
        },
    }

    # cleanup
    del model, optim, criterion
    torch.cuda.empty_cache()
    gc.collect()
    return result


# --------------------- hyperparameter sweep ------------------------------
embed_dims = [32, 64, 128, 256]
experiment_data = {"embed_dim_tuning": {"SPR_BENCH": {}}}

for dim in embed_dims:
    result = run_experiment(dim)
    experiment_data["embed_dim_tuning"]["SPR_BENCH"][str(dim)] = result

# ------------------------ save aggregated data ---------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All artifacts saved to", working_dir)
