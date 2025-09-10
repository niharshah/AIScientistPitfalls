import os, pathlib, random, time, math, json, copy
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    return sum(w for w, t, p in zip(weights, y_true, y_pred) if t == p) / sum(weights)


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    return sum(w for w, t, p in zip(weights, y_true, y_pred) if t == p) / sum(weights)


# --------------------------------------------------------------------------
# ----------------------------- DATA ---------------------------------------
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
        vocab.setdefault(tok, len(vocab))
vocab_size = len(vocab)

labels = sorted({ex["label"] for ex in spr["train"]})
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print(f"Vocab:{vocab_size} | Classes:{num_classes}")


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
    return {
        "ids": padded,
        "lengths": torch.tensor(lens),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


# mask for rule-generalisation score
train_tokens_set = set()
for seq in spr["train"]["sequence"]:
    train_tokens_set.update(seq_to_tokens(seq))


def compute_rgs_mask(seqs):
    return np.array(
        [any(tok not in train_tokens_set for tok in seq_to_tokens(s)) for s in seqs],
        dtype=bool,
    )


# -------------------------------- MODEL -----------------------------------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, ids):
        emb = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        summed = (emb * mask).sum(1)
        avg = summed / mask.sum(1).clamp(min=1)
        return self.fc(avg)


# -------------------------------- EVAL ------------------------------------
def evaluate(model, criterion, dloader):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    preds, seqs, trues = [], [], []
    with torch.no_grad():
        for batch in dloader:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            p = logits.argmax(-1)
            correct += (p == labels).sum().item()
            tot += labels.size(0)
            preds.extend(p.cpu().tolist())
            seqs.extend(batch["raw_seq"])
            trues.extend(labels.cpu().tolist())
    return loss_sum / tot, correct / tot, np.array(preds), seqs, np.array(trues)


# ----------------------- HYPERPARAMETER SEARCH ----------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
batch_sizes = [32, 64, 128, 256]
num_epochs = 5
embed_dim = 64
criterion = nn.CrossEntropyLoss()
experiment_data = {
    "batch_size": {
        "SPR_BENCH": {
            "metrics": {"train_acc": {}, "dev_acc": {}, "test_acc": {}},
            "losses": {"train": {}, "dev": {}},
            "rgs": {"dev": {}, "test": {}},
            "predictions": {"dev": {}, "test": {}},
            "ground_truth": {
                "dev": [label2id[l] for l in spr["dev"]["label"]],
                "test": [label2id[l] for l in spr["test"]["label"]],
            },
            "best_batch_size": None,
        }
    }
}

best_dev = -1.0
best_state = None
best_bs = None
loss_curve_dict = {}

for bs in batch_sizes:
    print(f"\n=== Training with batch_size={bs} ===")
    train_loader = DataLoader(
        SPRTorchDataset(spr["train"], vocab, label2id),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        SPRTorchDataset(spr["dev"], vocab, label2id),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        SPRTorchDataset(spr["test"], vocab, label2id),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate,
    )

    model = AvgEmbedClassifier(vocab_size, embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    dev_losses = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        run_loss, correct, tot = 0.0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
            tot += labels.size(0)
        tr_loss = run_loss / tot
        tr_acc = correct / tot
        dv_loss, dv_acc, dv_pred, dv_seq, dv_true = evaluate(
            model, criterion, dev_loader
        )
        train_losses.append(tr_loss)
        dev_losses.append(dv_loss)

        dv_mask = compute_rgs_mask(dv_seq)
        dv_rgs = (
            (dv_pred[dv_mask] == dv_true[dv_mask]).mean() if dv_mask.sum() > 0 else 0.0
        )
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} dev_loss={dv_loss:.4f} "
            f"dev_acc={dv_acc:.3f} dev_RGS={dv_rgs:.3f}"
        )

        # keep best global model
        if dv_acc > best_dev:
            best_dev = dv_acc
            best_state = copy.deepcopy(model.state_dict())
            best_bs = bs
            best_dev_pred = dv_pred.copy()
            best_dev_seq = list(dv_seq)
            best_dev_true = dv_true.copy()

    # store per-batch-size metrics
    experiment_data["batch_size"]["SPR_BENCH"]["metrics"]["train_acc"][bs] = tr_acc
    experiment_data["batch_size"]["SPR_BENCH"]["metrics"]["dev_acc"][bs] = dv_acc
    experiment_data["batch_size"]["SPR_BENCH"]["losses"]["train"][bs] = train_losses
    experiment_data["batch_size"]["SPR_BENCH"]["losses"]["dev"][bs] = dev_losses
    experiment_data["batch_size"]["SPR_BENCH"]["rgs"]["dev"][bs] = dv_rgs
    experiment_data["batch_size"]["SPR_BENCH"]["predictions"]["dev"][
        bs
    ] = dv_pred.tolist()
    loss_curve_dict[bs] = (train_losses, dev_losses)

# ----------------------- BEST MODEL TEST EVAL -----------------------------
print(f"\nBest batch size by dev accuracy: {best_bs} (acc={best_dev:.3f})")
final_model = AvgEmbedClassifier(vocab_size, embed_dim, num_classes).to(device)
final_model.load_state_dict(best_state)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"], vocab, label2id),
    batch_size=best_bs,
    shuffle=False,
    collate_fn=collate,
)
test_loss, test_acc, test_pred, test_seq, test_true = evaluate(
    final_model, criterion, test_loader
)
test_mask = compute_rgs_mask(test_seq)
test_rgs = (
    (test_pred[test_mask] == test_true[test_mask]).mean()
    if test_mask.sum() > 0
    else 0.0
)
swa = shape_weighted_accuracy(test_seq, test_true, test_pred)
cwa = color_weighted_accuracy(test_seq, test_true, test_pred)

print(
    f"TEST – loss={test_loss:.4f} acc={test_acc:.3f} RGS={test_rgs:.3f} "
    f"SWA={swa:.3f} CWA={cwa:.3f}"
)

d = experiment_data["batch_size"]["SPR_BENCH"]
d["metrics"]["test_acc"][best_bs] = test_acc
d["rgs"]["test"][best_bs] = test_rgs
d["predictions"]["test"][best_bs] = test_pred.tolist()
d["best_batch_size"] = best_bs

# ----------------------- SAVE ARTIFACTS -----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# plot all loss curves in one figure
plt.figure()
for bs, (tr, dev) in loss_curve_dict.items():
    plt.plot(tr, label=f"train_bs{bs}", alpha=0.7)
    plt.plot(dev, label=f"dev_bs{bs}", linestyle="--", alpha=0.7)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy")
plt.title("Loss Curves")
plt.savefig(os.path.join(working_dir, "SPR_loss_curve.png"))
print("All artifacts saved to", working_dir)
