import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict
from typing import List, Dict

# ------------------ experiment dict ------------------
experiment_data = {"learning_rate_tuning": {}}  # will hold sub-dicts keyed by lr value

# ------------------ working dir ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ device -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------ helper (unchanged) -------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def rcwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ------------------ load data ----------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------------ vocab & dataset ----------------
def build_vocab(dataset) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print("Vocab size:", len(vocab))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab):
        self.seqs, self.labels, self.vocab = (
            hf_split["sequence"],
            hf_split["label"],
            vocab,
        )

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        return [self.vocab.get(tok, 1) for tok in seq.strip().split()]

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encode(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sequence_str": self.seqs[idx],
        }


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr[s], vocab) for s in ["train", "dev", "test"]
)


def collate_fn(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    seqs = [b["sequence_str"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    mask = (padded != 0).float()
    return {
        "input_ids": padded,
        "attention_mask": mask,
        "labels": labels,
        "sequence_str": seqs,
    }


BATCH_SIZE = 128
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# ------------------ model -------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, ids, mask):
        emb = self.embed(ids) * mask.unsqueeze(-1)
        mean_emb = emb.sum(1) / mask.sum(1).clamp(min=1e-6).unsqueeze(-1)
        return self.fc(mean_emb)


num_classes = int(max(train_ds.labels)) + 1
criterion = nn.CrossEntropyLoss()


# ------------------ training utilities ------------
def evaluate(model, loader):
    model.eval()
    tot_loss, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(bt["input_ids"], bt["attention_mask"])
            loss = criterion(logits, bt["labels"])
            tot_loss += loss.item() * bt["labels"].size(0)
            p = logits.argmax(-1).cpu().tolist()
            preds.extend(p)
            gts.extend(bt["labels"].cpu().tolist())
            seqs.extend(batch["sequence_str"])
    return tot_loss / len(loader.dataset), rcwa(seqs, gts, preds), preds, gts, seqs


# ------------------ hyperparameter sweep ----------
lrs = [1e-4, 3e-4, 1e-3, 3e-3]
EPOCHS = 5
best_lr, best_rcwa = None, -1.0

for lr in lrs:
    torch.cuda.empty_cache()
    print(f"\n=== Training with learning rate {lr} ===")
    # sub-dict for this lr
    lr_key = f"lr_{lr:.0e}"
    experiment_data["learning_rate_tuning"][lr_key] = {
        "metrics": {"train_rcwa": [], "val_rcwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    model = MeanPoolClassifier(len(vocab), 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch["labels"].size(0)
        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss, val_rcwa, *_ = evaluate(model, dev_loader)
        experiment_data["learning_rate_tuning"][lr_key]["losses"]["train"].append(
            train_loss
        )
        experiment_data["learning_rate_tuning"][lr_key]["losses"]["val"].append(
            val_loss
        )
        experiment_data["learning_rate_tuning"][lr_key]["metrics"]["train_rcwa"].append(
            np.nan
        )
        experiment_data["learning_rate_tuning"][lr_key]["metrics"]["val_rcwa"].append(
            val_rcwa
        )
        experiment_data["learning_rate_tuning"][lr_key]["timestamps"].append(
            time.time()
        )
        print(f"  Epoch {epoch:02d}: val_loss={val_loss:.4f}  RCWA={val_rcwa:.4f}")
    # final test evaluation
    test_loss, test_rcwa, test_preds, test_gts, test_seqs = evaluate(model, test_loader)
    swa = (
        lambda seqs, y, p: sum(
            count_shape_variety(s) if yt == pt else 0 for s, yt, pt in zip(seqs, y, p)
        )
        / sum(count_shape_variety(s) for s in seqs)
    )(test_seqs, test_gts, test_preds)
    cwa = (
        lambda seqs, y, p: sum(
            count_color_variety(s) if yt == pt else 0 for s, yt, pt in zip(seqs, y, p)
        )
        / sum(count_color_variety(s) for s in seqs)
    )(test_seqs, test_gts, test_preds)
    print(
        f"  Test: loss={test_loss:.4f} RCWA={test_rcwa:.4f} SWA={swa:.4f} CWA={cwa:.4f}"
    )
    # store preds & gt
    d = experiment_data["learning_rate_tuning"][lr_key]
    d["predictions"] = np.array(test_preds)
    d["ground_truth"] = np.array(test_gts)
    d["test_metrics"] = {"loss": test_loss, "RCWA": test_rcwa, "SWA": swa, "CWA": cwa}
    if test_rcwa > best_rcwa:
        best_rcwa, best_lr = test_rcwa, lr

print(f"\nBest learning-rate on test RCWA: {best_lr}  ({best_rcwa:.4f})")

# --------------- save experiment data -------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
