import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict
from typing import List, Dict

# ------------------ working dir ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------ helper -----------------------
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
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def rcwa(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ------------------ load dataset -----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------- build vocabulary ------------------
def build_vocab(dataset) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print(f"Vocab size: {len(vocab)}")


# ------------- dataset class ---------------------
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
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.strip().split()]

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encode(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sequence_str": self.seqs[idx],
        }


train_ds, dev_ds, test_ds = [
    SPRTorchDataset(spr[s], vocab) for s in ("train", "dev", "test")
]


# ------------- collate fn ------------------------
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


# ------------- model -----------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, ids, mask):
        emb = self.embed(ids)
        masked = emb * mask.unsqueeze(-1)
        mean_emb = masked.sum(1) / mask.sum(1).clamp(min=1e-6).unsqueeze(-1)
        return self.fc(mean_emb)


num_classes = int(max(train_ds.labels)) + 1
model = MeanPoolClassifier(len(vocab), 64, num_classes).to(device)

# ------------- training setup --------------------
criterion = nn.CrossEntropyLoss()
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)  # swapped optimiser

experiment_data = {
    "optimizer_adamw": {
        "SPR_BENCH": {
            "metrics": {"train_rcwa": [], "val_rcwa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}
exp_ref = experiment_data["optimizer_adamw"]["SPR_BENCH"]


def evaluate(loader):
    model.eval()
    tot_loss, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
            seqs.extend(batch["sequence_str"])
    return tot_loss / len(loader.dataset), rcwa(seqs, gts, preds), preds, gts, seqs


EPOCHS = 5
for ep in range(1, EPOCHS + 1):
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
    val_loss, val_rcwa, *_ = evaluate(dev_loader)
    exp_ref["losses"]["train"].append(train_loss)
    exp_ref["losses"]["val"].append(val_loss)
    exp_ref["metrics"]["train_rcwa"].append(np.nan)
    exp_ref["metrics"]["val_rcwa"].append(val_rcwa)
    exp_ref["timestamps"].append(time.time())
    print(f"Epoch {ep}: val_loss={val_loss:.4f}  RCWA={val_rcwa:.4f}")

# ------------- final test evaluation -------------
test_loss, test_rcwa, test_preds, test_gts, test_seqs = evaluate(test_loader)
swa = sum(
    count_shape_variety(s) if y == p else 0
    for s, y, p in zip(test_seqs, test_gts, test_preds)
) / sum(count_shape_variety(s) for s in test_seqs)
cwa = sum(
    count_color_variety(s) if y == p else 0
    for s, y, p in zip(test_seqs, test_gts, test_preds)
) / sum(count_color_variety(s) for s in test_seqs)
print(f"Test loss={test_loss:.4f}  RCWA={test_rcwa:.4f}  SWA={swa:.4f}  CWA={cwa:.4f}")

exp_ref["predictions"] = np.array(test_preds)
exp_ref["ground_truth"] = np.array(test_gts)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
