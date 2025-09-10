import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict
from typing import List, Dict

# ------------------ reproducibility ------------------
torch.manual_seed(42)
np.random.seed(42)

# ------------------ working dir ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ device ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------ helper ---------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
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
    return len(set(t[0] for t in sequence.strip().split() if t))


def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def rcwa(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ------------------ load dataset ---------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------------ vocabulary -----------------------
def build_vocab(dataset) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print(f"Vocab size: {len(vocab)}")


# ------------------ torch dataset --------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq: str) -> List[int]:
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.strip().split()]

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encode(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sequence_str": self.seqs[idx],
        }


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr["train"], vocab),
    SPRTorchDataset(spr["dev"], vocab),
    SPRTorchDataset(spr["test"], vocab),
)


# ------------------ collate fn -----------------------
def collate_fn(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    seq_str = [b["sequence_str"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    mask = (padded != 0).float()
    return {
        "input_ids": padded,
        "attention_mask": mask,
        "labels": labels,
        "sequence_str": seq_str,
    }


BATCH_SIZE = 128
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# ------------------ model ----------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, dropout_prob: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, ids, mask):
        emb = self.embed(ids)  # B x T x D
        masked = emb * mask.unsqueeze(-1)  # zero-out pads
        sum_emb = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)
        mean_emb = sum_emb / lengths
        mean_emb = self.dropout(mean_emb)
        return self.fc(mean_emb)


# ------------------ evaluation -----------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total_loss, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch_t["input_ids"], batch_t["attention_mask"])
            loss = criterion(logits, batch_t["labels"])
            total_loss += loss.item() * batch_t["labels"].size(0)
            pred = logits.argmax(dim=-1).cpu().tolist()
            gt = batch_t["labels"].cpu().tolist()
            preds.extend(pred)
            gts.extend(gt)
            seqs.extend(batch["sequence_str"])
    rcwa_val = rcwa(seqs, gts, preds)
    return total_loss / len(loader.dataset), rcwa_val, preds, gts, seqs


# ------------------ hyperparameter sweep -------------
DROP_PROBS = [0.0, 0.1, 0.3, 0.5]
EPOCHS = 5
emb_dim = 64
num_classes = int(max(train_ds.labels)) + 1

experiment_data = {"dropout_prob": {}}

for dp in DROP_PROBS:
    print(f"\n=== Training with dropout_prob = {dp} ===")
    # fresh model & optimiser
    model = MeanPoolClassifier(len(vocab), emb_dim, num_classes, dp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # storage init
    exp_key = f"p_{dp}"
    experiment_data["dropout_prob"][exp_key] = {
        "SPR_BENCH": {
            "metrics": {"train_rcwa": [], "val_rcwa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
    record = experiment_data["dropout_prob"][exp_key]["SPR_BENCH"]

    # training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch_t["input_ids"], batch_t["attention_mask"])
            loss = criterion(logits, batch_t["labels"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_t["labels"].size(0)
        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss, val_rcwa, _, _, _ = evaluate(model, dev_loader)
        record["losses"]["train"].append(train_loss)
        record["losses"]["val"].append(val_loss)
        record["metrics"]["train_rcwa"].append(np.nan)  # placeholder
        record["metrics"]["val_rcwa"].append(val_rcwa)
        record["timestamps"].append(time.time())
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}  RCWA={val_rcwa:.4f}")

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
        f"TEST: loss={test_loss:.4f}  RCWA={test_rcwa:.4f}  SWA={swa:.4f}  CWA={cwa:.4f}"
    )

    # store predictions & ground truth
    record["predictions"] = np.array(test_preds)
    record["ground_truth"] = np.array(test_gts)
    record["test_metrics"] = {
        "loss": test_loss,
        "rcwa": test_rcwa,
        "swa": swa,
        "cwa": cwa,
    }

# --------------- save experiment data ---------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to 'experiment_data.npy'")
