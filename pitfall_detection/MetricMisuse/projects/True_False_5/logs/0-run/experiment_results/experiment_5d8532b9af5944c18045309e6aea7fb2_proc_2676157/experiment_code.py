import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset
from typing import List, Dict

# ------------------ working dir ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------ helper fns -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


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


train_ds = SPRTorchDataset(spr["train"], vocab)
dev_ds = SPRTorchDataset(spr["dev"], vocab)
test_ds = SPRTorchDataset(spr["test"], vocab)


# ------------- collate fn ------------------------
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


# ------------- model -----------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, ids, mask):
        emb = self.embed(ids) * mask.unsqueeze(-1)
        mean_emb = emb.sum(1) / mask.sum(1).clamp(min=1e-6).unsqueeze(-1)
        return self.fc(mean_emb)


# ---------------- experiment dict ----------------
experiment_data = {}

# ------------- evaluation fn ---------------------
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
            pred = logits.argmax(-1).cpu().tolist()
            gt = batch_t["labels"].cpu().tolist()
            preds.extend(pred)
            gts.extend(gt)
            seqs.extend(batch["sequence_str"])
    return total_loss / len(loader.dataset), rcwa(seqs, gts, preds), preds, gts, seqs


# ------------- hyperparameter sweep --------------
num_classes = int(max(train_ds.labels)) + 1
EPOCHS = 5
for wd in [0.0, 1e-5, 1e-4, 1e-3]:
    key = f"weight_decay_{wd}"
    experiment_data[key] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
            "weight_decay": wd,
        }
    }
    model = MeanPoolClassifier(len(vocab), 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    print(f"\n--- Training with weight_decay={wd} ---")
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
        experiment_data[key]["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data[key]["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data[key]["SPR_BENCH"]["metrics"]["train"].append(
            np.nan
        )  # placeholder
        experiment_data[key]["SPR_BENCH"]["metrics"]["val"].append(val_rcwa)
        experiment_data[key]["SPR_BENCH"]["timestamps"].append(time.time())
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  RCWA={val_rcwa:.4f}"
        )

    # ----------- final test evaluation ------------
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
        f"Test results (wd={wd}): loss={test_loss:.4f} RCWA={test_rcwa:.4f} SWA={swa:.4f} CWA={cwa:.4f}"
    )
    exp_block = experiment_data[key]["SPR_BENCH"]
    exp_block["predictions"] = np.array(test_preds)
    exp_block["ground_truth"] = np.array(test_gts)
    exp_block["test_metrics"] = {
        "loss": test_loss,
        "RCWA": test_rcwa,
        "SWA": swa,
        "CWA": cwa,
    }

# ------------- save all data ---------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved all experiment data to 'experiment_data.npy'")
