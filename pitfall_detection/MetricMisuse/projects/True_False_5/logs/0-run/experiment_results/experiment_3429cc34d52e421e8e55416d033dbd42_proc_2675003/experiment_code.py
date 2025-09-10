import os, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# ----- helper code from proposal -------------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def rule_complexity_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return float(sum(correct)) / float(sum(weights)) if sum(weights) else 0.0


# ------------- load SPR_BENCH ----------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

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


# attempt both provided absolute path and local folder
possible_paths = [
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
    pathlib.Path("SPR_BENCH/"),
]
dataset_path = None
for p in possible_paths:
    if p.exists():
        dataset_path = p
        break
if dataset_path is None:
    raise FileNotFoundError("Cannot locate SPR_BENCH folder.")

spr = load_spr_bench(dataset_path)
print("Loaded splits:", spr.keys(), {k: len(v) for k, v in spr.items()})

# ------------- vocabulary & label mapping ----------------------------------------
PAD, UNK = 0, 1
tok2id = {}


def add_token(t):
    if t not in tok2id:
        tok2id[t] = len(tok2id) + 2  # reserve 0,1


for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        add_token(tok)
vocab_size = len(tok2id) + 2
print(f"Vocabulary size (incl PAD/UNK): {vocab_size}")

train_labels = spr["train"]["label"]
label_set = sorted(set(train_labels))
label2id = {lab: i for i, lab in enumerate(label_set)}
id2label = {i: lab for lab, i in label2id.items()}
num_labels = len(label2id)
print(f"Num classes: {num_labels}")


def encode_sequence(seq):
    ids = []
    for tok in seq.split():
        ids.append(tok2id.get(tok, UNK))
    return ids


# ------------- Dataset wrappers --------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]
        self.labels = [label2id[l] for l in hf_ds["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode_sequence(self.seqs[idx]), dtype=torch.long
            ),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    lengths = [len(x["input_ids"]) for x in batch]
    max_len = max(lengths)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        seq = x["input_ids"]
        input_ids[i, : len(seq)] = seq
    labels = torch.stack([x["label"] for x in batch])
    raw_seqs = [x["raw_seq"] for x in batch]
    return {"input_ids": input_ids, "labels": labels, "raw_seqs": raw_seqs}


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
val_loader = DataLoader(
    SPRTorchDataset(spr["dev"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)


# ------------- Model -------------------------------------------------------------
class AvgEmbClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=PAD)
        self.fc = nn.Linear(emb_dim, num_labels)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)  # (B,L,D)
        mask = (input_ids != PAD).unsqueeze(-1)  # (B,L,1)
        summed = (emb * mask).sum(1)  # (B,D)
        lengths = mask.sum(1).clamp(min=1)
        pooled = summed / lengths
        return self.fc(pooled)


model = AvgEmbClassifier(vocab_size, 32, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

# ------------- experiment data dict ---------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------- training ----------------------------------------------------------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in train_loader:
        batch_t = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch_t["input_ids"])
        loss = criterion(logits, batch_t["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg_train_loss = total_loss / n_batches

    def evaluate(dataloader):
        model.eval()
        all_true, all_pred, all_seq = [], [], []
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in dataloader:
                batch_t = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch_t["input_ids"])
                loss = criterion(logits, batch_t["labels"])
                total_loss += loss.item()
                n += 1
                preds = logits.argmax(-1).cpu().tolist()
                trues = batch_t["labels"].cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(trues)
                all_seq.extend(batch["raw_seqs"])
        avg_loss = total_loss / n
        rcwa = rule_complexity_weighted_accuracy(all_seq, all_true, all_pred)
        return avg_loss, rcwa, all_true, all_pred, all_seq

    val_loss, val_rcwa, _, _, _ = evaluate(val_loader)

    print(
        f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
        f"validation_loss={val_loss:.4f}, val_RCWA={val_rcwa:.4f}"
    )

    # store metrics
    ts = time.time()
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ts, avg_train_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ts, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        (ts, None)
    )  # no RCWA for train
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((ts, val_rcwa))

# ------------- final test evaluation --------------------------------------------
test_loss, test_rcwa, gts, preds, seqs = evaluate(test_loader)
print(f"Final Test   : loss={test_loss:.4f}, RCWA={test_rcwa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
