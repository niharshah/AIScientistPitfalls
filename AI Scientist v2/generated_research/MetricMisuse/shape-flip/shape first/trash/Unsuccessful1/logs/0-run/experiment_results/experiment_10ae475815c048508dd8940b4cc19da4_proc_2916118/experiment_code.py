import os, pathlib, math, random, time, json
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# obligatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# experiment data container
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Provided helper code (slightly wrapped so it works if missing)
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except ImportError:
    # minimal stubs for synthetic mode
    def load_spr_bench(root):
        raise FileNotFoundError

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        return np.mean(np.array(y_true) == np.array(y_pred))

    def color_weighted_accuracy(seqs, y_true, y_pred):
        return np.mean(np.array(y_true) == np.array(y_pred))


# Try to load real benchmark ---------------------------------------------------
DATA_PATH = pathlib.Path(os.getcwd()) / "SPR_BENCH"
print(f"Expecting dataset at {DATA_PATH}")
use_synthetic = False
try:
    spr_bench = load_spr_bench(DATA_PATH)
except Exception as e:
    print("Dataset not found, switching to synthetic toy data.", e)
    use_synthetic = True


# ------------------------------------------------------------------
def create_synthetic_split(n):
    shapes = list("ABCDEFG")
    colors = list("abcde")
    X, y = [], []
    for i in range(n):
        seq_len = random.randint(3, 10)
        seq = " ".join(
            [random.choice(shapes) + random.choice(colors) for _ in range(seq_len)]
        )
        # arbitrary rule: label=1 if number of unique shapes == number of unique colours else 0
        label = int(
            len(set([t[0] for t in seq.split()]))
            == len(set([t[1] for t in seq.split()]))
        )
        X.append(seq)
        y.append(label)
    return {"sequence": X, "label": y}


if use_synthetic:
    train_data = create_synthetic_split(2000)
    val_data = create_synthetic_split(400)
    test_data = create_synthetic_split(400)
    spr_bench = {"train": train_data, "dev": val_data, "test": test_data}


# ------------------------------------------------------------------
# Vocabulary & label mapping
def build_vocab(sequences: List[str]) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in sequences:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr_bench["train"]["sequence"])
print(f"Vocab size: {len(vocab)}")

labels = sorted(list(set(spr_bench["train"]["label"])))
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}
print(f"Label set: {labels}")

# ------------------------------------------------------------------
max_len = max(len(seq.split()) for seq in spr_bench["train"]["sequence"])
print(f"Max sequence length: {max_len}")


def encode_sequence(seq: str) -> List[int]:
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in seq.split()]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids[:max_len]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_ids = torch.tensor(encode_sequence(self.seqs[idx]), dtype=torch.long)
        label = torch.tensor(label2idx[self.labels[idx]], dtype=torch.long)
        return {"input": seq_ids, "label": label, "raw_seq": self.seqs[idx]}


batch_size = 128
train_loader = DataLoader(
    SPRDataset(spr_bench["train"]), batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    SPRDataset(spr_bench["dev"]), batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    SPRDataset(spr_bench["test"]), batch_size=batch_size, shuffle=False
)


# ------------------------------------------------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embed(x)  # B,L,D
        mask = (x != 0).float().unsqueeze(-1)  # B,L,1
        summed = (emb * mask).sum(1)  # B,D
        len_den = mask.sum(1).clamp(min=1e-6)  # B,1
        pooled = summed / len_den  # mean pooling
        return self.fc(pooled)


model = MeanPoolClassifier(len(vocab), 64, len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------------------------------------------
def evaluate(dataloader):
    model.eval()
    total_loss, total_items = 0.0, 0
    all_preds, all_targets, all_seqs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input"])
            loss = criterion(logits, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)
            total_items += batch["label"].size(0)
            preds = logits.argmax(-1).cpu().tolist()
            targets = batch["label"].cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_seqs.extend(batch["raw_seq"])
    avg_loss = total_loss / total_items
    swa = shape_weighted_accuracy(all_seqs, all_targets, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_targets, all_preds)
    hwa = 0.0 if (swa == 0 or cwa == 0) else 2 * swa * cwa / (swa + cwa)
    return avg_loss, swa, cwa, hwa, all_preds, all_targets, all_seqs


# ------------------------------------------------------------------
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["label"].size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_loss, swa, cwa, hwa, _, _, _ = evaluate(val_loader)

    # record
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"swa": swa, "cwa": cwa, "hwa": hwa}
    )
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, SWA={swa:.4f}, CWA={cwa:.4f}, HWA={hwa:.4f}"
    )

# ------------------------------------------------------------------
# final evaluation on test
test_loss, swa, cwa, hwa, preds, targets, seqs = evaluate(test_loader)
print(
    f"\nTest   : test_loss={test_loss:.4f}, SWA={swa:.4f}, CWA={cwa:.4f}, HWA={hwa:.4f}"
)

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = targets
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "swa": swa,
    "cwa": cwa,
    "hwa": hwa,
    "loss": test_loss,
}

# ------------------------------------------------------------------
# Plot learning curves
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"], label="val")
plt.legend()
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()

plt.figure()
hwa_vals = [m["hwa"] for m in experiment_data["SPR_BENCH"]["metrics"]["val"]]
plt.plot(hwa_vals)
plt.title("Validation HWA")
plt.xlabel("epoch")
plt.ylabel("HWA")
plt.savefig(os.path.join(working_dir, "val_hwa.png"))
plt.close()

# ------------------------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {working_dir}")
