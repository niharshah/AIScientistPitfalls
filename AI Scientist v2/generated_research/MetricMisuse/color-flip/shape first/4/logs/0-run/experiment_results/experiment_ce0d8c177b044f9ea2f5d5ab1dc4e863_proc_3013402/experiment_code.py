import os, pathlib, random, time, math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# working directory for outputs
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# device handling (MUST always move tensors/models to device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Try to import the provided utility; if unavailable, create fallback
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except Exception as e:
    print("SPR utility not found, generating small synthetic dataset for demo.")

    def _make_synthetic_split(n):
        seqs, labels = [], []
        shapes = ["A", "B", "C"]
        colors = ["0", "1", "2"]
        for i in range(n):
            length = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            rule = seq[0]  # trivial rule: class is first token
            seqs.append(seq)
            labels.append(rule)
        return {"id": [str(i) for i in range(n)], "sequence": seqs, "label": labels}

    class _SyntheticDataset(Dataset):
        def __init__(self, split_dict):
            self.data = split_dict

        def __len__(self):
            return len(self.data["id"])

        def __getitem__(self, idx):
            return {
                "id": self.data["id"][idx],
                "sequence": self.data["sequence"][idx],
                "label": self.data["label"][idx],
            }

    def load_spr_bench(_):
        d = {
            "train": _SyntheticDataset(_make_synthetic_split(200)),
            "dev": _SyntheticDataset(_make_synthetic_split(80)),
            "test": _SyntheticDataset(_make_synthetic_split(80)),
        }
        return d

    def _count_variety(sequence, idx):  # idx=0 shape, idx=1 color
        return len(set(tok[idx] for tok in sequence.split() if len(tok) > idx))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [_count_variety(s, 0) for s in seqs]
        c = [w[i] if y_true[i] == y_pred[i] else 0 for i in range(len(w))]
        return sum(c) / sum(w) if sum(w) > 0 else 0

    def color_weighted_accuracy(seqs, y_true, y_pred):
        w = [_count_variety(s, 1) for s in seqs]
        c = [w[i] if y_true[i] == y_pred[i] else 0 for i in range(len(w))]
        return sum(c) / sum(w) if sum(w) > 0 else 0


# ------------------------------------------------------------------
# Load dataset
DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "./SPR_BENCH"))
dset = load_spr_bench(DATA_PATH)
print("Dataset loaded with splits:", dset.keys())


# ------------------------------------------------------------------
# Build vocabulary from training sequences
def tokenize(seq):
    return seq.strip().split()


vocab = {"<pad>": 0, "<unk>": 1}


def add_tokens(seq):
    for tok in tokenize(seq):
        if tok not in vocab:
            vocab[tok] = len(vocab)


for item in dset["train"]:
    add_tokens(item["sequence"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

label2id, id2label = {}, {}
for item in dset["train"]:
    lab = item["label"]
    if lab not in label2id:
        idx = len(label2id)
        label2id[lab] = idx
        id2label[idx] = lab
num_labels = len(label2id)
print(f"Num labels: {num_labels}")


# ------------------------------------------------------------------
# Torch Dataset wrapper
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.split = hf_split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[idx]
        ids = [vocab.get(tok, 1) for tok in tokenize(row["sequence"])]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(label2id[row["label"]], dtype=torch.long),
            "sequence": row["sequence"],
        }


def collate(batch):
    seq_lens = [len(ex["input_ids"]) for ex in batch]
    max_len = max(seq_lens)
    inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    sequences = [ex["sequence"] for ex in batch]
    for i, ex in enumerate(batch):
        l = len(ex["input_ids"])
        inputs[i, :l] = ex["input_ids"]
    return {
        "input_ids": inputs.to(device),
        "labels": labels.to(device),
        "sequences": sequences,
    }


train_loader = DataLoader(
    SPRTorchDataset(dset["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchDataset(dset["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorchDataset(dset["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ------------------------------------------------------------------
# Simple GRU classifier
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)  # h: (1,B,H)
        logits = self.fc(h.squeeze(0))
        return logits


model = GRUClassifier(vocab_size, emb_dim=64, hidden_dim=128, num_labels=num_labels).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------------
# experiment data store
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------------
# training loop
EPOCHS = 5


def evaluate(loader):
    model.eval()
    all_logits, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"])
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())
            all_seqs += batch["sequences"]
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds = all_logits.argmax(dim=1).numpy()
    gts = all_labels.numpy()
    pred_labels = [id2label[p] for p in preds]
    true_labels = [id2label[t] for t in gts]
    swa = shape_weighted_accuracy(all_seqs, true_labels, pred_labels)
    cwa = color_weighted_accuracy(all_seqs, true_labels, pred_labels)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0
    loss = criterion(all_logits, all_labels).item()
    return loss, swa, cwa, hwa, pred_labels, true_labels


for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, steps = 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    train_loss = total_loss / steps
    val_loss, swa, cwa, hwa, _, _ = evaluate(dev_loader)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  SWA={swa:.4f}  CWA={cwa:.4f}  HWA={hwa:.4f}"
    )

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"epoch": epoch, "loss": train_loss}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "loss": val_loss, "swa": swa, "cwa": cwa, "hwa": hwa}
    )

# Final evaluation on test set
test_loss, swa, cwa, hwa, preds, truths = evaluate(test_loader)
print(f"\nTEST: loss={test_loss:.4f}  SWA={swa:.4f}  CWA={cwa:.4f}  HWA={hwa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = truths
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "loss": test_loss,
    "swa": swa,
    "cwa": cwa,
    "hwa": hwa,
}

# ------------------------------------------------------------------
# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {working_dir}/experiment_data.npy")
