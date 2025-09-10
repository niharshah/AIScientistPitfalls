import os, random, string, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset, DatasetDict

# working dir -----------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# GPU device ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------------------------
# Helper: try SPR_BENCH, else make synthetic data
# -----------------------------------------------------------------------------
def generate_synthetic_split(
    n_rows: int, n_labels: int = 5, min_len: int = 5, max_len: int = 12
):
    shapes = list(string.ascii_uppercase[:6])  # A-F
    colors = list("1234")  # 1-4
    seqs, labels = [], []
    for _ in range(n_rows):
        ln = random.randint(min_len, max_len)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
        seqs.append(" ".join(toks))
        labels.append(str(random.randint(0, n_labels - 1)))
    return {"id": list(range(n_rows)), "sequence": seqs, "label": labels}


def load_data():
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        if DATA_PATH.exists():
            return load_spr_bench(DATA_PATH)
        raise FileNotFoundError
    except Exception:
        print("Falling back to synthetic SPR data.")
        d = DatasetDict()
        d["train"] = HFDataset.from_dict(generate_synthetic_split(2000))
        d["dev"] = HFDataset.from_dict(generate_synthetic_split(500))
        d["test"] = HFDataset.from_dict(generate_synthetic_split(1000))
        return d


spr = load_data()
print({k: len(v) for k, v in spr.items()})

# -----------------------------------------------------------------------------
# Vocabulary & encoding
# -----------------------------------------------------------------------------
PAD = "<PAD>"
UNK = "<UNK>"


def build_vocab(dataset):
    vocab = {PAD: 0, UNK: 1}
    for seq in dataset["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size: {len(vocab)}, Num classes: {num_classes}")


def encode_sequence(seq, max_len=20):
    ids = [vocab.get(tok, vocab[UNK]) for tok in seq.split()[:max_len]]
    if len(ids) < max_len:
        ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


max_len = 20
for split in spr:
    spr[split] = spr[split].map(
        lambda ex: {
            "input_ids": encode_sequence(ex["sequence"], max_len),
            "label_id": int(ex["label"]),
        }
    )


# -----------------------------------------------------------------------------
# Torch Dataset
# -----------------------------------------------------------------------------
class SPRTorch(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["label_id"], dtype=torch.long),
        }


train_loader = DataLoader(SPRTorch(spr["train"]), batch_size=128, shuffle=True)
dev_loader = DataLoader(SPRTorch(spr["dev"]), batch_size=256)
test_loader = DataLoader(SPRTorch(spr["test"]), batch_size=256)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class AvgPoolClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, ids):
        emb = self.emb(ids)  # (B,L,E)
        mask = (ids != 0).unsqueeze(-1)  # (B,L,1)
        summed = (emb * mask).sum(1)
        lens = mask.sum(1).clamp(min=1)
        avg = summed / lens
        return self.fc(avg)


model = AvgPoolClassifier(len(vocab), 32, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------------------------------------------------------
# experiment data dict
# -----------------------------------------------------------------------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
epochs = 5


def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(batch["input_ids"])
            loss = criterion(out, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        preds = out.argmax(1)
        total_loss += loss.item() * len(batch["labels"])
        correct += (preds == batch["labels"]).sum().item()
        total += len(batch["labels"])
    return total_loss / total, correct / total


for epoch in range(1, epochs + 1):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    val_loss, val_acc = run_epoch(dev_loader, False)
    experiment_data["SPR"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["metrics"]["train"].append(tr_acc)
    experiment_data["SPR"]["metrics"]["val"].append(val_acc)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_acc = {val_acc:.3f}")

# -----------------------------------------------------------------------------
# Test evaluation & URA
# -----------------------------------------------------------------------------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch["input_ids"].to(device)
        logits = model(inputs)
        preds = logits.argmax(1).cpu().numpy().tolist()
        labels = batch["labels"].cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))
train_seen = set(spr["train"]["label_id"])
unseen_idx = [i for i, l in enumerate(all_labels) if l not in train_seen]
ura = (
    np.mean([all_preds[i] == all_labels[i] for i in unseen_idx]) if unseen_idx else 0.0
)
print(f"Test Accuracy: {overall_acc:.3f}")
print(f"Unseen-Rule Accuracy (URA): {ura:.3f}")

experiment_data["SPR"]["predictions"] = all_preds
experiment_data["SPR"]["ground_truth"] = all_labels
experiment_data["SPR"]["metrics"]["test_acc"] = overall_acc
experiment_data["SPR"]["metrics"]["URA"] = ura

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
