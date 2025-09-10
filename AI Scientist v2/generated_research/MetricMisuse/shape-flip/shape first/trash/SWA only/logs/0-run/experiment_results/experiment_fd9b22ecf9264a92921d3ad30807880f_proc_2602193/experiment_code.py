import os, random, string, pathlib, time, json
from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# GPU/CPU selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################################################
# ---------- 1. DATA LOADING / SYNTHETIC FALLBACK ----------
############################################################
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")  # adapt if necessary
    spr = load_spr_bench(DATA_PATH)
    print("Loaded SPR_BENCH from disk.")
except Exception as e:
    print("Could not load real SPR_BENCH, creating a tiny synthetic dataset for demo.")

    def random_token():
        shape = random.choice(string.ascii_uppercase[:6])  # 6 shapes
        color = random.choice("0123")  # 4 colors
        return shape + color

    def make_example(idx):
        seq_len = random.randint(3, 8)
        seq = " ".join(random_token() for _ in range(seq_len))
        # simple synthetic rule: label 1 if number of unique shapes is even else 0
        label = int(len(set(tok[0] for tok in seq.split())) % 2 == 0)
        return {"id": idx, "sequence": seq, "label": label}

    def build_split(n, offset):
        return [make_example(offset + i) for i in range(n)]

    synthetic = DatasetDict()
    from datasets import Dataset as HFDataset

    synthetic["train"] = HFDataset.from_list(build_split(400, 0))
    synthetic["dev"] = HFDataset.from_list(build_split(100, 400))
    synthetic["test"] = HFDataset.from_list(build_split(200, 500))
    spr = synthetic

############################################################
# ---------- 2. VOCAB & DATASET WRAPPER --------------------
############################################################
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


def encode_sequence(seq, vocab):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab):
        self.seqs = [encode_sequence(s, vocab) for s in hf_split["sequence"]]
        self.labels = hf_split["label"]
        self.raw_sequences = hf_split["sequence"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.seqs[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "sequence_str": self.raw_sequences[idx],
        }


def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    inputs = []
    labels = []
    seq_str = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        inputs.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(item["label"])
        seq_str.append(item["sequence_str"])
    return {
        "input_ids": torch.stack(inputs),
        "label": torch.tensor(labels, dtype=torch.long),
        "sequence_str": seq_str,
    }


train_ds = SPRTorchDataset(spr["train"], vocab)
dev_ds = SPRTorchDataset(spr["dev"], vocab)
test_ds = SPRTorchDataset(spr["test"], vocab)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)


############################################################
# ---------- 3. MODEL --------------------------------------
############################################################
class BoWClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)  # [B, L, D]
        mask = (input_ids != 0).float().unsqueeze(-1)  # [B, L, 1]
        emb_sum = (emb * mask).sum(dim=1)  # [B, D]
        lengths = mask.sum(dim=1).clamp(min=1)  # [B,1]
        avg_emb = emb_sum / lengths  # [B, D]
        return self.fc(avg_emb)


model = BoWClassifier(len(vocab), 64, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


############################################################
# ---------- 4. METRIC HELPERS -----------------------------
############################################################
def rule_signature(sequence):
    shapes = "".join(sorted(set(tok[0] for tok in sequence.split())))
    colors = "".join(sorted(set(tok[1] for tok in sequence.split() if len(tok) > 1)))
    return shapes + "|" + colors


train_signatures = set(rule_signature(s) for s in spr["train"]["sequence"])


def compute_accuracy(preds, labels):
    return (preds == labels).sum() / len(labels)


def compute_NRGS(seqs, preds, labels):
    sigs = [rule_signature(s) for s in seqs]
    idxs = [i for i, sig in enumerate(sigs) if sig not in train_signatures]
    if len(idxs) == 0:
        return 0.0
    return (preds[idxs] == labels[idxs]).sum() / len(idxs)


############################################################
# ---------- 5. EXPERIMENT DATA STORAGE --------------------
############################################################
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "val_loss": [], "NRGS": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

############################################################
# ---------- 6. TRAINING LOOP ------------------------------
############################################################
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["label"].size(0)
    train_acc = correct / total
    experiment_data["SPR_BENCH"]["losses"]["train"].append(epoch_loss / total)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)

    # ---------- Validation ----------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * batch["label"].size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["label"]).sum().item()
            total += batch["label"].size(0)
    val_acc = correct / total
    val_loss_avg = val_loss / total
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss_avg)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())
    print(
        f"Epoch {epoch}: validation_loss = {val_loss_avg:.4f}, val_acc = {val_acc:.4f}"
    )

############################################################
# ---------- 7. TEST EVALUATION & NRGS ---------------------
############################################################
model.eval()
all_preds = []
all_labels = []
all_seqs = []
with torch.no_grad():
    for batch in test_loader:
        batch_gpu = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch_gpu["input_ids"])
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = batch["label"].numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_seqs.extend(batch["sequence_str"])
test_acc = compute_accuracy(np.array(all_preds), np.array(all_labels))
NRGS = compute_NRGS(all_seqs, np.array(all_preds), np.array(all_labels))
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Novel Rule Generalization Score (NRGS): {NRGS:.4f}")

experiment_data["SPR_BENCH"]["metrics"]["NRGS"].append(NRGS)
experiment_data["SPR_BENCH"]["predictions"] = all_preds
experiment_data["SPR_BENCH"]["ground_truth"] = all_labels

############################################################
# ---------- 8. SAVE EVERYTHING ----------------------------
############################################################
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data to ./working/experiment_data.npy")
