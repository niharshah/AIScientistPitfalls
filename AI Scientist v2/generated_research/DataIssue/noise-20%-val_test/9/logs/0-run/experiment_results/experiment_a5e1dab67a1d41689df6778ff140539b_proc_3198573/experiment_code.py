import os, pathlib, random, string, time, json
import numpy as np
from collections import Counter, defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------- house-keeping ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "val_rba": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "rules": {},
    }
}


# -------------------------- load / fallback synthetic data ------------------------
def try_load_spr():
    try:
        from SPR import load_spr_bench  # uses provided helper

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        return load_spr_bench(DATA_PATH)
    except Exception as e:
        print("Could not load SPR_BENCH, generating synthetic data instead.")

        # create trivial synthetic problem: classify sequences length parity
        def gen(n):
            seqs, labels = [], []
            for i in range(n):
                l = random.randint(1, 15)
                s = "".join(random.choice(string.ascii_lowercase[:6]) for _ in range(l))
                seqs.append(s)
                labels.append(l % 2)  # 0 even, 1 odd
            return {"id": list(range(n)), "sequence": seqs, "label": labels}

        d = {}
        d["train"] = gen(2000)
        d["dev"] = gen(400)
        d["test"] = gen(400)
        # wrap in simple Dataset-like dict
        return {split: d[split] for split in ["train", "dev", "test"]}


dset = try_load_spr()


# -----------------------------------------------------------------------------
# Build vocabulary (character level)
def build_vocab(seqs):
    chars = set()
    for s in seqs:
        chars.update(list(s))
    vocab = {ch: i for i, ch in enumerate(sorted(chars))}
    return vocab


all_train_seqs = (
    dset["train"]["sequence"]
    if isinstance(dset["train"], dict)
    else dset["train"]["sequence"]
)
vocab = build_vocab(all_train_seqs)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


def vectorize(seqs):
    rows = np.zeros((len(seqs), vocab_size), dtype=np.float32)
    for i, s in enumerate(seqs):
        for ch, cnt in Counter(s).items():
            if ch in vocab:
                rows[i, vocab[ch]] = cnt
    return rows


def make_tensor_dataset(split):
    if isinstance(dset[split], dict):  # synthetic fallback
        seqs = dset[split]["sequence"]
        labels = dset[split]["label"]
    else:
        seqs = dset[split]["sequence"]
        labels = dset[split]["label"]
    X = vectorize(seqs)
    y = np.array(labels, dtype=np.int64)
    return TensorDataset(torch.tensor(X), torch.tensor(y))


train_ds = make_tensor_dataset("train")
val_ds = make_tensor_dataset("dev")
test_ds = make_tensor_dataset("test")

batch_size = 256
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

num_classes = len(set([y.item() for _, y in train_ds]))
print(f"Num classes: {num_classes}")


# ------------------------------- model ----------------------------------------
class Logistic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


model = Logistic(vocab_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


# ------------------------------ helpers ---------------------------------------
def accuracy(loader, model=None, rule_based=False, rules=None):
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            if rule_based:
                # rule prediction using extracted rules dict {class:[chars]}
                preds = []
                for vec in xb.numpy():
                    scores = [0] * num_classes
                    for cls, chars in rules.items():
                        for ch in chars:
                            idx = vocab.get(ch, None)
                            if idx is not None and vec[idx] > 0:
                                scores[cls] += 1
                    # default fallback: majority class 0 if tie
                    preds.append(int(np.argmax(scores)))
                preds = torch.tensor(preds)
            else:
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu()
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total


# ------------------------------ training loop ---------------------------------
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(yb)
        epoch_correct += (logits.argmax(1) == yb).sum().item()
        epoch_total += len(yb)
    train_loss = epoch_loss / epoch_total
    train_acc = epoch_correct / epoch_total

    model.eval()
    val_loss, val_total = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * len(yb)
            val_total += len(yb)
    val_loss = val_loss / val_total
    val_acc = accuracy(val_loader, model)

    # extract rules every epoch (top-3 chars per class)
    with torch.no_grad():
        W = model.fc.weight.cpu().numpy()  # shape C x V
    rules = defaultdict(list)
    for c in range(num_classes):
        diffs = W[c] - np.max(np.delete(W, c, axis=0), axis=0)
        top_idx = diffs.argsort()[-3:][::-1]
        rules[c] = [list(vocab.keys())[i] for i in top_idx]
    val_rba = accuracy(val_loader, rule_based=True, rules=rules)

    # logging
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_rba"].append(val_rba)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["rules"] = rules

    print(
        f"Epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} val_rba={val_rba:.3f} val_loss={val_loss:.4f}"
    )

# --------------------------- final evaluation ----------------------------------
test_acc = accuracy(test_loader, model)
test_rba = accuracy(
    test_loader, rule_based=True, rules=experiment_data["SPR_BENCH"]["rules"]
)
print(f"\nTest neural accuracy: {test_acc:.3f}")
print(f"Test Rule-Based Accuracy (RBA): {test_rba:.3f}")

# Save predictions on test set (neural model) for completeness
model.eval()
preds, gts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds.extend(logits.argmax(1).cpu().numpy().tolist())
        gts.extend(yb.numpy().tolist())
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# --------------------------- persist experiment data --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to 'working/experiment_data.npy'")
