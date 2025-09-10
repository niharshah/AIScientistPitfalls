import os, pathlib, random, time, string, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset

# ----------------- workspace dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- device ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- attempt to load benchmark -----
def load_spr_bench_folder(root: pathlib.Path) -> DatasetDict:
    from SPR import load_spr_bench  # provided utility

    return load_spr_bench(root)


def make_synthetic_split(n: int, rng: random.Random):
    shapes = list(string.ascii_uppercase[:6])  # 6 shapes
    colors = ["0", "1", "2", "3"]  # 4 colours
    seqs, labels = [], []
    for _ in range(n):
        length = rng.randint(4, 10)
        seq = " ".join(rng.choice(shapes) + rng.choice(colors) for _ in range(length))
        label = rng.randint(0, 1)  # binary synthetic rule
        seqs.append(seq)
        labels.append(label)
    return {"id": list(range(n)), "sequence": seqs, "label": labels}


def load_data():
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        dset = load_spr_bench_folder(DATA_PATH)
        print("Loaded real SPR_BENCH.")
    except Exception as e:
        print("Could not load SPR_BENCH, building synthetic toy data.", e)
        rng = random.Random(0)
        d_train = make_synthetic_split(2000, rng)
        d_dev = make_synthetic_split(400, rng)
        d_test = make_synthetic_split(400, rng)
        dset = DatasetDict(
            {
                "train": Dataset.from_dict(d_train),
                "dev": Dataset.from_dict(d_dev),
                "test": Dataset.from_dict(d_test),
            }
        )
    return dset


dset = load_data()

# ----------------- vocab & tokenisation ----------
PAD = "<pad>"
UNK = "<unk>"


def build_vocab(sequences):
    vocab = {PAD: 0, UNK: 1}
    for seq in sequences:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(dset["train"]["sequence"])
print(f"Vocab size: {len(vocab)}")


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.strip().split()]


# map datasets to encodings
for split in dset:
    dset[split] = dset[split].map(lambda ex: {"input_ids": encode(ex["sequence"])})


# ----------------- DataLoader --------------------
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = [x + [vocab[PAD]] * (max_len - len(x)) for x in ids]
    return {"input_ids": torch.tensor(padded, dtype=torch.long), "labels": labels}


train_loader = DataLoader(
    dset["train"], batch_size=64, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dset["dev"], batch_size=128, shuffle=False, collate_fn=collate)


# ----------------- Model -------------------------
class SPRModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hid_dim=128, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        logits = self.out(h.squeeze(0))
        return logits


model = SPRModel(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------- augmentation utils ------------
def augment_sequence(seq, rng=random):
    tokens = seq.strip().split()
    shapes = list({t[0] for t in tokens})
    colors = list({t[1] for t in tokens if len(t) > 1})
    shape_map = {s: rng.choice(string.ascii_uppercase) for s in shapes}
    color_map = {c: rng.choice("0123456789") for c in colors}
    aug_tokens = [
        shape_map[t[0]] + (color_map[t[1]] if len(t) > 1 else "") for t in tokens
    ]
    return " ".join(aug_tokens)


def augmentation_consistency(model, sequences, labels, k=3):
    model.eval()
    correct_fractions = []
    with torch.no_grad():
        for seq, y in zip(sequences, labels):
            variants = [seq] + [augment_sequence(seq, random) for _ in range(k)]
            enc = [encode(s) for s in variants]
            max_len = max(len(e) for e in enc)
            padded = [e + [vocab[PAD]] * (max_len - len(e)) for e in enc]
            inp = torch.tensor(padded, dtype=torch.long).to(device)
            logits = model(inp)
            preds = logits.argmax(1).cpu().tolist()
            frac = sum(1 for p in preds if p == y) / (k + 1)
            correct_fractions.append(frac)
    return float(np.mean(correct_fractions))


# ----------------- experiment data dict ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "val_acs": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": dset["test"]["label"],
    }
}

# ----------------- training loop -----------------
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["labels"].size(0)
        preds = logits.argmax(1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    train_loss = epoch_loss / total
    train_acc = correct / total

    # ---- validation ----
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(1)
            val_correct += (preds == batch["labels"]).sum().item()
            val_total += batch["labels"].size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total
    val_acs = augmentation_consistency(
        model, dset["dev"]["sequence"], dset["dev"]["label"]
    )

    # ---- logging ----
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}, val_ACS = {val_acs:.4f}"
    )
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acs"].append(val_acs)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

# ----------------- test predictions --------------
test_loader = DataLoader(
    dset["test"], batch_size=128, shuffle=False, collate_fn=collate
)
model.eval()
preds_all = []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        preds_all.extend(logits.argmax(1).cpu().tolist())
experiment_data["SPR_BENCH"]["predictions"] = preds_all

# ----------------- save --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
