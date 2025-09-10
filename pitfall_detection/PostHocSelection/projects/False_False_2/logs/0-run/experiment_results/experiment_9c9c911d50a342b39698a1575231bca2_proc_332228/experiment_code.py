import os, pathlib, warnings, string, random, time
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------- working dir + device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------- try importing official helpers
try:
    from SPR import load_spr_bench, shape_weighted_accuracy
except Exception as e:
    warnings.warn(
        f"SPR helpers not found ({e}); fallback implementations will be used."
    )

    def load_spr_bench(_):
        raise FileNotFoundError

    def count_shape_variety(seq: str):
        return len(set(tok[0] for tok in seq.strip().split() if tok))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        weights = [count_shape_variety(s) for s in seqs]
        correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
        return sum(correct) / (sum(weights) + 1e-9)


# ------------------------------------------------- synthetic fallback
def synthetic_data(n_rows=512):
    shapes = list(string.ascii_uppercase[:6])
    colors = list(string.ascii_lowercase[:6])
    seqs, labels = [], []
    for _ in range(n_rows):
        ln = random.randint(4, 10)
        tokens = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
        seqs.append(" ".join(tokens))
        labels.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labels}


# ------------------------------------------------- load SPR_BENCH or synthetic
root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    dsets = load_spr_bench(root)
    print("Loaded SPR_BENCH located at", root)
    train_seqs, train_labels = dsets["train"]["sequence"], dsets["train"]["label"]
    dev_seqs, dev_labels = dsets["dev"]["sequence"], dsets["dev"]["label"]
    test_seqs, test_labels = dsets["test"]["sequence"], dsets["test"]["label"]
except Exception as e:
    warnings.warn(f"Could not load SPR_BENCH ({e}); using synthetic data.")
    tr = synthetic_data(1024)
    dv = synthetic_data(256)
    te = synthetic_data(256)
    train_seqs, train_labels = tr["sequence"], tr["label"]
    dev_seqs, dev_labels = dv["sequence"], dv["label"]
    test_seqs, test_labels = te["sequence"], te["label"]

n_classes = int(max(max(train_labels), max(dev_labels), max(test_labels))) + 1

# ------------------------------------------------- vocabulary on shapes (first char of token)
shape_vocab = sorted({tok[0] for seq in train_seqs for tok in seq.split()})
shape2id = {s: i + 1 for i, s in enumerate(shape_vocab)}  # 0 reserved for PAD
pad_id = 0
print(f"Shape vocab size = {len(shape2id)}")


# ------------------------------------------------- dataset object
class SPRShapeDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.hist_len = len(shape2id)

    def __len__(self):
        return len(self.sequences)

    def encode_seq(self, seq):
        ids = [shape2id.get(tok[0], 0) for tok in seq.split() if tok]
        hist = np.zeros(self.hist_len, dtype=np.float32)
        for idx in ids:
            if idx > 0:
                hist[idx - 1] += 1.0
        return ids, hist

    def __getitem__(self, idx):
        ids, hist = self.encode_seq(self.sequences[idx])
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "hist": torch.tensor(hist, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.sequences[idx],  # keep for metric
        }


def collate(batch):
    max_len = max(len(item["ids"]) for item in batch)
    seqs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    hists = torch.stack([item["hist"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    lengths = []
    raws = []
    for i, item in enumerate(batch):
        l = len(item["ids"])
        seqs[i, :l] = item["ids"]
        lengths.append(l)
        raws.append(item["raw_seq"])
    return {
        "ids": seqs,
        "len": torch.tensor(lengths),
        "hist": hists,
        "label": labels,
        "raw_seq": raws,
    }


batch_size = 128
train_loader = DataLoader(
    SPRShapeDataset(train_seqs, train_labels),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRShapeDataset(dev_seqs, dev_labels),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRShapeDataset(test_seqs, test_labels),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)


# ------------------------------------------------- neural-symbolic model
class NeuralSymbolic(nn.Module):
    def __init__(self, vocab_size, embed_dim, hid_dim, hist_dim, n_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hid_dim * 2 + hist_dim, 128)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, ids, lengths, hist):
        emb = self.embed(ids)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h: (2, B, hid)
        h_cat = torch.cat([h[0], h[1]], dim=1)  # (B, 2*hid)
        out = torch.cat([h_cat, hist], dim=1)
        out = self.act(self.fc1(out))
        return self.fc2(out)


model = NeuralSymbolic(len(shape2id), 16, 32, len(shape2id), n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ------------------------------------------------- experiment tracking
experiment_data = {
    "SPR_neural_symbolic": {
        "metrics": {"train_SWA": [], "dev_SWA": []},
        "losses": {"train": [], "dev": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
        "test_metrics": {},
    }
}


# ------------------------------------------------- helper to compute SWA on a loader
def eval_loader(loader):
    model.eval()
    all_preds, all_gts, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device)
            ln = batch["len"].to(device)
            hist = batch["hist"].to(device)
            logits = model(ids, ln, hist)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_gts.extend(batch["label"].numpy())
            all_seqs.extend(batch["raw_seq"])
    swa = shape_weighted_accuracy(all_seqs, all_gts, all_preds)
    return swa, np.array(all_preds)


# ------------------------------------------------- training loop with early stopping
max_epochs = 40
patience = 6
best_dev_swa, wait = -1.0, 0
best_state = None

for epoch in range(1, max_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        ids = batch["ids"].to(device)
        ln = batch["len"].to(device)
        hist = batch["hist"].to(device)
        y = batch["label"].to(device)
        logits = model(ids, ln, hist)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * ids.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_swa, _ = eval_loader(train_loader)
    dev_swa, _ = eval_loader(dev_loader)

    # log
    log = experiment_data["SPR_neural_symbolic"]
    log["epochs"].append(epoch)
    log["losses"]["train"].append(train_loss)
    log["metrics"]["train_SWA"].append(train_swa)
    log["metrics"]["dev_SWA"].append(dev_swa)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}  dev_SWA={dev_swa:.4f}")

    # early stopping
    if dev_swa > best_dev_swa + 1e-4:
        best_dev_swa = dev_swa
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ------------------------------------------------- restore best & evaluate on test
if best_state is not None:
    model.load_state_dict(best_state)

test_swa, test_preds = eval_loader(test_loader)
print(f"\nTest Shape-Weighted Accuracy (SWA) = {test_swa:.4f}")

# fill remaining experiment data and save
log = experiment_data["SPR_neural_symbolic"]
log["losses"]["dev"] = log["losses"].get("dev", [])  # ensure key exists
log["metrics"]["dev_SWA"] = log["metrics"]["dev_SWA"]
log["predictions"] = test_preds
log["ground_truth"] = np.array(test_labels)
log["test_metrics"] = {"SWA": test_swa}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment artefacts saved to ./working")
