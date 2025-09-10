import os, random, pathlib, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# working directory & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# (1) helper – metrics & dataset loader (copied from prompt)
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 2 * swa * cwa / max(swa + cwa, 1e-8)


# ------------------------------------------------------------------
# (2) load SPR_BENCH or create synthetic fallback
def try_load_real():
    root = pathlib.Path(os.getenv("SPR_PATH", "SPR_BENCH"))
    if not root.exists():
        raise FileNotFoundError

    def _load(split):
        fn = root / f"{split}.csv"
        ids, seqs, labels = [], [], []
        with open(fn) as f:
            next(f)  # skip header
            for line in f:
                i, s, l = line.rstrip("\n").split(",", 2)
                ids.append(i)
                seqs.append(s)
                labels.append(l)
        return ids, seqs, labels

    data = {sp: _load(sp) for sp in ["train", "dev", "test"]}
    return data


def make_synth(n=1000):
    shapes = list("ABCDEFG")
    colors = list("xyzuvw")
    labels = list("01234")  # 5 classes

    def rand_seq():
        tks = [
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 9))
        ]
        return " ".join(tks)

    def rand_label(seq):
        return str(len(seq.split()) % len(labels))

    ids, seqs, lbs = [], [], []
    for i in range(n):
        s = rand_seq()
        ids.append(str(i))
        seqs.append(s)
        lbs.append(rand_label(s))
    split = int(0.8 * n)
    return {
        "train": (ids[:split], seqs[:split], lbs[:split]),
        "dev": (ids[split:], seqs[split:], lbs[split:]),
        "test": (ids[split:], seqs[split:], lbs[split:]),
    }


try:
    dataset = try_load_real()
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Falling back to synthetic data.", e)
    dataset = make_synth(1200)

# ------------------------------------------------------------------
# (3) build vocabulary & encode
all_tokens = set()
for seq in dataset["train"][1]:
    all_tokens.update(seq.split())
vocab = {
    "<pad>": 0,
    "<unk>": 1,
    **{tok: i + 2 for i, tok in enumerate(sorted(all_tokens))},
}
num_classes = len(set(dataset["train"][2]))
max_len = min(50, max(len(s.split()) for s in dataset["train"][1]))


def encode(seq):
    ids = [vocab.get(t, 1) for t in seq.split()][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


# ------------------------------------------------------------------
# (4) PyTorch Dataset
class SPRTorchDataset(Dataset):
    def __init__(self, ids, seqs, labels):
        self.ids, self.seqs, self.labels = ids, seqs, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "sequence": self.seqs[idx],
        }


train_ds = SPRTorchDataset(*dataset["train"])
dev_ds = SPRTorchDataset(*dataset["dev"])

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_dl = DataLoader(dev_ds, batch_size=256)


# ------------------------------------------------------------------
# (5) Model
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hid_dim, num_cls):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, num_cls)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h_cat)


model = BiGRUClassifier(len(vocab), 64, 64, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------------
# (6) tracking containers
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------------------------
# (7) training & evaluation loops
def evaluate(dloader):
    model.eval()
    losses, preds, gts, seqs = [], [], [], []
    with torch.no_grad():
        for batch in dloader:
            seqs.extend(batch["sequence"])
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            out = model(x)
            loss = criterion(out, y)
            losses.append(loss.item())
            pred = out.argmax(1).cpu().tolist()
            preds.extend(pred)
            gts.extend(y.cpu().tolist())
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(seqs, gts, preds)
    return np.mean(losses), swa, cwa, hwa, preds, gts


num_epochs = 5
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_losses = []
    for batch in train_dl:
        optimizer.zero_grad()
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch["input_ids"])
        loss = criterion(out, batch["label"])
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    train_loss = np.mean(epoch_losses)

    val_loss, swa, cwa, hwa, preds, gts = evaluate(dev_dl)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"swa": swa, "cwa": cwa, "hwa": hwa, "epoch": epoch}
    )
    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = gts

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  SWA={swa:.3f}  CWA={cwa:.3f}  HWA={hwa:.3f}"
    )

# ------------------------------------------------------------------
# (8) save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", working_dir)
