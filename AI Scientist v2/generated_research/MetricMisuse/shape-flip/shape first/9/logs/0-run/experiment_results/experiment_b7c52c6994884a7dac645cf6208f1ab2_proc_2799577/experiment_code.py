import os, random, time, pathlib, numpy as np, torch
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader

# ------------------------------- housekeeping -----------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_swa": [], "val_swa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ------------------------------- SPR loader -------------------------------------------
def install_and_import_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError:
        import subprocess, sys

        print("Installing HF datasets …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "datasets"]
        )
    finally:
        from datasets import load_dataset, DatasetDict  # noqa: F401
    from datasets import load_dataset, DatasetDict

    return load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path):
    """
    Return a DatasetDict {'train','dev','test'} for one SPR_BENCH folder.
    """
    load_dataset, DatasetDict = install_and_import_datasets()

    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=os.path.join(root, ".cache_dsets"),
        )

    dset = DatasetDict()
    dset["train"], dset["dev"], dset["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(s) for s in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ----------------------------- fallback tiny dataset -----------------------------------
def synthetic_spr(n=1500):
    shapes, colors = "ABC", "rgb"

    def mk(num):
        seqs, labels = [], []
        for idx in range(num):
            length = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            # arbitrary parity rule on 'A' occurrences
            lbl = int(sum(tok[0] == "A" for tok in seq.split()) % 2)
            seqs.append(seq)
            labels.append(lbl)
        return {"id": list(range(num)), "sequence": seqs, "label": labels}

    return {
        "train": mk(int(n * 0.6)),
        "dev": mk(int(n * 0.2)),
        "test": mk(int(n * 0.2)),
    }


# -------------------------------- dataset or fallback ----------------------------------
DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "./SPR_BENCH"))
if DATA_PATH.exists():
    try:
        spr_bench = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH from:", DATA_PATH)
        # HuggingFace datasets → dict-like
        spr_dict = {
            split: {k: spr_bench[split][k] for k in spr_bench[split].column_names}
            for split in spr_bench
        }
    except Exception as e:
        print("Failed to load real benchmark, using synthetic. Reason:", e)
        spr_dict = synthetic_spr(1500)
else:
    print("SPR_BENCH folder not found, using synthetic dataset.")
    spr_dict = synthetic_spr(1500)

# ---------------------------- vocabulary + encoding ------------------------------------
vocab = {"<PAD>": 0, "<UNK>": 1}
counter = Counter(tok for seq in spr_dict["train"]["sequence"] for tok in seq.split())
for t in counter:
    vocab[t] = len(vocab)
pad_idx, unk_idx = vocab["<PAD>"], vocab["<UNK>"]


def encode(seq):
    return [vocab.get(tok, unk_idx) for tok in seq.split()]


num_classes = len(set(spr_dict["train"]["label"]))


# ---------------------------- torch Dataset & Loader -----------------------------------
class SPRDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.seq, self.lab, self.ids = split["sequence"], split["label"], split["id"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s = self.seq[idx]
        tok_ids = encode(s)
        sym = torch.tensor(
            [len(set(t[0] for t in s.split())), len(set(t[1] for t in s.split()))],
            dtype=torch.float,
        )
        return {
            "seq": torch.tensor(tok_ids, dtype=torch.long),
            "len": len(tok_ids),
            "label": torch.tensor(self.lab[idx], dtype=torch.long),
            "sym": sym,
            "raw": s,
        }


def pad_collate(batch):
    lengths = [b["len"] for b in batch]
    mx = max(lengths)
    seqs = torch.full((len(batch), mx), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        seqs[i, : b["len"]] = b["seq"]
    labels = torch.stack([b["label"] for b in batch])
    syms = torch.stack([b["sym"] for b in batch])
    raws = [b["raw"] for b in batch]
    return {
        "seq": seqs,
        "len": torch.tensor(lengths),
        "label": labels,
        "sym": syms,
        "raw": raws,
    }


train_ds, val_ds, test_ds = (
    SPRDataset(spr_dict["train"]),
    SPRDataset(spr_dict["dev"]),
    SPRDataset(spr_dict["test"]),
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=pad_collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=pad_collate)


# ------------------------------- model -------------------------------------------------
class NeuroSymbolic(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, sym_dim, num_cls, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2 + sym_dim, num_cls)

    def forward(self, seq, lengths, sym):
        em = self.emb(seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        out = torch.cat([h, sym], dim=-1)
        return self.fc(out)


model = NeuroSymbolic(
    len(vocab), emb_dim=64, hid_dim=128, sym_dim=2, num_cls=num_classes, pad_idx=pad_idx
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# ------------------------------- helpers ----------------------------------------------
def batch_to_device(bt):
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in bt.items()
    }


def evaluate(loader):
    model.eval()
    total_loss, ys, ps, seqs = 0.0, [], [], []
    with torch.no_grad():
        for bt in loader:
            bt = batch_to_device(bt)
            logits = model(bt["seq"], bt["len"], bt["sym"])
            loss = criterion(logits, bt["label"])
            total_loss += loss.item() * bt["label"].size(0)
            pred = logits.argmax(1).cpu().tolist()
            ys.extend(bt["label"].cpu().tolist())
            ps.extend(pred)
            seqs.extend(bt["raw"])
    swa = shape_weighted_accuracy(seqs, ys, ps)
    return total_loss / len(ys), swa, ys, ps


# -------------------------------- training loop ---------------------------------------
EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for bt in train_loader:
        bt = batch_to_device(bt)
        optimizer.zero_grad()
        logits = model(bt["seq"], bt["len"], bt["sym"])
        loss = criterion(logits, bt["label"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * bt["label"].size(0)

    train_loss = running_loss / len(train_ds)
    val_loss, val_swa, _, _ = evaluate(val_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_swa"].append(val_swa)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"val_SWA={val_swa:.4f}"
    )

# ---------------------------------- final test ----------------------------------------
test_loss, test_swa, ys, ps = evaluate(test_loader)
experiment_data["SPR_BENCH"]["predictions"] = ps
experiment_data["SPR_BENCH"]["ground_truth"] = ys
print(f"Test Loss = {test_loss:.4f} | Test SWA = {test_swa:.4f}")

# ---------------------------------- save ----------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
