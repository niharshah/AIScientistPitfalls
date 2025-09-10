import os, pathlib, random, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict

# -------------------------------------------------------------------------
# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------------
# helper: try to locate the real SPR_BENCH folder or create a dummy set
def locate_or_build_spr() -> DatasetDict:
    """
    Return a DatasetDict with 'train','dev','test'.
    Search for SPR_BENCH in several locations; if not found, synthesize data.
    """
    candidate_roots = [
        pathlib.Path(os.getenv("SPR_DATA", "")),
        pathlib.Path(os.getcwd()) / "SPR_BENCH",
        pathlib.Path(os.getcwd()).parent / "SPR_BENCH",
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),  # earlier example
    ]
    candidate_roots = [p for p in candidate_roots if str(p) != ""]
    for root in candidate_roots:
        if (root / "train.csv").exists():
            print(f"Found SPR_BENCH at {root}")

            def _ld(csv_name: str):
                return load_dataset(
                    "csv",
                    data_files=str(root / csv_name),
                    split="train",
                    cache_dir=".cache_dsets",
                )

            return DatasetDict(
                train=_ld("train.csv"),
                dev=_ld("dev.csv"),
                test=_ld("test.csv"),
            )

    # -----------------------------------------------------------------
    # If we get here, no dataset was found – build a toy synthetic one.
    print("SPR_BENCH not found – generating synthetic toy dataset.")
    shapes = ["a", "b", "c", "d"]
    colors = ["1", "2", "3", "4"]

    def synth_example(label):
        # very naïve rule: label == (#unique_shapes + #unique_colors) % 3
        seq_len = random.randint(4, 10)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(seq_len)]
        seq = " ".join(toks)
        return {"sequence": seq, "label": label}

    def build_split(n_rows):
        data = []
        for _ in range(n_rows):
            seq_len = random.randint(4, 10)
            toks = [
                random.choice(shapes) + random.choice(colors) for _ in range(seq_len)
            ]
            seq = " ".join(toks)
            lbl = (len(set(t[0] for t in toks)) + len(set(t[1] for t in toks))) % 3
            data.append({"sequence": seq, "label": lbl})
        return Dataset.from_list(data)

    synth = DatasetDict(
        train=build_split(800),
        dev=build_split(200),
        test=build_split(200),
    )
    return synth


spr = locate_or_build_spr()
print({k: len(v) for k, v in spr.items()})


# -------------------------------------------------------------------------
# metric utilities
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def swa(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    return sum(w for w, yt, yp in zip(weights, y_true, y_pred) if yt == yp) / max(
        sum(weights), 1
    )


def cwa(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    return sum(w for w, yt, yp in zip(weights, y_true, y_pred) if yt == yp) / max(
        sum(weights), 1
    )


def cwca(seqs, y_true, y_pred):
    weights = [0.5 * (count_shape_variety(s) + count_color_variety(s)) for s in seqs]
    return sum(w for w, yt, yp in zip(weights, y_true, y_pred) if yt == yp) / max(
        sum(weights), 1
    )


# -------------------------------------------------------------------------
# build vocabulary
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


# -------------------------------------------------------------------------
# encode / collate
def encode(seq: str):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


def collate(batch):
    seqs = [encode(ex["sequence"]) for ex in batch]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    maxlen = max(lengths).item()
    padded = torch.zeros(len(seqs), maxlen, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    return {
        "seq": padded,
        "len": lengths,
        "label": labels,
        "raw_seq": [ex["sequence"] for ex in batch],
    }


# -------------------------------------------------------------------------
# model
class GRUClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim=64, hidden=64, n_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, seq, lengths):
        seq, lengths = seq.to(device), lengths.to(device)
        emb = self.embed(seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(h)


num_classes = len(set(spr["train"]["label"]))
model = GRUClassifier(vocab_size, n_classes=num_classes).to(device)

# -------------------------------------------------------------------------
# loaders
BATCH = 256
train_loader = DataLoader(
    spr["train"], batch_size=BATCH, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(spr["dev"], batch_size=BATCH, shuffle=False, collate_fn=collate)
test_loader = DataLoader(
    spr["test"], batch_size=BATCH, shuffle=False, collate_fn=collate
)

# -------------------------------------------------------------------------
# optimizer / loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------------------------------------------
# experiment data dict
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_cwca": [], "val_cwca": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------------------------------------------------------------
# evaluation
def evaluate(loader):
    model.eval()
    total_loss, y_true, y_pred, sequences = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch_t["seq"], batch_t["len"])
            loss = criterion(logits, batch_t["label"])
            total_loss += loss.item() * batch_t["seq"].size(0)
            preds = logits.argmax(dim=-1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch_t["label"].cpu().tolist())
            sequences.extend(batch_t["raw_seq"])
    avg_loss = total_loss / len(loader.dataset)
    cwca_val = cwca(sequences, y_true, y_pred)
    return avg_loss, cwca_val, y_true, y_pred, sequences


# -------------------------------------------------------------------------
# training loop
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss, y_tr, yhat_tr, seqs_tr = 0.0, [], [], []
    for batch in train_loader:
        optimizer.zero_grad()
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch_t["seq"], batch_t["len"])
        loss = criterion(logits, batch_t["label"])
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * batch_t["seq"].size(0)
        preds = logits.argmax(dim=-1).detach().cpu().tolist()
        yhat_tr.extend(preds)
        y_tr.extend(batch_t["label"].detach().cpu().tolist())
        seqs_tr.extend(batch_t["raw_seq"])
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_cwca = cwca(seqs_tr, y_tr, yhat_tr)

    val_loss, val_cwca, _, _, _ = evaluate(dev_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(avg_train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_cwca"].append(train_cwca)
    experiment_data["SPR_BENCH"]["metrics"]["val_cwca"].append(val_cwca)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
        f"train_CWCA={train_cwca:.4f}, val_CWCA={val_cwca:.4f}"
    )

# -------------------------------------------------------------------------
# final test evaluation
test_loss, test_cwca, y_true, y_pred, sequences = evaluate(test_loader)
swa_score = swa(sequences, y_true, y_pred)
cwa_score = cwa(sequences, y_true, y_pred)
print(
    f"Test loss={test_loss:.4f}, CWCA={test_cwca:.4f}, SWA={swa_score:.4f}, CWA={cwa_score:.4f}"
)

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
