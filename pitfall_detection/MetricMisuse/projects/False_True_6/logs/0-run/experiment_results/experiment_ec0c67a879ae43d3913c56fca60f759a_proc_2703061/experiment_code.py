import os, pathlib, time, math, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_HYBRID": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
rec = experiment_data["SPR_HYBRID"]


# ---------- metric helpers ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


# ---------- load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- vocab ----------
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1}
    for tok in sorted(set(itertools.chain.from_iterable(s.split() for s in seqs))):
        vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


label_set = sorted(set(spr["train"]["label"]))
lab2idx = {l: i for i, l in enumerate(label_set)}
idx2lab = {i: l for l, i in lab2idx.items()}


# ---------- torch dataset ----------
class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [lab2idx[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i]
        return {
            "input_ids": torch.tensor(encode(seq), dtype=torch.long),
            "label": torch.tensor(self.labels[i]),
            "shape_cnt": torch.tensor(count_shape_variety(seq), dtype=torch.float),
            "color_cnt": torch.tensor(count_color_variety(seq), dtype=torch.float),
            "raw": seq,
        }


def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    shapes = torch.stack([b["shape_cnt"] for b in batch])
    colors = torch.stack([b["color_cnt"] for b in batch])
    pad_ids = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=vocab[PAD])
    raws = [b["raw"] for b in batch]
    return {
        "input_ids": pad_ids,
        "labels": labels,
        "shape_cnt": shapes,
        "color_cnt": colors,
        "raw": raws,
    }


train_loader = DataLoader(
    SPRDataset(spr["train"]), 128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(SPRDataset(spr["dev"]), 256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(
    SPRDataset(spr["test"]), 256, shuffle=False, collate_fn=collate
)


# ---------- hybrid model ----------
class HybridClassifier(nn.Module):
    def __init__(self, vocab_sz, emb=32, hid=64, n_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hid + 2, hid), nn.ReLU(), nn.Linear(hid, n_labels)
        )

    def forward(self, ids, sym_feats):
        x = self.embedding(ids)
        _, h = self.gru(x)
        h = h.squeeze(0)
        out = self.fc(torch.cat([h, sym_feats], dim=-1))
        return out


model = HybridClassifier(len(vocab), 32, 64, len(label_set)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- evaluation ----------
def evaluate(loader):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    all_seq, y_true, y_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labs = batch["labels"].to(device)
            feats = torch.stack([batch["shape_cnt"], batch["color_cnt"]], dim=1).to(
                device
            )
            logits = model(ids, feats)
            loss = criterion(logits, labs)
            loss_sum += loss.item() * len(labs)
            preds = logits.argmax(-1)
            correct += (preds == labs).sum().item()
            tot += len(labs)
            all_seq.extend(batch["raw"])
            y_true.extend(labs.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    acc = correct / tot
    swa = shape_weighted_accuracy(all_seq, y_true, y_pred)
    return loss_sum / tot, acc, swa, y_pred, y_true, all_seq


# ---------- training ----------
BEST, PATIENCE, no_imp = math.inf, 3, 0
for epoch in range(1, 21):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        labs = batch["labels"].to(device)
        feats = torch.stack([batch["shape_cnt"], batch["color_cnt"]], dim=1).to(device)
        optimizer.zero_grad()
        logits = model(ids, feats)
        loss = criterion(logits, labs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(labs)
    train_loss = epoch_loss / len(train_loader.dataset)
    val_loss, val_acc, val_swa, *_ = evaluate(dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  val_SWA = {val_swa:.3f}")
    rec["losses"]["train"].append(train_loss)
    rec["losses"]["val"].append(val_loss)
    rec["metrics"]["train"].append({"epoch": epoch})
    rec["metrics"]["val"].append({"epoch": epoch, "swa": val_swa, "acc": val_acc})
    rec["timestamps"].append(time.time())
    if val_loss < BEST - 1e-4:
        BEST, best_state = val_loss, model.state_dict()
        no_imp = 0
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print("Early stopping.")
            break
model.load_state_dict(best_state)

# ---------- test ----------
test_loss, test_acc, test_swa, preds, trues, seqs = evaluate(test_loader)
print(f"\nTEST  loss={test_loss:.4f}  acc={test_acc:.3f}  SWA={test_swa:.3f}")
rec["predictions"], rec["ground_truth"] = preds, trues
rec["metrics"]["test"] = {"loss": test_loss, "acc": test_acc, "swa": test_swa}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
