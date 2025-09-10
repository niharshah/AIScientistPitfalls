import os, pathlib, random, time, json, math
import torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment container ----------
experiment_data = {
    "dropout_rate": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- locate SPR_BENCH ----------
def find_spr_bench_path() -> pathlib.Path:
    candidates = [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for c in candidates:
        if not c:
            continue
        p = pathlib.Path(c).expanduser().resolve()
        if (p / "train.csv").exists() and (p / "dev.csv").exists():
            print(f"Found SPR_BENCH at: {p}")
            return p
    raise FileNotFoundError(
        "SPR_BENCH directory with train.csv/dev.csv/test.csv not found."
    )


DATA_PATH = find_spr_bench_path()


# ---------- dataset ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(f"{s}.csv")
    return d


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- vocab / labels ----------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for ex in dataset:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_label_map(dataset):
    labels = sorted({ex["label"] for ex in dataset})
    return {lab: i for i, lab in enumerate(labels)}


vocab = build_vocab(spr["train"])
label2id = build_label_map(spr["train"])
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
pad_id = vocab["<pad>"]
print(f"Vocab size={len(vocab)}, labels={num_labels}")


# ---------- torch dataset ----------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vocab, label2id):
        self.data = hf_dataset
        self.vocab = vocab
        self.label2id = label2id

    def encode_seq(self, seq):
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        ids = self.encode_seq(ex["sequence"])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.label2id[ex["label"]], dtype=torch.long),
            "sequence": ex["sequence"],
        }


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    sequences = []
    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        input_ids[i, :l] = b["input_ids"]
        labels[i] = b["label"]
        sequences.append(b["sequence"])
    return {"input_ids": input_ids, "labels": labels, "sequences": sequences}


train_ds = SPRTorchDataset(spr["train"], vocab, label2id)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id)

train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0
)
dev_loader = DataLoader(
    dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0
)


# ---------- model ----------
class BiLSTMClassifier(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0, dropout_rate=0.0
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.pad_idx = pad_idx

    def forward(self, x):
        emb = self.emb(x)
        outputs, _ = self.lstm(emb)
        mask = (x != self.pad_idx).unsqueeze(-1)
        mean = (outputs * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.dropout(mean))


# ---------- hyperparameter search ----------
dropout_rates = [0.0, 0.25, 0.5]
epochs = 5
criterion = nn.CrossEntropyLoss()

for dr in dropout_rates:
    print(f"\n=== Training with dropout_rate={dr} ===")
    model = BiLSTMClassifier(
        len(vocab), 64, 128, num_labels, pad_idx=pad_id, dropout_rate=dr
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
        train_loss = total_loss / len(train_ds)
        experiment_data["dropout_rate"]["SPR_BENCH"]["losses"]["train"].append(
            (dr, epoch, train_loss)
        )
        experiment_data["dropout_rate"]["SPR_BENCH"]["metrics"]["train"].append(
            {"dropout": dr, "epoch": epoch, "loss": train_loss}
        )

        # ---- eval ----
        model.eval()
        val_loss, preds, trues, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch_t = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch_t["input_ids"])
                loss = criterion(logits, batch_t["labels"])
                val_loss += loss.item() * batch_t["labels"].size(0)
                ps = logits.argmax(-1).cpu().tolist()
                ts = batch_t["labels"].cpu().tolist()
                preds.extend(ps)
                trues.extend(ts)
                seqs.extend(batch["sequences"])
        val_loss /= len(dev_ds)
        swa = shape_weighted_accuracy(seqs, trues, preds)
        cwa = color_weighted_accuracy(seqs, trues, preds)
        hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0
        experiment_data["dropout_rate"]["SPR_BENCH"]["losses"]["val"].append(
            (dr, epoch, val_loss)
        )
        experiment_data["dropout_rate"]["SPR_BENCH"]["metrics"]["val"].append(
            {
                "dropout": dr,
                "epoch": epoch,
                "swa": swa,
                "cwa": cwa,
                "hwa": hwa,
                "loss": val_loss,
            }
        )
        if epoch == epochs:  # store predictions of final epoch
            experiment_data["dropout_rate"]["SPR_BENCH"]["predictions"].append(
                {"dropout": dr, "preds": preds}
            )
            experiment_data["dropout_rate"]["SPR_BENCH"]["ground_truth"].append(
                {"dropout": dr, "truth": trues}
            )
        print(
            f"d={dr} | Epoch {epoch}: train={train_loss:.4f} "
            f"val={val_loss:.4f} SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
