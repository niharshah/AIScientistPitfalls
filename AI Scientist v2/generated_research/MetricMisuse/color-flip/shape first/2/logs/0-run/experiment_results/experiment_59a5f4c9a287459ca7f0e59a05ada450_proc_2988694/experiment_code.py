# hyperparam_tuning_learning_rate.py
import os, pathlib, random, time, json, math, sys
import torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ---------- reproducibility ----------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment container ----------
experiment_data = {"learning_rate": {}}


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


# ---------- dataset utilities ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ---------- load dataset ----------
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- vocab / label maps ----------
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
print(f"Vocab size = {len(vocab)}, num_labels = {num_labels}")


# ---------- Torch dataset ----------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vocab, label2id):
        self.data, self.vocab, self.label2id = hf_dataset, vocab, label2id

    def __len__(self):
        return len(self.data)

    def encode_seq(self, seq):
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "input_ids": torch.tensor(
                self.encode_seq(ex["sequence"]), dtype=torch.long
            ),
            "label": torch.tensor(self.label2id[ex["label"]], dtype=torch.long),
            "sequence": ex["sequence"],
        }


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    seqs = []
    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        input_ids[i, :l] = b["input_ids"]
        labels[i] = b["label"]
        seqs.append(b["sequence"])
    return {"input_ids": input_ids, "labels": labels, "sequences": seqs}


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
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        emb = self.emb(x)
        outputs, _ = self.lstm(emb)
        mask = (x != pad_id).unsqueeze(-1)
        mean = (outputs * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(mean)


# ---------- hyperparameter sweep ----------
lrs = [3e-4, 5e-4, 1e-3, 2e-3]
epochs = 5

for lr in lrs:
    tag = f"lr_{lr:.0e}" if lr < 1e-3 else f"lr_{lr}"
    print(f"\n=== Training with learning_rate = {lr} ===")
    experiment_data["learning_rate"][tag] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = BiLSTMClassifier(len(vocab), 64, 128, num_labels, pad_idx=pad_id).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        # ---- eval ----
        model.eval()
        val_loss, all_pred, all_true, all_seq = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                tb = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(tb["input_ids"])
                loss = criterion(logits, tb["labels"])
                val_loss += loss.item() * tb["labels"].size(0)
                preds = logits.argmax(-1).cpu().tolist()
                truths = tb["labels"].cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(truths)
                all_seq.extend(batch["sequences"])
        val_loss /= len(dev_ds)

        swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
        cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
        hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0

        # logging
        exp = experiment_data["learning_rate"][tag]
        exp["losses"]["train"].append(train_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["train"].append({"epoch": epoch, "loss": train_loss})
        exp["metrics"]["val"].append(
            {"epoch": epoch, "swa": swa, "cwa": cwa, "hwa": hwa, "loss": val_loss}
        )
        if epoch == epochs:  # save final preds
            exp["predictions"] = all_pred
            exp["ground_truth"] = all_true

        print(
            f"[{tag}] Epoch {epoch}: train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | SWA={swa:.4f} "
            f"CWA={cwa:.4f} HWA={hwa:.4f}"
        )

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
