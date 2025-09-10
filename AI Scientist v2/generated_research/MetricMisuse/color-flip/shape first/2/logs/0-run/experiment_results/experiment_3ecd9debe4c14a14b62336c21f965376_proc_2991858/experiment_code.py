# -------------------- hyper-param tuning :  DROPOUT --------------------
import os, pathlib, random, time, json, math
import torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ---------- reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- save container ----------
experiment_data = {}


# ---------- locate SPR_BENCH ----------
def find_spr_bench_path() -> pathlib.Path:
    for p in [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]:
        if not p:
            continue
        p = pathlib.Path(p).expanduser().resolve()
        if (p / "train.csv").exists() and (p / "dev.csv").exists():
            return p
    raise FileNotFoundError("SPR_BENCH train/dev/test csv not found")


DATA_PATH = find_spr_bench_path()


# ---------- dataset ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- utils ----------
def count_shape_variety(sequence):  # how many unique first char tokens
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence):  # how many unique second char tokens
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


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
pad_id = vocab["<pad>"]
num_labels = len(label2id)
print("Vocab:", len(vocab), "Labels:", num_labels)


# ---------- torch dataset ----------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vocab, label2id):
        self.data, self.vocab, self.label2id = hf_dataset, vocab, label2id

    def __len__(self):
        return len(self.data)

    def encode(self, seq):
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in seq.split()]

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "input_ids": torch.tensor(self.encode(ex["sequence"]), dtype=torch.long),
            "label": torch.tensor(self.label2id[ex["label"]], dtype=torch.long),
            "sequence": ex["sequence"],
        }


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels, sequences = [], []
    for i, b in enumerate(batch):
        seq_len = len(b["input_ids"])
        input_ids[i, :seq_len] = b["input_ids"]
        labels.append(b["label"])
        sequences.append(b["sequence"])
    return {
        "input_ids": input_ids,
        "labels": torch.stack(labels),
        "sequences": sequences,
    }


train_ds, dev_ds = SPRTorchDataset(spr["train"], vocab, label2id), SPRTorchDataset(
    spr["dev"], vocab, label2id
)
train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0
)
dev_loader = DataLoader(
    dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0
)


# ---------- model ----------
class BiLSTMClassifier(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0, dropout=0.1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        mask = (x != pad_id).unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mean = self.dropout(mean)
        return self.fc(mean)


# ---------- training routine ----------
def run_experiment(drop_rate):
    tag = f"dropout_{drop_rate}"
    experiment_data[tag] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    model = BiLSTMClassifier(
        len(vocab), 64, 128, num_labels, pad_idx=pad_id, dropout=drop_rate
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            loss = criterion(model(batch["input_ids"]), batch["labels"])
            loss.backward()
            optimizer.step()
            running += loss.item() * batch["labels"].size(0)
        train_loss = running / len(train_ds)
        experiment_data[tag]["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data[tag]["SPR_BENCH"]["metrics"]["train"].append(
            {"epoch": epoch, "loss": train_loss}
        )
        # eval
        model.eval()
        vloss = 0.0
        preds = []
        truths = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                tensor_batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(tensor_batch["input_ids"])
                vloss += criterion(
                    logits, tensor_batch["labels"]
                ).item() * tensor_batch["labels"].size(0)
                preds.extend(logits.argmax(-1).cpu().tolist())
                truths.extend(tensor_batch["labels"].cpu().tolist())
                seqs.extend(batch["sequences"])
        vloss /= len(dev_ds)
        swa = shape_weighted_accuracy(seqs, truths, preds)
        cwa = color_weighted_accuracy(seqs, truths, preds)
        hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
        experiment_data[tag]["SPR_BENCH"]["losses"]["val"].append(vloss)
        experiment_data[tag]["SPR_BENCH"]["metrics"]["val"].append(
            {"epoch": epoch, "swa": swa, "cwa": cwa, "hwa": hwa, "loss": vloss}
        )
        experiment_data[tag]["SPR_BENCH"]["predictions"] = preds
        experiment_data[tag]["SPR_BENCH"]["ground_truth"] = truths
        print(
            f"[{tag}] Epoch {epoch}: train {train_loss:.3f} | val {vloss:.3f} | HWA {hwa:.3f}"
        )
    return


# ---------- run for multiple dropout settings ----------
for drop in [0.0, 0.1, 0.3, 0.5]:
    run_experiment(drop)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved to", os.path.join(working_dir, "experiment_data.npy"))
