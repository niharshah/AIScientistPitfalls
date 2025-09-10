import os, pathlib, random, json, math, time
import torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment container ----------
experiment_data = {"batch_size": {}}  # will hold a sub-dict per tried batch size

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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
            print("Found SPR_BENCH at:", p)
            return p
    raise FileNotFoundError("Could not locate SPR_BENCH dataset.")


DATA_PATH = find_spr_bench_path()


# ---------- helpers ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    return DatasetDict({spl: _load(f"{spl}.csv") for spl in ["train", "dev", "test"]})


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
pad_id = vocab["<pad>"]
num_labels = len(label2id)
print("Vocab size:", len(vocab), "  Num labels:", num_labels)


# ---------- dataset class ----------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vocab, label2id):
        self.d = hf_dataset
        self.vocab = vocab
        self.label2id = label2id

    def encode(self, seq):
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in seq.split()]

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        ex = self.d[idx]
        return {
            "input_ids": torch.tensor(self.encode(ex["sequence"]), dtype=torch.long),
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


# ---------- model ----------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        mask = (x != pad_id).unsqueeze(-1)
        summed = (out * mask).sum(1)
        lens = mask.sum(1).clamp(min=1)
        mean = summed / lens
        return self.fc(mean)


# ---------- training routine ----------
def run_experiment(train_bs: int, epochs: int = 5):
    key = str(train_bs)
    experiment_data["batch_size"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    train_loader = DataLoader(
        SPRTorchDataset(spr["train"], vocab, label2id),
        batch_size=train_bs,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    dev_loader = DataLoader(
        SPRTorchDataset(spr["dev"], vocab, label2id),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = BiLSTMClassifier(len(vocab), 64, 128, num_labels, pad_idx=pad_id).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        run_loss = 0.0
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
            run_loss += loss.item() * batch["labels"].size(0)
        train_loss = run_loss / len(train_loader.dataset)

        # ---- eval ----
        model.eval()
        val_loss, preds, trues, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                t_batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(t_batch["input_ids"])
                loss = criterion(logits, t_batch["labels"])
                val_loss += loss.item() * t_batch["labels"].size(0)
                p = logits.argmax(-1).cpu().tolist()
                t = t_batch["labels"].cpu().tolist()
                preds.extend(p)
                trues.extend(t)
                seqs.extend(batch["sequences"])
        val_loss /= len(dev_loader.dataset)

        swa = shape_weighted_accuracy(seqs, trues, preds)
        cwa = color_weighted_accuracy(seqs, trues, preds)
        hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0

        # ---- log ----
        experiment_data["batch_size"][key]["losses"]["train"].append(train_loss)
        experiment_data["batch_size"][key]["losses"]["val"].append(val_loss)
        experiment_data["batch_size"][key]["metrics"]["train"].append(
            {"epoch": epoch, "loss": train_loss}
        )
        experiment_data["batch_size"][key]["metrics"]["val"].append(
            {"epoch": epoch, "swa": swa, "cwa": cwa, "hwa": hwa, "loss": val_loss}
        )
        experiment_data["batch_size"][key]["predictions"] = preds
        experiment_data["batch_size"][key]["ground_truth"] = trues

        print(
            f"[bs={train_bs}] Epoch {epoch}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )


# ---------- run hyperparam search ----------
for bs in [32, 64, 128, 256]:
    run_experiment(train_bs=bs, epochs=5)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to", os.path.join(working_dir, "experiment_data.npy"))
