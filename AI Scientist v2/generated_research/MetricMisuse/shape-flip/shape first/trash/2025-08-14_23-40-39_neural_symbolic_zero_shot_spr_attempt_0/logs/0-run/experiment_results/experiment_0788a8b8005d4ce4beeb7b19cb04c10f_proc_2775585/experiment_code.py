import os, math, pathlib, random, gc
from collections import Counter
from datetime import datetime
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# ----------------- Global experiment container -----------------
experiment_data = {"epochs_tuning": {"SPR_BENCH": {}}}  # filled per epoch-count run

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- Disable HF global cache -----------------
disable_caching()


# ----------------- Data path resolver -----------------
def resolve_spr_path() -> pathlib.Path:
    env_path = os.getenv("SPR_PATH")
    if env_path:
        p = pathlib.Path(env_path).expanduser()
        if (p / "train.csv").exists():
            return p
    cur = pathlib.Path.cwd()
    for parent in [cur] + list(cur.parents):
        cand = parent / "SPR_BENCH"
        if (cand / "train.csv").exists():
            return cand
    fb = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fb / "train.csv").exists():
        return fb
    raise FileNotFoundError("SPR_BENCH not found.")


# ----------------- SPR utilities -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / (sum(w) or 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / (sum(w) or 1)


# ----------------- Hyperparameters -----------------
EMB_DIM, HIDDEN_DIM, BATCH_SIZE, LR = 64, 128, 128, 1e-3
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
EPOCH_CANDIDATES = [5, 10, 15, 20]

# ----------------- Dataset / vocab -----------------
spr_path = resolve_spr_path()
spr = load_spr_bench(spr_path)
train_sequences = spr["train"]["sequence"]
token_counter = Counter(tok for seq in train_sequences for tok in seq.split())
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for tok in token_counter:
    vocab.setdefault(tok, len(vocab))
label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
NUM_CLASSES = len(label2id)


def encode(seq):
    return [vocab.get(tok, 1) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(encode(self.seqs[i]), dtype=torch.long),
            "labels": torch.tensor(label2id[self.labels[i]], dtype=torch.long),
            "seq_str": self.seqs[i],
        }


def collate(batch):
    lens = [len(b["input_ids"]) for b in batch]
    mlen = max(lens)
    pad_val = vocab[PAD_TOKEN]
    inputs = torch.full((len(batch), mlen), pad_val, dtype=torch.long)
    for i, b in enumerate(batch):
        inputs[i, : len(b["input_ids"])] = b["input_ids"]
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "input_ids": inputs,
        "labels": labels,
        "seq_strs": [b["seq_str"] for b in batch],
    }


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
)


# ----------------- Model -----------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, EMB_DIM, padding_idx=0)
        self.fc1, self.relu = nn.Linear(EMB_DIM, HIDDEN_DIM), nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, ids):
        mask = (ids != 0).float().unsqueeze(-1)
        emb = self.emb(ids)
        avg = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.fc2(self.relu(self.fc1(avg)))


# ----------------- Evaluation -----------------
def evaluate(model, loader, criterion):
    model.eval()
    tot, n = 0.0, 0
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            bs = batch["labels"].size(0)
            tot += loss.item() * bs
            n += bs
            p = logits.argmax(1).cpu().tolist()
            l = batch["labels"].cpu().tolist()
            preds.extend(p)
            labels.extend(l)
            seqs.extend(batch["seq_strs"])
    val_loss = tot / (n or 1)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    bps = math.sqrt(max(swa, 0) * max(cwa, 0))
    return val_loss, swa, cwa, bps, preds, labels


# ----------------- Main tuning loop -----------------
for epochs in EPOCH_CANDIDATES:
    print(f"\n=== Training for {epochs} epochs ===")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = SPRClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_swa": [],
        "val_cwa": [],
        "val_bps": [],
    }
    timestamps = []

    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
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
            bs = batch["labels"].size(0)
            tot += loss.item() * bs
            n += bs
        train_loss = tot / n
        val_loss, swa, cwa, bps, *_ = evaluate(model, dev_loader, criterion)
        print(
            f"Ep {ep}: train {train_loss:.4f} | val {val_loss:.4f} | "
            f"SWA {swa:.4f} | CWA {cwa:.4f} | BPS {bps:.4f}"
        )
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_swa"].append(swa)
        metrics["val_cwa"].append(cwa)
        metrics["val_bps"].append(bps)
        timestamps.append(datetime.utcnow().isoformat())

    # final evaluations
    d_loss, d_swa, d_cwa, d_bps, d_preds, d_labels = evaluate(
        model, dev_loader, criterion
    )
    t_loss, t_swa, t_cwa, t_bps, t_preds, t_labels = evaluate(
        model, test_loader, criterion
    )
    print(f"DEV  | loss {d_loss:.4f} swa {d_swa:.4f} cwa {d_cwa:.4f} bps {d_bps:.4f}")
    print(f"TEST | loss {t_loss:.4f} swa {t_swa:.4f} cwa {t_cwa:.4f} bps {t_bps:.4f}")

    # store all info
    run_dict = {
        "metrics": metrics,
        "timestamps": timestamps,
        "final_dev": {
            "loss": d_loss,
            "swa": d_swa,
            "cwa": d_cwa,
            "bps": d_bps,
            "preds": d_preds,
            "labels": d_labels,
        },
        "final_test": {
            "loss": t_loss,
            "swa": t_swa,
            "cwa": t_cwa,
            "bps": t_bps,
            "preds": t_preds,
            "labels": t_labels,
        },
    }
    experiment_data["epochs_tuning"]["SPR_BENCH"][str(epochs)] = run_dict

    # cleanup
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

# ----------------- Save experiment data -----------------
np.save("experiment_data.npy", experiment_data)
