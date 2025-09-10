import os, math, pathlib, random, time
from collections import Counter
from datetime import datetime

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# ----------------- Global experiment container -----------------
experiment_data = {"EMB_DIM": {"SPR_BENCH": {}}}

# ----------------- Device & seeds -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ----------------- Disable HF cache -----------------
disable_caching()


# ----------------- Data-set path resolver -----------------
def resolve_spr_path() -> pathlib.Path:
    env_path = os.getenv("SPR_PATH")
    if env_path and (pathlib.Path(env_path) / "train.csv").exists():
        return pathlib.Path(env_path)
    cur = pathlib.Path.cwd()
    for p in [cur] + list(cur.parents):
        if (p / "SPR_BENCH" / "train.csv").exists():
            return p / "SPR_BENCH"
    default = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (default / "train.csv").exists():
        return default
    raise FileNotFoundError("SPR_BENCH not found.")


# ----------------- SPR helpers -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # type: ignore
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=str(pathlib.Path.cwd() / "working" / ".cache_dsets"),
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def variety(seq, idx):  # idx 0 for shape, 1 for color
    return len({tok[idx] for tok in seq.strip().split() if len(tok) > idx})


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [variety(s, 0) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [variety(s, 1) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


# ----------------- Hyperparameters (except EMB_DIM) -----------------
HIDDEN_DIM, BATCH_SIZE, EPOCHS, LR = 128, 128, 5, 1e-3
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
EMB_DIM_LIST = [32, 64, 128, 256]

# ----------------- Prepare dataset & dataloaders (shared) -----------------
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)

train_sequences = spr["train"]["sequence"]
token_counter = Counter(tok for seq in train_sequences for tok in seq.strip().split())
vocab = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1,
    **{tok: i + 2 for i, tok in enumerate(token_counter)},
}
inv_vocab = {i: t for t, i in vocab.items()}
label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(label2id)


def encode_sequence(seq: str):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.strip().split()]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return dict(
            input_ids=torch.tensor(encode_sequence(self.seqs[idx]), dtype=torch.long),
            labels=torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
            seq_str=self.seqs[idx],
        )


def collate_fn(batch):
    lengths = [len(b["input_ids"]) for b in batch]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), vocab[PAD_TOKEN], dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
    labels = torch.stack([b["labels"] for b in batch])
    seqs = [b["seq_str"] for b in batch]
    return dict(
        input_ids=input_ids, labels=labels, seq_strs=seqs, lengths=torch.tensor(lengths)
    )


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# ----------------- Model definition -----------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(emb_dim, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, out_dim)
        )

    def forward(self, ids):
        mask = (ids != 0).float().unsqueeze(-1)
        emb = self.emb(ids)
        avg = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.net(avg)


# ----------------- Evaluation helper -----------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot, n = 0, 0
    preds = []
    labels = []
    seqs = []
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
            preds += p
            labels += l
            seqs += batch["seq_strs"]
    loss = tot / max(n, 1)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    bps = math.sqrt(swa * cwa)
    return loss, swa, cwa, bps, preds, labels


# ----------------- Training loop for each embedding size -----------------
for EMB_DIM in EMB_DIM_LIST:
    print(f"\n=== Training with EMB_DIM={EMB_DIM} ===")
    model = SPRClassifier(len(vocab), EMB_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_swa": [],
        "val_cwa": [],
        "val_bps": [],
    }
    timestamps = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0
        seen = 0
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
            seen += batch["labels"].size(0)
        train_loss = run_loss / seen
        val_loss, swa, cwa, bps, _, _ = evaluate(model, dev_loader)
        print(
            f"Epoch {epoch}: tr {train_loss:.4f} | val {val_loss:.4f} | swa {swa:.3f} | cwa {cwa:.3f} | bps {bps:.3f}"
        )
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_swa"].append(swa)
        metrics["val_cwa"].append(cwa)
        metrics["val_bps"].append(bps)
        timestamps.append(datetime.utcnow().isoformat())
    dev_loss, dev_swa, dev_cwa, dev_bps, dev_preds, dev_labels = evaluate(
        model, dev_loader
    )
    test_loss, test_swa, test_cwa, test_bps, test_preds, test_labels = evaluate(
        model, test_loader
    )
    print(
        f"Final DEV  loss {dev_loss:.4f} swa {dev_swa:.3f} cwa {dev_cwa:.3f} bps {dev_bps:.3f}"
    )
    print(
        f"Final TEST loss {test_loss:.4f} swa {test_swa:.3f} cwa {test_cwa:.3f} bps {test_bps:.3f}"
    )

    # store results -----------------------------------
    experiment_data["EMB_DIM"]["SPR_BENCH"][EMB_DIM] = dict(
        metrics=metrics,
        predictions={"dev": dev_preds, "test": test_preds},
        ground_truth={"dev": dev_labels, "test": test_labels},
        final_scores={
            "dev": {"loss": dev_loss, "swa": dev_swa, "cwa": dev_cwa, "bps": dev_bps},
            "test": {
                "loss": test_loss,
                "swa": test_swa,
                "cwa": test_cwa,
                "bps": test_bps,
            },
        },
        timestamps=timestamps,
    )
    torch.cuda.empty_cache()

# ----------------- Save -----------------
np.save("experiment_data.npy", experiment_data)
