import os, math, pathlib, gc
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# ---------- experiment data container ----------
experiment_data = {
    "batch_size_sweep": {
        "SPR_BENCH": {
            "metrics": {
                "train_loss": {},  # will be dict[bs] -> list
                "val_loss": {},
                "val_swa": {},
                "val_cwa": {},
                "val_bps": {},
            },
            "predictions": {"dev": {}, "test": {}},
            "ground_truth": {"dev": {}, "test": {}},
            "timestamps": {},
        }
    }
}

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
disable_caching()


# ----------------- Data-set path resolver -----------------
def resolve_spr_path() -> pathlib.Path:
    env_path = os.getenv("SPR_PATH")
    if env_path:
        p = pathlib.Path(env_path).expanduser()
        if (p / "train.csv").exists():
            print(f"[Data] Using SPR_BENCH from SPR_PATH={p}")
            return p
    cur = pathlib.Path.cwd()
    for parent in [cur] + list(cur.parents):
        candidate = parent / "SPR_BENCH"
        if (candidate / "train.csv").exists():
            print(f"[Data] Found SPR_BENCH at {candidate}")
            return candidate
    fallback = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fallback / "train.csv").exists():
        print(f"[Data] Using fallback SPR_BENCH at {fallback}")
        return fallback
    raise FileNotFoundError("Cannot locate SPR_BENCH dataset")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ----------------- Hyper-parameters (static) -----------------
EMB_DIM, HIDDEN_DIM, EPOCHS, LR = 64, 128, 5, 1e-3
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
BATCH_SIZES = [32, 64, 128, 256, 512]

# ----------------- Load data & vocab (once) -----------------
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)
train_sequences = spr["train"]["sequence"]
token_counter = Counter(tok for seq in train_sequences for tok in seq.strip().split())
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for tok in token_counter:
    vocab[tok] = len(vocab)
label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(label2id)
print(f"Vocab size {len(vocab)} | Classes {NUM_CLASSES}")


def encode_sequence(seq: str):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.strip().split()]


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode_sequence(self.seqs[idx]), dtype=torch.long
            ),
            "labels": torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
            "seq_str": self.seqs[idx],
        }


def collate_fn(batch):
    lens = [len(i["input_ids"]) for i in batch]
    max_len = max(lens)
    ids = torch.full((len(batch), max_len), vocab[PAD_TOKEN], dtype=torch.long)
    for i, item in enumerate(batch):
        ids[i, : len(item["input_ids"])] = item["input_ids"]
    labels = torch.stack([b["labels"] for b in batch])
    seqs = [b["seq_str"] for b in batch]
    return {"input_ids": ids, "labels": labels, "seq_strs": seqs}


train_ds, dev_ds, test_ds = (
    SPRDataset(spr["train"]),
    SPRDataset(spr["dev"]),
    SPRDataset(spr["test"]),
)


class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1 = nn.Linear(emb_dim, HIDDEN_DIM)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, out_dim)

    def forward(self, input_ids):
        mask = (input_ids != 0).float().unsqueeze(-1)
        emb = self.emb(input_ids)
        summed = (emb * mask).sum(1)
        avg = summed / mask.sum(1).clamp(min=1e-6)
        return self.fc2(self.act(self.fc1(avg)))


def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = n_items = 0
    preds_all, labels_all, seqs_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            bs = batch["labels"].size(0)
            tot_loss += loss.item() * bs
            n_items += bs
            preds = logits.argmax(1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            preds_all.extend(preds)
            labels_all.extend(labels)
            seqs_all.extend(batch["seq_strs"])
    avg_loss = tot_loss / max(n_items, 1)
    swa = shape_weighted_accuracy(seqs_all, labels_all, preds_all)
    cwa = color_weighted_accuracy(seqs_all, labels_all, preds_all)
    bps = math.sqrt(swa * cwa)
    return avg_loss, swa, cwa, bps, preds_all, labels_all


criterion = nn.CrossEntropyLoss()

for bs in BATCH_SIZES:
    print(f"\n===== Training with batch size {bs} =====")
    # data loaders
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn
    )

    # model & optimizer
    model = SPRClassifier(len(vocab), EMB_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # prepare experiment_data dict slots
    for k in experiment_data["batch_size_sweep"]["SPR_BENCH"]["metrics"]:
        experiment_data["batch_size_sweep"]["SPR_BENCH"]["metrics"][k][bs] = []
    experiment_data["batch_size_sweep"]["SPR_BENCH"]["timestamps"][bs] = []

    # training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = seen = 0
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
        val_loss, swa, cwa, bps, *_ = evaluate(model, dev_loader, criterion)
        print(
            f"Epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f} | SWA {swa:.4f} | CWA {cwa:.4f} | BPS {bps:.4f}"
        )
        # log
        experiment_data["batch_size_sweep"]["SPR_BENCH"]["metrics"]["train_loss"][
            bs
        ].append(train_loss)
        experiment_data["batch_size_sweep"]["SPR_BENCH"]["metrics"]["val_loss"][
            bs
        ].append(val_loss)
        experiment_data["batch_size_sweep"]["SPR_BENCH"]["metrics"]["val_swa"][
            bs
        ].append(swa)
        experiment_data["batch_size_sweep"]["SPR_BENCH"]["metrics"]["val_cwa"][
            bs
        ].append(cwa)
        experiment_data["batch_size_sweep"]["SPR_BENCH"]["metrics"]["val_bps"][
            bs
        ].append(bps)
        experiment_data["batch_size_sweep"]["SPR_BENCH"]["timestamps"][bs].append(
            datetime.utcnow().isoformat()
        )

    # final dev/test evaluation
    dev_loss, dev_swa, dev_cwa, dev_bps, dev_preds, dev_labels = evaluate(
        model, dev_loader, criterion
    )
    test_loss, test_swa, test_cwa, test_bps, test_preds, test_labels = evaluate(
        model, test_loader, criterion
    )
    print(f"==> BS {bs} DEV BPS {dev_bps:.4f} | TEST BPS {test_bps:.4f}")
    experiment_data["batch_size_sweep"]["SPR_BENCH"]["predictions"]["dev"][
        bs
    ] = dev_preds
    experiment_data["batch_size_sweep"]["SPR_BENCH"]["ground_truth"]["dev"][
        bs
    ] = dev_labels
    experiment_data["batch_size_sweep"]["SPR_BENCH"]["predictions"]["test"][
        bs
    ] = test_preds
    experiment_data["batch_size_sweep"]["SPR_BENCH"]["ground_truth"]["test"][
        bs
    ] = test_labels

    # free memory
    del model, optimizer, train_loader, dev_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

# save all data
np.save("experiment_data.npy", experiment_data)
print("Saved experiment_data.npy")
