import os, math, pathlib, gc, warnings
from collections import Counter
from datetime import datetime

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# ----------------- Device & misc -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
disable_caching()
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------- Working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ----------------- Resolve SPR_BENCH path -----------------
def resolve_spr_path() -> pathlib.Path:
    env_path = os.getenv("SPR_PATH")
    if env_path and (pathlib.Path(env_path) / "train.csv").exists():
        print(f"[Data] Using SPR_BENCH from SPR_PATH={env_path}")
        return pathlib.Path(env_path)
    cur = pathlib.Path.cwd()
    for parent in [cur] + list(cur.parents):
        cand = parent / "SPR_BENCH"
        if (cand / "train.csv").exists():
            print(f"[Data] Found SPR_BENCH at {cand}")
            return cand
    fallback = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fallback / "train.csv").exists():
        print(f"[Data] Using fallback SPR_BENCH at {fallback}")
        return fallback
    raise FileNotFoundError("SPR_BENCH dataset not found.")


# ----------------- SPR utilities -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=str(working_dir) + "/.cache_dsets",
        )

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


# ----------------- Hyper-params -----------------
EMB_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, LR = 64, 128, 128, 5, 1e-3
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"

# ----------------- Dataset build -----------------
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)
token_counter = Counter(tok for seq in spr["train"]["sequence"] for tok in seq.split())
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
vocab.update({tok: i + 2 for i, tok in enumerate(token_counter)})
label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
id2label = {i: l for l, i in label2id.items()}


def encode_sequence(seq):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

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
    lengths = [len(x["input_ids"]) for x in batch]
    max_len = max(lengths)
    inp = torch.full((len(batch), max_len), vocab[PAD_TOKEN], dtype=torch.long)
    for i, it in enumerate(batch):
        inp[i, : len(it["input_ids"])] = it["input_ids"]
    return {
        "input_ids": inp,
        "labels": torch.stack([b["labels"] for b in batch]),
        "seq_strs": [b["seq_str"] for b in batch],
        "lengths": torch.tensor(lengths),
    }


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# ----------------- Model -----------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1, self.relu = nn.Linear(emb_dim, HIDDEN_DIM), nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, out_dim)

    def forward(self, ids):
        mask = (ids != 0).float().unsqueeze(-1)
        avg = (self.emb(ids) * mask).sum(1) / mask.sum(1).clamp_(min=1e-6)
        return self.fc2(self.relu(self.fc1(avg)))


# ----------------- Evaluation -----------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot_loss = n_items = 0
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logit = model(batch["input_ids"])
            loss = criterion(logit, batch["labels"])
            bs = batch["labels"].size(0)
            tot_loss += loss.item() * bs
            n_items += bs
            p = logit.argmax(1).cpu().tolist()
            l = batch["labels"].cpu().tolist()
            preds.extend(p)
            labels.extend(l)
            seqs.extend(batch["seq_strs"])
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    bps = math.sqrt(max(swa, 0) * max(cwa, 0))
    return tot_loss / max(n_items, 1), swa, cwa, bps, preds, labels


# ----------------- Experiment container -----------------
experiment_data = {"optimizer_type": {}}

# ----------------- Optimizer sweep -----------------
optimizers_config = {
    "Adam": lambda params: torch.optim.Adam(params, lr=LR),
    "AdamW": lambda params: torch.optim.AdamW(params, lr=LR, weight_decay=1e-2),
    "SGD": lambda params: torch.optim.SGD(params, lr=LR, momentum=0.9),
}

for opt_name, opt_fn in optimizers_config.items():
    print(f"\n=== Training with {opt_name} ===")
    model = SPRClassifier(len(vocab), EMB_DIM, len(label2id)).to(device)
    optimizer = opt_fn(model.parameters())
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_swa": [],
        "val_cwa": [],
        "val_bps": [],
    }
    timestamps = []
    # ---- training loop ----
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
        val_loss, swa, cwa, bps, _, _ = evaluate(model, dev_loader)
        print(
            f"{opt_name} | Epoch {epoch}: train {train_loss:.4f} | "
            f"val {val_loss:.4f} | BPS {bps:.4f}"
        )
        for k, v in zip(metrics, [train_loss, val_loss, swa, cwa, bps]):
            metrics[k].append(v)
        timestamps.append(datetime.utcnow().isoformat())
    # ---- final eval ----
    dev_loss, dev_swa, dev_cwa, dev_bps, dev_preds, dev_labels = evaluate(
        model, dev_loader
    )
    test_loss, test_swa, test_cwa, test_bps, test_preds, test_labels = evaluate(
        model, test_loader
    )
    # ---- store ----
    experiment_data["optimizer_type"][opt_name] = {
        "metrics": metrics,
        "predictions": {"dev": dev_preds, "test": test_preds},
        "ground_truth": {"dev": dev_labels, "test": test_labels},
        "final_scores": {"dev_bps": dev_bps, "test_bps": test_bps},
        "timestamps": timestamps,
    }
    # free memory
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

# ----------------- Save all -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
