import os, math, pathlib, gc
from collections import Counter
from datetime import datetime
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# ----------------- Experiment container -----------------
experiment_data = {"learning_rate": {}}  # each key will be the lr value e.g. "1e-3"

# ----------------- Device & misc -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
disable_caching()

# ----------------- Hyper-params except LR -----------------
EMB_DIM, HIDDEN_DIM = 64, 128
BATCH_SIZE, EPOCHS = 128, 5
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
LR_CANDIDATES = [1e-4, 3e-4, 5e-4, 1e-3, 2e-3]

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ----------------- Helpers for SPR -----------------
def resolve_spr_path() -> pathlib.Path:
    env_path = os.getenv("SPR_PATH")
    if env_path and (pathlib.Path(env_path) / "train.csv").exists():
        return pathlib.Path(env_path)
    cur = pathlib.Path.cwd()
    for p in [cur] + list(cur.parents):
        if (p / "SPR_BENCH" / "train.csv").exists():
            return p / "SPR_BENCH"
    fallback = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fallback / "train.csv").exists():
        return fallback
    raise FileNotFoundError("SPR_BENCH not found")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=str(working_dir) + "/.cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ----------------- Dataset prep (shared) -----------------
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)

token_counter = Counter(
    tok for seq in spr["train"]["sequence"] for tok in seq.strip().split()
)
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for tok in token_counter:
    vocab[tok] = len(vocab)
inv_vocab = {i: t for t, i in vocab.items()}

label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(label2id)
print("Vocab", len(vocab), "Classes", NUM_CLASSES)


def encode_sequence(seq: str):
    return [vocab.get(tok, 1) for tok in seq.split()]


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
    lens = [len(d["input_ids"]) for d in batch]
    max_len = max(lens)
    padded = torch.full((len(batch), max_len), 0, dtype=torch.long)
    for i, d in enumerate(batch):
        padded[i, : len(d["input_ids"])] = d["input_ids"]
    return {
        "input_ids": padded,
        "labels": torch.stack([d["labels"] for d in batch]),
        "seq_strs": [d["seq_str"] for d in batch],
        "lengths": torch.tensor(lens),
    }


train_set, dev_set, test_set = (
    SPRDataset(spr["train"]),
    SPRDataset(spr["dev"]),
    SPRDataset(spr["test"]),
)
train_loader = lambda bs: DataLoader(
    train_set, batch_size=bs, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    dev_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# ----------------- Model -----------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.fc1, self.relu = nn.Linear(emb_dim, HIDDEN_DIM), nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, out_dim)

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        emb = self.emb(x)
        summed = (emb * mask).sum(1)
        avg = summed / mask.sum(1).clamp(min=1e-6)
        return self.fc2(self.relu(self.fc1(avg)))


# ----------------- Evaluation -----------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot_loss = n = 0
    preds = labels = seqs = []
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            outs = model(batch["input_ids"])
            loss = criterion(outs, batch["labels"])
            bs = batch["labels"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            pr = outs.argmax(1).cpu().tolist()
            la = batch["labels"].cpu().tolist()
            preds += pr
            labels += la
            seqs += batch["seq_strs"]
    loss = tot_loss / max(n, 1)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    bps = math.sqrt(max(swa, 0) * max(cwa, 0))
    return loss, swa, cwa, bps, preds, labels


# ----------------- Training over LR sweep -----------------
for LR in LR_CANDIDATES:
    lr_key = f"{LR:.0e}"
    print(f"\n=== Training with LR={LR} ===")
    experiment_data["learning_rate"][lr_key] = {
        "metrics": {
            "train_loss": [],
            "val_loss": [],
            "val_swa": [],
            "val_cwa": [],
            "val_bps": [],
        },
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
        "timestamps": [],
    }

    model = SPRClassifier(len(vocab), EMB_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = seen = 0
        for batch in train_loader(BATCH_SIZE):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optimizer.zero_grad()
            outs = model(batch["input_ids"])
            loss = criterion(outs, batch["labels"])
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * batch["labels"].size(0)
            seen += batch["labels"].size(0)
        scheduler.step()
        train_loss = run_loss / seen
        val_loss, swa, cwa, bps, *_ = evaluate(model, dev_loader)
        print(
            f"  Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | BPS {bps:.4f}"
        )

        md = experiment_data["learning_rate"][lr_key]["metrics"]
        md["train_loss"].append(train_loss)
        md["val_loss"].append(val_loss)
        md["val_swa"].append(swa)
        md["val_cwa"].append(cwa)
        md["val_bps"].append(bps)
        experiment_data["learning_rate"][lr_key]["timestamps"].append(
            datetime.utcnow().isoformat()
        )

    # final evaluation
    dev_loss, dev_swa, dev_cwa, dev_bps, dev_preds, dev_labels = evaluate(
        model, dev_loader
    )
    test_loss, test_swa, test_cwa, test_bps, test_preds, test_labels = evaluate(
        model, test_loader
    )
    print(f"DEV  BPS {dev_bps:.4f} | TEST BPS {test_bps:.4f}")

    exp = experiment_data["learning_rate"][lr_key]
    exp["predictions"]["dev"] = dev_preds
    exp["ground_truth"]["dev"] = dev_labels
    exp["predictions"]["test"] = test_preds
    exp["ground_truth"]["test"] = test_labels
    # cleanup
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

# ----------------- Save everything -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
