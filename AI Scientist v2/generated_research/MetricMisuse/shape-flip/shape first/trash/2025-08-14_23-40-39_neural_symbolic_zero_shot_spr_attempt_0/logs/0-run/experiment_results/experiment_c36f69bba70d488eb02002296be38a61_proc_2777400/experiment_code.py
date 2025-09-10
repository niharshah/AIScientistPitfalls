import os, math, pathlib, numpy as np, torch
from collections import Counter
from datetime import datetime
from datasets import load_dataset, DatasetDict, disable_caching
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------
# experiment data container (required format)
experiment_data = {"learning_rate": {}}  # top-level key is tuning type
# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
disable_caching()


# ----------------- Data helpers -----------------
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
    raise FileNotFoundError("SPR_BENCH dataset not found")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=str(working_dir) + "/.cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
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


# ----------------- Hyperparams (static) -----------------
EMB_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS = 64, 128, 128, 5
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
LR_GRID = [3e-4, 5e-4, 7e-4, 1.5e-3]

# ----------------- Dataset & vocab -----------------
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)
train_sequences = spr["train"]["sequence"]
token_counter = Counter(tok for seq in train_sequences for tok in seq.strip().split())
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for tok in token_counter:
    vocab[tok] = len(vocab)
inv_vocab = {i: t for t, i in vocab.items()}
label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(label2id)
print(f"Vocab size: {len(vocab)} | Classes: {NUM_CLASSES}")


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
    lens = [len(item["input_ids"]) for item in batch]
    max_len = max(lens)
    ids = torch.full((len(batch), max_len), vocab[PAD_TOKEN], dtype=torch.long)
    for i, item in enumerate(batch):
        ids[i, : lens[i]] = item["input_ids"]
    return {
        "input_ids": ids,
        "labels": torch.stack([b["labels"] for b in batch]),
        "seq_strs": [b["seq_str"] for b in batch],
        "lengths": torch.tensor(lens),
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
    def __init__(self, vocab_size, emb_dim, hidden_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, input_ids):
        mask = (input_ids != 0).float().unsqueeze(-1)
        emb = self.emb(input_ids)
        summed = (emb * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1e-6)
        avg = summed / lengths
        x = self.relu(self.fc1(avg))
        return self.fc2(x)


# ----------------- Evaluation -----------------
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, n = 0.0, 0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            bs = batch["labels"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            preds = logits.argmax(1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_seqs.extend(batch["seq_strs"])
    avg_loss = tot_loss / max(n, 1)
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    bps = math.sqrt(max(swa, 0) * max(cwa, 0))
    return avg_loss, swa, cwa, bps, all_preds, all_labels


# ----------------- Hyperparameter sweep -----------------
for LR in LR_GRID:
    lr_key = f"{LR:.0e}"
    print(f"\n===== Training with learning_rate={LR} =====")
    # storage init
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
    # fresh model
    model = SPRClassifier(len(vocab), EMB_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # training epochs
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            bs = batch["labels"].size(0)
            run_loss += loss.item() * bs
            seen += bs
        train_loss = run_loss / seen
        val_loss, swa, cwa, bps, *_ = evaluate(model, dev_loader, criterion)
        print(
            f"LR {LR:.0e} | Epoch {epoch}: "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"SWA {swa:.4f} | CWA {cwa:.4f} | BPS {bps:.4f}"
        )
        # log
        ed = experiment_data["learning_rate"][lr_key]["metrics"]
        ed["train_loss"].append(train_loss)
        ed["val_loss"].append(val_loss)
        ed["val_swa"].append(swa)
        ed["val_cwa"].append(cwa)
        ed["val_bps"].append(bps)
        experiment_data["learning_rate"][lr_key]["timestamps"].append(
            datetime.utcnow().isoformat()
        )

    # final dev/test evaluation
    dev_loss, dev_swa, dev_cwa, dev_bps, dev_preds, dev_labels = evaluate(
        model, dev_loader, criterion
    )
    test_loss, test_swa, test_cwa, test_bps, test_preds, test_labels = evaluate(
        model, test_loader, criterion
    )
    experiment_data["learning_rate"][lr_key]["predictions"]["dev"] = dev_preds
    experiment_data["learning_rate"][lr_key]["ground_truth"]["dev"] = dev_labels
    experiment_data["learning_rate"][lr_key]["predictions"]["test"] = test_preds
    experiment_data["learning_rate"][lr_key]["ground_truth"]["test"] = test_labels
    print(
        f"LR {LR:.0e} FINAL DEV loss {dev_loss:.4f} | SWA {dev_swa:.4f} | "
        f"CWA {dev_cwa:.4f} | BPS {dev_bps:.4f}"
    )
    print(
        f"LR {LR:.0e} FINAL TEST loss {test_loss:.4f} | SWA {test_swa:.4f} | "
        f"CWA {test_cwa:.4f} | BPS {test_bps:.4f}"
    )

# ----------------- Save everything -----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nSaved results to working/experiment_data.npy")
