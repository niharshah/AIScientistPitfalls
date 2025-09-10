import os, math, pathlib, numpy as np, torch, random
from collections import Counter
from datetime import datetime
from datasets import load_dataset, DatasetDict, disable_caching
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------- Repro / paths -----------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
disable_caching()


# ----------------- Dataset locating -----------------
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
    fallback = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fallback / "train.csv").exists():
        return fallback
    raise FileNotFoundError(
        "SPR_BENCH not found; set SPR_PATH or place dataset appropriately."
    )


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


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# ----------------- Hyper-params common -----------------
EMB_DIM = 64
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
HIDDEN_SWEEP = [64, 128, 256, 512]

# ----------------- Data prep -----------------
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
print(f"Vocab size {len(vocab)} | Classes {NUM_CLASSES}")


def encode_sequence(seq: str):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

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
    lens = [len(b["input_ids"]) for b in batch]
    max_len = max(lens)
    input_ids = torch.full((len(batch), max_len), vocab[PAD_TOKEN], dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
    labels = torch.stack([b["labels"] for b in batch])
    seqs = [b["seq_str"] for b in batch]
    return {"input_ids": input_ids, "labels": labels, "seq_strs": seqs}


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# ----------------- experiment data container -----------------
experiment_data = {
    "HIDDEN_DIM": {
        "SPR_BENCH": {
            "metrics": {
                "train_loss": [],
                "val_loss": [],
                "val_swa": [],
                "val_cwa": [],
                "val_bps": [],
            },
            "predictions": {"dev": [], "test": []},
            "ground_truth": {"dev": [], "test": []},
            "hparams": [],
            "timestamps": [],
        }
    }
}


# ----------------- Helper -----------------
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = 0
    n = 0
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
            tot_loss += loss.item() * bs
            n += bs
            p = logits.argmax(1).cpu().tolist()
            l = batch["labels"].cpu().tolist()
            preds.extend(p)
            labels.extend(l)
            seqs.extend(batch["seq_strs"])
    avg_loss = tot_loss / max(n, 1)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    bps = math.sqrt(swa * cwa) if swa >= 0 and cwa >= 0 else 0.0
    return avg_loss, swa, cwa, bps, preds, labels


# ----------------- Model class -----------------
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
        avg = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.fc2(self.relu(self.fc1(avg)))


# ----------------- Sweep -----------------
for hdim in HIDDEN_SWEEP:
    print(f"\n===== Training with HIDDEN_DIM={hdim} =====")
    model = SPRClassifier(len(vocab), EMB_DIM, hdim, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
        val_loss, swa, cwa, bps, _, _ = evaluate(model, dev_loader, criterion)[:6]
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"SWA={swa:.4f} | CWA={cwa:.4f} | BPS={bps:.4f}"
        )

    # Final evaluation
    dev_res = evaluate(model, dev_loader, criterion)
    test_res = evaluate(model, test_loader, criterion)

    # Record
    ed = experiment_data["HIDDEN_DIM"]["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(train_loss)
    ed["metrics"]["val_loss"].append(dev_res[0])
    ed["metrics"]["val_swa"].append(dev_res[1])
    ed["metrics"]["val_cwa"].append(dev_res[2])
    ed["metrics"]["val_bps"].append(dev_res[3])
    ed["predictions"]["dev"].append(dev_res[4])
    ed["ground_truth"]["dev"].append(dev_res[5])
    ed["predictions"]["test"].append(test_res[4])
    ed["ground_truth"]["test"].append(test_res[5])
    ed["hparams"].append({"HIDDEN_DIM": hdim})
    ed["timestamps"].append(datetime.utcnow().isoformat())

    torch.cuda.empty_cache()

# ----------------- Save -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy with all results.")
