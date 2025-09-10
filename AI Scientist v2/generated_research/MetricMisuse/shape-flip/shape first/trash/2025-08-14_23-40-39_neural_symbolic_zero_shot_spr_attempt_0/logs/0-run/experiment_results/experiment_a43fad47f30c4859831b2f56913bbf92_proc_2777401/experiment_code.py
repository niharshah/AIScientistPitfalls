import os, math, pathlib, numpy as np, torch, random, gc
from collections import Counter
from datetime import datetime
from datasets import load_dataset, DatasetDict, disable_caching
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------- Reproducibility -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------- Device & folders -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device={device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
disable_caching()


# ----------------- Locate SPR_BENCH -----------------
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
    raise FileNotFoundError("Cannot locate SPR_BENCH dataset.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):
        return load_dataset(
            "csv",
            data_files=str(root / fname),
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
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ----------------- Hyper-parameters (static) -----------------
EMB_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, LR = 64, 128, 128, 5, 1e-3
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
WEIGHT_DECAYS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]

# ----------------- Load data and vocab -----------------
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)
token_counter = Counter(tok for seq in spr["train"]["sequence"] for tok in seq.split())
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
vocab.update({tok: i + 2 for i, tok in enumerate(token_counter)})
label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
NUM_CLASSES = len(label2id)
print(f"Vocab={len(vocab)}, Classes={NUM_CLASSES}")


def encode_sequence(seq):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.split()]


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
    lens = [len(b["input_ids"]) for b in batch]
    max_len = max(lens)
    input_ids = torch.full((len(batch), max_len), vocab[PAD_TOKEN], dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
    return {
        "input_ids": input_ids,
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


# ----------------- Model definition -----------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1, self.relu = nn.Linear(emb_dim, HIDDEN_DIM), nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, out_dim)

    def forward(self, input_ids):
        mask = (input_ids != 0).float().unsqueeze(-1)
        avg = ((self.emb(input_ids) * mask).sum(1)) / mask.sum(1).clamp(min=1e-6)
        return self.fc2(self.relu(self.fc1(avg)))


# ----------------- Evaluation helper -----------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot_loss = n_items = 0
    preds = labels = seqs = []
    all_preds, all_labels, all_seqs = [], [], []
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
            p = logits.argmax(1).cpu().tolist()
            l = batch["labels"].cpu().tolist()
            all_preds += p
            all_labels += l
            all_seqs += batch["seq_strs"]
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    bps = math.sqrt(swa * cwa) if swa >= 0 and cwa >= 0 else 0.0
    return tot_loss / max(n_items, 1), swa, cwa, bps, all_preds, all_labels


# ----------------- Experiment data container -----------------
experiment_data = {"weight_decay": {"SPR_BENCH": {}}}

# ----------------- Hyperparameter sweep -----------------
for wd in WEIGHT_DECAYS:
    print(f"\n===== Training with weight_decay={wd} =====")
    run_key = str(wd)
    experiment_data["weight_decay"]["SPR_BENCH"][run_key] = {
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
    # fresh model & optimizer
    model = SPRClassifier(len(vocab), EMB_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
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
        val_loss, swa, cwa, bps, *_ = evaluate(model, dev_loader)
        print(
            f"  Ep{epoch}: train {train_loss:.4f} | val {val_loss:.4f} | SWA {swa:.4f} | CWA {cwa:.4f} | BPS {bps:.4f}"
        )
        # log
        m = experiment_data["weight_decay"]["SPR_BENCH"][run_key]["metrics"]
        m["train_loss"].append(train_loss)
        m["val_loss"].append(val_loss)
        m["val_swa"].append(swa)
        m["val_cwa"].append(cwa)
        m["val_bps"].append(bps)
        experiment_data["weight_decay"]["SPR_BENCH"][run_key]["timestamps"].append(
            datetime.utcnow().isoformat()
        )
    # final eval
    dev_loss, dev_swa, dev_cwa, dev_bps, dev_preds, dev_labels = evaluate(
        model, dev_loader
    )
    test_loss, test_swa, test_cwa, test_bps, test_preds, test_labels = evaluate(
        model, test_loader
    )
    print(f"  >>> DEV BPS {dev_bps:.4f} | TEST BPS {test_bps:.4f}")
    exp = experiment_data["weight_decay"]["SPR_BENCH"][run_key]
    exp["predictions"]["dev"], exp["ground_truth"]["dev"] = dev_preds, dev_labels
    exp["predictions"]["test"], exp["ground_truth"]["test"] = test_preds, test_labels
    # free gpu for next run
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

# ----------------- Save everything -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
