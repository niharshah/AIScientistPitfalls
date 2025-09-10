import os, math, pathlib, random, numpy as np, torch
from collections import Counter
from datetime import datetime
from datasets import load_dataset, DatasetDict, disable_caching
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------- Repro/paths -----------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
disable_caching()


def resolve_spr_path() -> pathlib.Path:
    env = os.getenv("SPR_PATH")
    if env and (pathlib.Path(env) / "train.csv").exists():
        return pathlib.Path(env)
    cur = pathlib.Path.cwd()
    for p in [cur] + list(cur.parents):
        if (p / "SPR_BENCH" / "train.csv").exists():
            return p / "SPR_BENCH"
    fallback = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fallback / "train.csv").exists():
        return fallback
    raise FileNotFoundError("SPR_BENCH not found")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fn):
        return load_dataset(
            "csv",
            data_files=str(root / fn),
            split="train",
            cache_dir=str(working_dir) + "/.cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_var(seq):
    return len({t[0] for t in seq.strip().split() if t})


def count_color_var(seq):
    return len({t[1] for t in seq.strip().split() if len(t) > 1})


def swa_calc(seqs, y_true, y_pred):
    w = [count_shape_var(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def cwa_calc(seqs, y_true, y_pred):
    w = [count_color_var(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ----------------- Hyper-params (fixed) -----------------
EMB_DIM, HIDDEN_DIM, EPOCHS, LR = 64, 128, 5, 1e-3
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
BATCH_CANDIDATES = [32, 64, 128, 256]  # ← tuned

# ----------------- Data & vocab -----------------
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)
token_counter = Counter(tok for seq in spr["train"]["sequence"] for tok in seq.split())
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
[vocab.setdefault(t, len(vocab)) for t in token_counter]
inv_vocab = {i: t for t, i in vocab.items()}
label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(label2id)


def encode(seq):
    return [vocab.get(tok, 1) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return dict(
            input_ids=torch.tensor(encode(self.seqs[i]), dtype=torch.long),
            labels=torch.tensor(label2id[self.labels[i]], dtype=torch.long),
            seq_str=self.seqs[i],
        )


def collate(batch):
    lens = [len(x["input_ids"]) for x in batch]
    mx = max(lens)
    inp = torch.full((len(batch), mx), 0, dtype=torch.long)
    for i, x in enumerate(batch):
        inp[i, : len(x["input_ids"])] = x["input_ids"]
    return dict(
        input_ids=inp,
        labels=torch.stack([x["labels"] for x in batch]),
        seq_strs=[x["seq_str"] for x in batch],
        lengths=torch.tensor(lens),
    )


# ----------------- Model -----------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, EMB_DIM, padding_idx=0)
        self.fc1, self.relu = nn.Linear(EMB_DIM, HIDDEN_DIM), nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        avg = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.fc2(self.relu(self.fc1(avg)))


# ----------------- Eval -----------------
def evaluate(model, loader, criterion):
    model.eval()
    tloss, n = 0, 0
    preds = []
    labs = []
    seqs = []
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["input_ids"])
            loss = criterion(out, b["labels"])
            bs = b["labels"].size(0)
            tloss += loss.item() * bs
            n += bs
            pr = out.argmax(1).cpu().tolist()
            la = b["labels"].cpu().tolist()
            preds += pr
            labs += la
            seqs += b["seq_strs"]
    tloss /= max(n, 1)
    swa = swa_calc(seqs, labs, preds)
    cwa = cwa_calc(seqs, labs, preds)
    bps = math.sqrt(swa * cwa) if swa >= 0 and cwa >= 0 else 0
    return tloss, swa, cwa, bps, preds, labs


# ----------------- Experiment container -----------------
experiment_data = {
    "BATCH_SIZE": {
        "SPR_BENCH": {
            "batch_sizes": BATCH_CANDIDATES,
            "metrics": {
                "train_loss": [],
                "val_loss": [],
                "val_swa": [],
                "val_cwa": [],
                "val_bps": [],
            },
            "per_bs_metrics": [],
            "predictions": {"dev": {}, "test": {}},
            "ground_truth": {"dev": {}, "test": {}},
            "timestamps": [],
        }
    }
}

# ----------------- Hyperparameter sweep -----------------
for bs in BATCH_CANDIDATES:
    print(f"\n=== Training with batch size {bs} ===")
    train_loader = DataLoader(
        SPRDataset(spr["train"]), batch_size=bs, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        SPRDataset(spr["dev"]), batch_size=bs, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        SPRDataset(spr["test"]), batch_size=bs, shuffle=False, collate_fn=collate
    )

    model = SPRClassifier(len(vocab)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    bs_train_losses, bs_val_losses, bs_swa, bs_cwa, bs_bps = [], [], [], [], []

    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss, seen = 0, 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optim.zero_grad()
            out = model(batch["input_ids"])
            loss = criterion(out, batch["labels"])
            loss.backward()
            optim.step()
            run_loss += loss.item() * batch["labels"].size(0)
            seen += batch["labels"].size(0)
        tr_loss = run_loss / seen
        val_loss, swa, cwa, bps, _, _ = evaluate(model, dev_loader, criterion)
        print(
            f"Ep{ep} | train {tr_loss:.4f} | val {val_loss:.4f} | swa {swa:.3f} cwa {cwa:.3f} bps {bps:.3f}"
        )
        bs_train_losses.append(tr_loss)
        bs_val_losses.append(val_loss)
        bs_swa.append(swa)
        bs_cwa.append(cwa)
        bs_bps.append(bps)
        experiment_data["BATCH_SIZE"]["SPR_BENCH"]["timestamps"].append(
            datetime.utcnow().isoformat()
        )

    # store per-batchsize aggregates
    experiment_data["BATCH_SIZE"]["SPR_BENCH"]["metrics"]["train_loss"].append(
        bs_train_losses[-1]
    )
    experiment_data["BATCH_SIZE"]["SPR_BENCH"]["metrics"]["val_loss"].append(
        bs_val_losses[-1]
    )
    experiment_data["BATCH_SIZE"]["SPR_BENCH"]["metrics"]["val_swa"].append(bs_swa[-1])
    experiment_data["BATCH_SIZE"]["SPR_BENCH"]["metrics"]["val_cwa"].append(bs_cwa[-1])
    experiment_data["BATCH_SIZE"]["SPR_BENCH"]["metrics"]["val_bps"].append(bs_bps[-1])
    experiment_data["BATCH_SIZE"]["SPR_BENCH"]["per_bs_metrics"].append(
        dict(
            train_loss=bs_train_losses,
            val_loss=bs_val_losses,
            swa=bs_swa,
            cwa=bs_cwa,
            bps=bs_bps,
        )
    )

    # final dev/test predictions
    _, _, _, _, dev_preds, dev_labels = evaluate(model, dev_loader, criterion)
    _, _, _, _, test_preds, test_labels = evaluate(model, test_loader, criterion)
    experiment_data["BATCH_SIZE"]["SPR_BENCH"]["predictions"]["dev"][
        str(bs)
    ] = dev_preds
    experiment_data["BATCH_SIZE"]["SPR_BENCH"]["ground_truth"]["dev"][
        str(bs)
    ] = dev_labels
    experiment_data["BATCH_SIZE"]["SPR_BENCH"]["predictions"]["test"][
        str(bs)
    ] = test_preds
    experiment_data["BATCH_SIZE"]["SPR_BENCH"]["ground_truth"]["test"][
        str(bs)
    ] = test_labels

# ----------------- Save -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved results to", os.path.join(working_dir, "experiment_data.npy"))
