import os, pathlib, numpy as np, torch, torch.nn as nn, random, math, time
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------- reproducibility -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------- working dir / device -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------- locate SPR_BENCH ---------------
def find_spr_bench() -> pathlib.Path:
    cand = []
    env_path = os.environ.get("SPR_DATA_PATH")
    if env_path:
        cand.append(env_path)
    cand += [
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cand:
        if p and pathlib.Path(p).expanduser().joinpath("train.csv").exists():
            return pathlib.Path(p).expanduser().resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ------------- helpers & metrics --------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.strip().split() if t))


def count_color_variety(seq):
    return len(set(t[1] for t in seq.strip().split() if len(t) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0


# ------------- load dataset -------------------
spr = load_spr_bench(DATA_PATH)

# ------------- vocab --------------------------
all_tokens = set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1


def encode(seq: str):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, num_classes={num_classes}")


# ------------- torch Dataset ------------------
class SPRTorchSet(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.enc = [encode(s) for s in self.seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.enc[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    inputs, labels, raw = [], [], []
    for b in batch:
        seq = b["input_ids"]
        if maxlen - len(seq):
            seq = torch.cat(
                [seq, torch.full((maxlen - len(seq),), PAD_ID, dtype=torch.long)]
            )
        inputs.append(seq)
        labels.append(b["label"])
        raw.append(b["raw_seq"])
    return {
        "input_ids": torch.stack(inputs),
        "label": torch.stack(labels),
        "raw_seq": raw,
    }


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# ------------- model with dropout -------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hidden=128, num_cls=2, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.emb_dp = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.out_dp = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_cls)

    def forward(self, x):
        emb = self.emb_dp(self.embed(x))
        lengths = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        out = self.out_dp(out)
        return self.fc(out)


# ------------- experiment container -----------
experiment_data = {"dropout_rate": {"SPR_BENCH": {}}}

# ------------- hyperparameter grid ------------
dropout_grid = [0.0, 0.2, 0.3, 0.5]
EPOCHS = 6

for p in dropout_grid:
    print(f"\n=== Training with dropout_rate={p} ===")
    model = BiLSTMClassifier(vocab_size, num_cls=num_classes, dropout=p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        n_batch = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        train_loss = total_loss / n_batch
        exp_rec["losses"]["train"].append((epoch, train_loss))

        # ---- validation ----
        model.eval()
        val_loss_tot = 0.0
        nb = 0
        all_preds, all_labels, all_seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                val_loss_tot += loss.item()
                nb += 1
                preds = logits.argmax(-1).cpu().tolist()
                labels = batch["label"].cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_seqs.extend(batch["raw_seq"])
        val_loss = val_loss_tot / nb
        exp_rec["losses"]["val"].append((epoch, val_loss))

        swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
        cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        exp_rec["metrics"]["val"].append((epoch, swa, cwa, hwa))
        exp_rec["predictions"] = all_preds
        exp_rec["ground_truth"] = all_labels

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )

    experiment_data["dropout_rate"]["SPR_BENCH"][f"p={p}"] = exp_rec

# ------------- save data ----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
