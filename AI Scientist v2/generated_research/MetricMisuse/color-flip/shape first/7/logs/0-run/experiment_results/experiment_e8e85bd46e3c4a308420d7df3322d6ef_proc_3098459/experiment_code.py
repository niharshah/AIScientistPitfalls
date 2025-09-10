import os, pathlib, numpy as np, torch, torch.nn as nn, random, math, time
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- reproducibility ----------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ---------- experiment container -----
experiment_data = {"batch_size": {}}

# ---------- working directory ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- locate SPR_BENCH ----------
def find_spr_bench() -> pathlib.Path:
    candidates = []
    env_path = os.environ.get("SPR_DATA_PATH")
    if env_path:
        candidates.append(env_path)
    candidates += [
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in candidates:
        if p and pathlib.Path(p).expanduser().joinpath("train.csv").exists():
            return pathlib.Path(p).expanduser().resolve()
    raise FileNotFoundError("Could not find SPR_BENCH dataset.")


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ---------- helpers -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(sequence: str):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str):
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0


# ---------- load dataset & vocab ------
spr = load_spr_bench(DATA_PATH)
all_tokens = set()
[all_tokens.update(ex["sequence"].split()) for ex in spr["train"]]
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1


def encode(seq: str):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, classes={num_classes}")


# ---------- Dataset wrappers ----------
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
    maxlen = max(len(item["input_ids"]) for item in batch)
    inp, lab, raw = [], [], []
    for item in batch:
        seq = item["input_ids"]
        pad = maxlen - len(seq)
        if pad:
            seq = torch.cat([seq, torch.full((pad,), PAD_ID, dtype=torch.long)])
        inp.append(seq)
        lab.append(item["label"])
        raw.append(item["raw_seq"])
    return {"input_ids": torch.stack(inp), "label": torch.stack(lab), "raw_seq": raw}


# ---------- model ---------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hidden=128, num_cls=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_cls)

    def forward(self, x):
        emb = self.embed(x)
        lengths = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(out)


# ---------- hyperparameter tuning -----
BATCH_SIZES = [32, 64, 128, 256]
EPOCHS = 6
for bs in BATCH_SIZES:
    print(f"\n======== Training with batch size {bs} ========")
    train_loader = DataLoader(
        SPRTorchSet(spr["train"]), batch_size=bs, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        SPRTorchSet(spr["dev"]), batch_size=512, shuffle=False, collate_fn=collate_fn
    )
    model = BiLSTMClassifier(vocab_size, num_cls=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for epoch in range(1, EPOCHS + 1):
        # training
        model.train()
        tot_loss = 0
        n = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            n += 1
        tr_loss = tot_loss / n
        exp_rec["losses"]["train"].append((epoch, tr_loss))

        # validation
        model.eval()
        val_loss = 0
        nb = 0
        all_preds, all_labels, all_seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                val_loss += loss.item()
                nb += 1
                preds = logits.argmax(-1).cpu().tolist()
                labs = batch["label"].cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labs)
                all_seqs.extend(batch["raw_seq"])
        val_loss /= nb
        exp_rec["losses"]["val"].append((epoch, val_loss))
        swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
        cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        exp_rec["metrics"]["val"].append((epoch, swa, cwa, hwa))
        exp_rec["predictions"], exp_rec["ground_truth"] = all_preds, all_labels
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )

    experiment_data["batch_size"][str(bs)] = exp_rec

# ---------- save results --------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
