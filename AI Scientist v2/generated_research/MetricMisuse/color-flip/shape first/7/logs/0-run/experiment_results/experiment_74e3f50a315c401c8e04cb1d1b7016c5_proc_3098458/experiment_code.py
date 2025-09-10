import os, pathlib, numpy as np, torch, torch.nn as nn, random, math, time
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# --------- reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

# ---------- working directory -------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- locate SPR_BENCH --------
def find_spr_bench() -> pathlib.Path:
    env = os.environ.get("SPR_DATA_PATH")
    cand = [env] if env else []
    cand += [
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cand:
        if p and pathlib.Path(p).expanduser().joinpath("train.csv").exists():
            return pathlib.Path(p).expanduser().resolve()
    raise FileNotFoundError("SPR_BENCH not found.")


DATA_PATH = find_spr_bench()
print("Found SPR_BENCH at", DATA_PATH)


# ---------- helpers & metrics -------
def load_spr_bench(root):
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


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


# ---------- dataset -----------------
spr = load_spr_bench(DATA_PATH)
all_tokens = set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1


def encode(seq):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))


class SPRTorchSet(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.encoded = [encode(s) for s in self.seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encoded[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    maxlen = max(len(it["input_ids"]) for it in batch)
    inp, lab, raw = [], [], []
    for it in batch:
        seq = it["input_ids"]
        if maxlen - len(seq):
            seq = torch.cat(
                [seq, torch.full((maxlen - len(seq),), PAD_ID, dtype=torch.long)]
            )
        inp.append(seq)
        lab.append(it["label"])
        raw.append(it["raw_seq"])
    return {"input_ids": torch.stack(inp), "label": torch.stack(lab), "raw_seq": raw}


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# ---------- model -------------------
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
        out = torch.cat([h_n[-2], h_n[-1]], 1)
        return self.fc(out)


# --------- experiment container -----
experiment_data = {"learning_rate": {"SPR_BENCH": {}}}


# --------- training routine ---------
def train_and_eval(lr, epochs=6):
    model = BiLSTMClassifier(vocab_size, num_cls=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for ep in range(1, epochs + 1):
        # train
        model.train()
        tot_loss = 0
        nb = 0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            nb += 1
        tr_loss = tot_loss / nb
        rec["losses"]["train"].append((ep, tr_loss))
        # val
        model.eval()
        tot, nb = 0, 0
        preds, labels, seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                tot += loss.item()
                nb += 1
                p = logits.argmax(-1).cpu().tolist()
                l = batch["label"].cpu().tolist()
                preds.extend(p)
                labels.extend(l)
                seqs.extend(batch["raw_seq"])
        v_loss = tot / nb
        rec["losses"]["val"].append((ep, v_loss))
        swa = shape_weighted_accuracy(seqs, labels, preds)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        rec["metrics"]["val"].append((ep, swa, cwa, hwa))
        rec["predictions"], rec["ground_truth"] = preds, labels
        print(
            f"[lr={lr}] Epoch {ep}: train_loss={tr_loss:.4f} val_loss={v_loss:.4f} "
            f"SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )
    return rec


# --------- hyperparameter sweep -----
for lr in [3e-4, 1e-3, 3e-3]:
    torch.cuda.empty_cache()
    experiment_data["learning_rate"]["SPR_BENCH"][str(lr)] = train_and_eval(lr)
    del lr

# --------- save ---------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
