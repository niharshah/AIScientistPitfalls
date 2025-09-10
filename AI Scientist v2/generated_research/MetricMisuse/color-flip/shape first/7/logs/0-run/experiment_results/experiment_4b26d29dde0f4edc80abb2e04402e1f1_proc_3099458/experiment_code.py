import os, pathlib, numpy as np, torch, torch.nn as nn, random, math, time
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------------- reproducibility ----------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# --------------- working directory & device ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- locate SPR_BENCH ---------------------------
def find_spr_bench() -> pathlib.Path:
    cand = [
        os.environ.get("SPR_DATA_PATH"),
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


# ---------------- helper: load SPR_BENCH ---------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(DATA_PATH)


# ---------------- misc metric helpers ------------------------
def count_shape_variety(seq: str):
    return len(set(t[0] for t in seq.split() if t))


def count_color_variety(seq: str):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


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


# ------------------- vocabulary ------------------------------
all_tokens = set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1


def encode(seq: str):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, classes={num_classes}")


# ----------------- dataset & dataloader ----------------------
class SPRTorchSet(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]
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
    pad = lambda x: (
        torch.cat([x, torch.full((maxlen - len(x),), PAD_ID, dtype=torch.long)])
        if len(x) < maxlen
        else x
    )
    return {
        "input_ids": torch.stack([pad(b["input_ids"]) for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# -------------------- model def --------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hidden=128, num_cls=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_cls)

    def forward(self, x):
        emb = self.embed(x)
        lens = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = torch.cat([h_n[-2], h_n[-1]], 1)
        return self.fc(out)


# -------------------- hyper-parameter sweep --------------------
EPOCHS = 6
embed_dims = [32, 64, 128, 256]
experiment_data = {}
for ed in embed_dims:
    tag = f"emb_dim_{ed}"
    experiment_data[tag] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    # fresh model/optim
    model = BiLSTMClassifier(vocab_size, emb_dim=ed, num_cls=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        tot_loss = 0.0
        nb = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            nb += 1
        tr_loss = tot_loss / nb
        experiment_data[tag]["SPR_BENCH"]["losses"]["train"].append((epoch, tr_loss))
        # ---- validate ----
        model.eval()
        v_loss = 0.0
        nb = 0
        preds = []
        labels = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                v_loss += loss.item()
                nb += 1
                p = logits.argmax(-1).cpu().tolist()
                l = batch["label"].cpu().tolist()
                preds.extend(p)
                labels.extend(l)
                seqs.extend(batch["raw_seq"])
        v_loss /= nb
        swa = shape_weighted_accuracy(seqs, labels, preds)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        experiment_data[tag]["SPR_BENCH"]["losses"]["val"].append((epoch, v_loss))
        experiment_data[tag]["SPR_BENCH"]["metrics"]["val"].append(
            (epoch, swa, cwa, hwa)
        )
        experiment_data[tag]["SPR_BENCH"]["predictions"] = preds
        experiment_data[tag]["SPR_BENCH"]["ground_truth"] = labels
        print(
            f"[{tag}] Ep{epoch}: tr_loss={tr_loss:.4f} val_loss={v_loss:.4f} "
            f"SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )

# ------------------ save all experiment data ------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
