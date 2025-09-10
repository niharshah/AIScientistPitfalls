import os, pathlib, numpy as np, torch, torch.nn as nn, random, math, time
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------- working directory & device ----------
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
        pth = pathlib.Path(p).expanduser()
        if p and pth.joinpath("train.csv").exists():
            return pth.resolve()
    raise FileNotFoundError("SPR_BENCH not found.")


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ---------- helper: load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(DATA_PATH)

# ---------- build vocabulary ----------
all_tokens = {tok for ex in spr["train"] for tok in ex["sequence"].split()}
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1


def encode(seq):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, num_classes={num_classes}")


# ---------- metrics ----------
def count_shape_variety(sequence):
    return len({tok[0] for tok in sequence.split() if tok})


def count_color_variety(sequence):
    return len({tok[1] for tok in sequence.split() if len(tok) > 1})


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


# ---------- PyTorch dataset ----------
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
    maxlen = max(len(x["input_ids"]) for x in batch)
    ids, lab, raw = [], [], []
    for itm in batch:
        seq = itm["input_ids"]
        pad = maxlen - len(seq)
        if pad:
            seq = torch.cat([seq, torch.full((pad,), PAD_ID, dtype=torch.long)])
        ids.append(seq)
        lab.append(itm["label"])
        raw.append(itm["raw_seq"])
    return {"input_ids": torch.stack(ids), "label": torch.stack(lab), "raw_seq": raw}


train_loader_full = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader_full = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# ---------- model ----------
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


# ---------- hyper-parameter tuning (num_epochs) ----------
EPOCH_LIST = [6, 12, 18]

experiment_data = {"num_epochs": {"SPR_BENCH": {}}}


def train_for_epochs(num_epochs: int):
    model = BiLSTMClassifier(vocab_size, num_cls=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_data = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for ep in range(1, num_epochs + 1):
        # train
        model.train()
        tot_loss = 0
        n = 0
        for batch in train_loader_full:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optim.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optim.step()
            tot_loss += loss.item()
            n += 1
        train_loss = tot_loss / n
        run_data["losses"]["train"].append((ep, train_loss))

        # val
        model.eval()
        vtot = 0
        vn = 0
        preds, labels, raw = [], [], []
        with torch.no_grad():
            for batch in dev_loader_full:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                vtot += loss.item()
                vn += 1
                p = logits.argmax(-1).cpu().tolist()
                l = batch["label"].cpu().tolist()
                preds.extend(p)
                labels.extend(l)
                raw.extend(batch["raw_seq"])
        val_loss = vtot / vn
        run_data["losses"]["val"].append((ep, val_loss))
        swa = shape_weighted_accuracy(raw, labels, preds)
        cwa = color_weighted_accuracy(raw, labels, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        run_data["metrics"]["val"].append((ep, swa, cwa, hwa))
        run_data["predictions"], run_data["ground_truth"] = preds, labels
        print(
            f"[{num_epochs} ep run] Epoch {ep}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} HWA={hwa:.4f}"
        )
    return run_data


for ep_budget in EPOCH_LIST:
    experiment_data["num_epochs"]["SPR_BENCH"][ep_budget] = train_for_epochs(ep_budget)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
