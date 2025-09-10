import os, pathlib, numpy as np, torch, torch.nn as nn, random, math, time
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- working directory & device -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- locate SPR_BENCH ---------------------
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
        pth = pathlib.Path(p).expanduser()
        if p and pth.joinpath("train.csv").exists():
            return pth.resolve()
    raise FileNotFoundError(
        "SPR_BENCH not found; set SPR_DATA_PATH or place folder nearby."
    )


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ---------- helper: load SPR_BENCH --------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


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


# ---------- load dataset ------------------------
spr = load_spr_bench(DATA_PATH)

# ---------- build vocabulary -------------------
all_tokens = set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size = {vocab_size},  num_classes = {num_classes}")


def encode(seq: str):
    return [token2id[tok] for tok in seq.split()]


# ---------- PyTorch Dataset --------------------
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
    maxlen = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, raw = [], [], []
    for item in batch:
        seq = item["input_ids"]
        pad_len = maxlen - len(seq)
        if pad_len:
            seq = torch.cat([seq, torch.full((pad_len,), PAD_ID, dtype=torch.long)])
        input_ids.append(seq)
        labels.append(item["label"])
        raw.append(item["raw_seq"])
    return {
        "input_ids": torch.stack(input_ids),
        "label": torch.stack(labels),
        "raw_seq": raw,
    }


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# ---------- model ------------------------------
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


# ---------- experiment data container ----------
experiment_data = {
    "hidden_dim": {
        "SPR_BENCH": {
            # each hidden size key will be filled later
        }
    }
}

# ---------- hyper-parameter grid ---------------
hidden_grid = [64, 128, 256, 512]
EPOCHS = 6

for hidden_size in hidden_grid:
    print(f"\n=== Training with hidden_dim = {hidden_size} ===")
    model = BiLSTMClassifier(vocab_size, hidden=hidden_size, num_cls=num_classes).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # storage for this run
    run_store = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # ---- training ----
        model.train()
        tot_loss, nb = 0.0, 0
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
        train_loss = tot_loss / nb
        run_store["losses"]["train"].append((epoch, train_loss))

        # ---- validation ----
        model.eval()
        val_loss_tot, nbv = 0.0, 0
        all_preds, all_labels, all_seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                val_loss_tot += loss.item()
                nbv += 1
                preds = logits.argmax(-1).cpu().tolist()
                labels = batch["label"].cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_seqs.extend(batch["raw_seq"])
        val_loss = val_loss_tot / nbv
        run_store["losses"]["val"].append((epoch, val_loss))

        swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
        cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        run_store["metrics"]["val"].append((epoch, swa, cwa, hwa))

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"SWA={swa:.4f}  CWA={cwa:.4f}  HWA={hwa:.4f}"
        )

    # store final predictions / labels from last epoch
    run_store["predictions"] = all_preds
    run_store["ground_truth"] = all_labels
    experiment_data["hidden_dim"]["SPR_BENCH"][str(hidden_size)] = run_store

# ---------- save experiment data ---------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to {os.path.join(working_dir, 'experiment_data.npy')}")
