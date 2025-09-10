import os, pathlib, numpy as np, torch, torch.nn as nn, random, time
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict


# ---------------- misc utils --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- find SPR_BENCH ----------------
def find_spr_bench() -> pathlib.Path:
    cand = []
    if os.environ.get("SPR_DATA_PATH"):
        cand.append(os.environ["SPR_DATA_PATH"])
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


# ---------------- load dataset -----------------
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


spr = load_spr_bench(DATA_PATH)

# ---------------- vocabulary --------------------
all_tokens = set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}  num_classes={num_classes}")


def encode(seq: str):
    return [token2id[t] for t in seq.split()]


# -------------- metrics helpers ----------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0


# ---------------- torch dataset ----------------
class SPRTorchSet(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]
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
    maxlen = max(len(x["input_ids"]) for x in batch)
    ids, lbls, raws = [], [], []
    for item in batch:
        seq = item["input_ids"]
        pad = maxlen - len(seq)
        if pad:
            seq = torch.cat([seq, torch.full((pad,), PAD_ID, dtype=torch.long)])
        ids.append(seq)
        lbls.append(item["label"])
        raws.append(item["raw_seq"])
    return {
        "input_ids": torch.stack(ids),
        "label": torch.stack(lbls),
        "raw_seq": raws,
    }


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# ---------------- model ------------------------
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
        feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(feat)


# ------------- experiment container -------------
experiment_data = {"weight_decay": {"SPR_BENCH": {}}}

# ------------- training over weight decay -------
EPOCHS = 6
weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]

for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    run_key = f"wd_{wd}"
    exp_run = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    model = BiLSTMClassifier(vocab_size, num_cls=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
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
        exp_run["losses"]["train"].append((epoch, train_loss))

        # ---- validation ----
        model.eval()
        v_tot, v_nb = 0.0, 0
        all_preds, all_lbls, all_seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                v_tot += loss.item()
                v_nb += 1
                preds = logits.argmax(-1).cpu().tolist()
                lbls = batch["label"].cpu().tolist()
                all_preds.extend(preds)
                all_lbls.extend(lbls)
                all_seqs.extend(batch["raw_seq"])
        val_loss = v_tot / v_nb
        exp_run["losses"]["val"].append((epoch, val_loss))

        swa = shape_weighted_accuracy(all_seqs, all_lbls, all_preds)
        cwa = color_weighted_accuracy(all_seqs, all_lbls, all_preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        exp_run["metrics"]["val"].append((epoch, swa, cwa, hwa))

        if epoch == EPOCHS:  # store predictions once
            exp_run["predictions"] = all_preds
            exp_run["ground_truth"] = all_lbls

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"SWA={swa:.4f}  CWA={cwa:.4f}  HWA={hwa:.4f}"
        )

    experiment_data["weight_decay"]["SPR_BENCH"][run_key] = exp_run

    # free memory for next run
    del model, optimizer, criterion
    torch.cuda.empty_cache()

# ------------- save experiment data -------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to {os.path.join(working_dir, 'experiment_data.npy')}")
