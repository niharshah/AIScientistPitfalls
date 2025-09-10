import os, pathlib, time, json, numpy as np, torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from collections import Counter
from typing import List, Dict
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "weight_decay_tuning": {
        "SPR_BENCH": {
            "configs": [],  # list of weight_decay values tried
            "metrics": {"train": [], "val": []},  # each item is list over epochs
            "losses": {"train": [], "val": []},  #   ''   ''
            "predictions": [],  # final dev predictions per config
            "ground_truth": [],  # corresponding GT (same each run)
            "timestamps": [],  # end-time per epoch for each config
        }
    }
}
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# DATA --------------------------------------------------------------
def resolve_spr_path() -> pathlib.Path:
    candidates = []
    if "SPR_BENCH_PATH" in os.environ:
        candidates.append(os.environ["SPR_BENCH_PATH"])
    cwd = pathlib.Path.cwd()
    candidates += [
        cwd / "SPR_BENCH",
        cwd.parent / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for cand in candidates:
        p = pathlib.Path(cand)
        if (p / "train.csv").exists():
            print(f"Found SPR_BENCH dataset at {p.resolve()}")
            return p.resolve()
    raise FileNotFoundError(
        "SPR_BENCH dataset not found. Set env SPR_BENCH_PATH or place csvs in ./SPR_BENCH"
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


spr_root = resolve_spr_path()
spr = load_spr_bench(spr_root)
print("Loaded SPR_BENCH with sizes:", {k: len(v) for k, v in spr.items()})


# ----------------- VOCAB ------------------------------------------
def tokenize(seq: str) -> List[str]:
    return seq.strip().split()


all_tokens = [tok for seq in spr["train"]["sequence"] for tok in tokenize(seq)]
vocab_counter = Counter(all_tokens)
vocab = ["<PAD>", "<UNK>"] + sorted(vocab_counter)
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = stoi["<PAD>"], stoi["<UNK>"]

all_labels = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(all_labels)}


def encode(seq: str) -> List[int]:
    return [stoi.get(tok, unk_idx) for tok in tokenize(seq)]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lengths = [len(x["input_ids"]) for x in batch]
    maxlen = max(lengths)
    input_ids = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, : len(seq)] = seq
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "label": labels}


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ---------------- MODEL -------------------------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_labels, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(emb_dim, num_labels)
        self.pad = pad_idx

    def forward(self, x):
        mask = (x != self.pad).unsqueeze(-1)
        emb = self.emb(x)
        mean = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(mean))


# ---------------- TRAINING / TUNING -------------------------------
num_epochs = 5
weight_decays = [0.0, 1e-5, 1e-4, 1e-3]

for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    model = MeanPoolClassifier(len(vocab), 64, len(all_labels), pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    run_train_losses, run_val_losses = [], []
    run_train_f1, run_val_f1 = [], []
    run_timestamps = []

    for epoch in range(1, num_epochs + 1):
        # -------- train --------
        model.train()
        tr_loss, tr_preds, tr_trues = 0.0, [], []
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
            tr_loss += loss.item() * batch["label"].size(0)
            tr_preds.extend(logits.argmax(1).cpu().numpy())
            tr_trues.extend(batch["label"].cpu().numpy())
        tr_loss /= len(train_loader.dataset)
        tr_macro = f1_score(tr_trues, tr_preds, average="macro")

        # -------- validation -----
        model.eval()
        val_loss, val_preds, val_trues = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                val_loss += loss.item() * batch["label"].size(0)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_trues.extend(batch["label"].cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_macro = f1_score(val_trues, val_preds, average="macro")

        run_train_losses.append(tr_loss)
        run_val_losses.append(val_loss)
        run_train_f1.append(tr_macro)
        run_val_f1.append(val_macro)
        run_timestamps.append(time.time())

        print(
            f"Epoch {epoch}: wd={wd} | val_loss={val_loss:.4f} | val_MacroF1={val_macro:.4f}"
        )

    # ------------- log results ------------------------------------
    ed = experiment_data["weight_decay_tuning"]["SPR_BENCH"]
    ed["configs"].append(wd)
    ed["losses"]["train"].append(run_train_losses)
    ed["losses"]["val"].append(run_val_losses)
    ed["metrics"]["train"].append(run_train_f1)
    ed["metrics"]["val"].append(run_val_f1)
    ed["predictions"].append(val_preds)  # from last epoch
    ed["ground_truth"].append(val_trues)  # same each config
    ed["timestamps"].append(run_timestamps)

    # shape/color weighted accuracy for analysis
    def count_shape(sequence: str):
        return len(set(tok[0] for tok in sequence.strip().split() if tok))

    def count_color(sequence: str):
        return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))

    seqs_dev = spr["dev"]["sequence"]
    swa = sum(
        (count_shape(s) if t == p else 0)
        for s, t, p in zip(seqs_dev, val_trues, val_preds)
    ) / sum(map(count_shape, seqs_dev))
    cwa = sum(
        (count_color(s) if t == p else 0)
        for s, t, p in zip(seqs_dev, val_trues, val_preds)
    ) / sum(map(count_color, seqs_dev))
    print(f"Final Dev SWA: {swa:.4f} | CWA: {cwa:.4f}")

# ---------------- SAVE --------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
