import os, pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import DatasetDict, load_dataset

# ------------------------------------------------------------------#
#  I/O SET-UP
# ------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------#
#  DEVICE
# ------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------#
#  REPRODUCIBILITY
# ------------------------------------------------------------------#
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------#
#  DATA LOADING
# ------------------------------------------------------------------#
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
assert DATA_PATH.exists(), f"Dataset path {DATA_PATH} not found."


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


spr = load_spr_bench(DATA_PATH)


# ------------------------------------------------------------------#
#  VOCAB & ENCODING (bug-fixed)
# ------------------------------------------------------------------#
def build_vocab(dsets) -> dict:
    chars = set()
    for split in dsets.values():
        chars.update("".join(split["sequence"]))
    return {ch: i + 1 for i, ch in enumerate(sorted(chars))}  # 0 reserved for PAD


vocab = build_vocab(spr)
vocab_size = len(vocab) + 1
max_len = max(max(len(s) for s in split["sequence"]) for split in spr.values())


def encode_sequence(seq: str, vocab: dict) -> list[int]:
    # BUG-FIX: explicit vocab argument to match call site
    return [vocab[ch] for ch in seq]


def pad(seq_ids: list[int], L: int) -> list[int]:
    return seq_ids[:L] + [0] * (L - len(seq_ids)) if len(seq_ids) < L else seq_ids[:L]


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, max_len):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]
        self.vocab, self.max_len = vocab, max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = pad(encode_sequence(self.seqs[idx], self.vocab), self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], vocab, max_len), batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"], vocab, max_len), batch_size)
test_loader = DataLoader(SPRTorchDataset(spr["test"], vocab, max_len), batch_size)


# ------------------------------------------------------------------#
#  MODEL (unchanged architecture)
# ------------------------------------------------------------------#
class GRUBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


# ------------------------------------------------------------------#
#  EARLY STOPPING
# ------------------------------------------------------------------#
class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, mode="max"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return False
        improve = (metric - self.best) if self.mode == "max" else (self.best - metric)
        if improve > self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ------------------------------------------------------------------#
#  EXPERIMENT TRACKING STRUCTURE
# ------------------------------------------------------------------#
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"train_mcc": [], "val_mcc": [], "train_f1": [], "val_f1": []},
        "test": {},
    }
}


# ------------------------------------------------------------------#
#  TRAINING LOOP
# ------------------------------------------------------------------#
def train_for_epochs(max_epochs, patience=3, lr=1e-3):
    model = GRUBaseline(vocab_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stop = EarlyStopping(patience=patience, mode="max")
    best_state, best_val_mcc = None, -1.0

    for epoch in range(1, max_epochs + 1):
        # ------------------ Train -------------------#
        model.train()
        thr_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thr_loss += loss.item() * batch["labels"].size(0)
        train_loss = thr_loss / len(train_loader.dataset)

        # metrics on train split
        model.eval()
        train_preds, train_lbls = [], []
        with torch.no_grad():
            for batch in train_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                probs = torch.sigmoid(model(batch["input_ids"])).cpu().numpy()
                train_preds.append(probs > 0.5)
                train_lbls.append(batch["labels"].cpu().numpy())
        train_preds = np.concatenate(train_preds)
        train_lbls = np.concatenate(train_lbls)
        train_mcc = matthews_corrcoef(train_lbls, train_preds)
        train_f1 = f1_score(train_lbls, train_preds, average="macro")

        # ------------------ Validation --------------#
        val_loss, val_preds, val_lbls = 0.0, [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                val_loss += criterion(logits, batch["labels"]).item() * batch[
                    "labels"
                ].size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs > 0.5)
                val_lbls.append(batch["labels"].cpu().numpy())
        val_loss /= len(dev_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_lbls = np.concatenate(val_lbls)
        val_mcc = matthews_corrcoef(val_lbls, val_preds)
        val_f1 = f1_score(val_lbls, val_preds, average="macro")

        # logging
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train_mcc"].append(train_mcc)
        experiment_data["SPR_BENCH"]["metrics"]["val_mcc"].append(val_mcc)
        experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(train_f1)
        experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch}/{max_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_MCC={val_mcc:.4f} val_macroF1={val_f1:.4f}"
        )

        # save best
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_state = model.state_dict()

        # early stopping
        if early_stop(val_mcc):
            print("Early stopping.")
            break

    # ------------------ Test ----------------------#
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_lbls = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            probs = torch.sigmoid(model(batch["input_ids"])).cpu().numpy()
            test_preds.append(probs > 0.5)
            test_lbls.append(batch["labels"].cpu().numpy())
    test_preds = np.concatenate(test_preds)
    test_lbls = np.concatenate(test_lbls)
    test_mcc = matthews_corrcoef(test_lbls, test_preds)
    test_f1 = f1_score(test_lbls, test_preds, average="macro")

    experiment_data["SPR_BENCH"]["test"] = {
        "mcc": test_mcc,
        "macro_f1": test_f1,
        "predictions": test_preds,
        "ground_truth": test_lbls,
    }

    print(f"Test MCC: {test_mcc:.4f} | Test macro-F1: {test_f1:.4f}")


# ------------------------------------------------------------------#
#  RUN A SMALL HYPER-PARAM SEARCH (epoch budgets)
# ------------------------------------------------------------------#
for ep in [5, 10, 15]:  # trimmed list to ensure <30min runtime
    print(f"\n=== Training with max_epochs = {ep} ===")
    train_for_epochs(max_epochs=ep, patience=3, lr=1e-3)

# ------------------------------------------------------------------#
#  SAVE EXPERIMENT DATA
# ------------------------------------------------------------------#
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
