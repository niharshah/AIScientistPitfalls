import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, DatasetDict

# ----------------- I/O & working dir -----------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- Reproducibility ------------------------------ #
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------- Device ---------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- Load SPR_BENCH -------------------------------#
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


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


spr = load_spr_bench(DATA_PATH)


# ----------------- Build vocab & encoding helpers ---------------#
def build_vocab(dsets) -> dict:
    chars = set()
    for split in dsets.values():
        chars.update("".join(split["sequence"]))
    return {ch: i + 1 for i, ch in enumerate(sorted(chars))}  # 0 = PAD


vocab = build_vocab(spr)
vocab_size = len(vocab) + 1
max_len = max(max(len(s) for s in split["sequence"]) for split in spr.values())


def encode_sequence(seq: str) -> list[int]:
    return [vocab[ch] for ch in seq]


def pad(seq_ids: list[int], L: int) -> list[int]:
    return seq_ids[:L] + [0] * (L - len(seq_ids)) if len(seq_ids) < L else seq_ids[:L]


# ----------------- Torch Dataset --------------------------------#
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, max_len: int):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = pad(encode_sequence(self.seqs[idx]), self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ----------------- DataLoaders ----------------------------------#
batch_size = 256  # slightly larger batch for tuning
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], max_len), batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"], max_len), batch_size)
test_loader = DataLoader(SPRTorchDataset(spr["test"], max_len), batch_size)


# ----------------- Model ----------------------------------------#
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


# ----------------- Early Stopping -------------------------------#
class EarlyStopping:
    def __init__(self, patience: int = 4, min_delta: float = 1e-4, mode: str = "max"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = None
        self.count = 0
        self.stop = False

    def __call__(self, metric: float):
        if self.best is None:
            self.best = metric
            return False
        improve = (metric - self.best) if self.mode == "max" else (self.best - metric)
        if improve > self.min_delta:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True
        return self.stop


# ----------------- Experiment container -------------------------#
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epoch_budgets": [],
    }
}


# ----------------- Train routine --------------------------------#
def run_training(max_epochs: int = 30, patience: int = 4, lr: float = 3e-4):
    model = GRUBaseline(vocab_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    es = EarlyStopping(patience=patience, mode="max")
    best_state, best_mcc = None, -1.0

    for epoch in range(1, max_epochs + 1):
        # ----- train --------------------------------------------------#
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch["labels"].size(0)
        train_loss = epoch_loss / len(train_loader.dataset)

        # -- train metrics
        model.eval()
        tr_preds, tr_labels = [], []
        with torch.no_grad():
            for batch in train_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                out = torch.sigmoid(model(batch["input_ids"]))
                tr_preds.append((out > 0.5).cpu().numpy())
                tr_labels.append(batch["labels"].cpu().numpy())
        tr_preds = np.concatenate(tr_preds)
        tr_labels = np.concatenate(tr_labels)
        train_mcc = matthews_corrcoef(tr_labels, tr_preds)
        train_f1 = f1_score(tr_labels, tr_preds, average="macro")

        # ----- validation ---------------------------------------------#
        val_loss, v_preds, v_labels = 0.0, [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                logits = model(batch["input_ids"])
                val_loss += criterion(logits, batch["labels"]).item() * batch[
                    "labels"
                ].size(0)
                v_preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy())
                v_labels.append(batch["labels"].cpu().numpy())
        val_loss /= len(dev_loader.dataset)
        v_preds = np.concatenate(v_preds)
        v_labels = np.concatenate(v_labels)
        val_mcc = matthews_corrcoef(v_labels, v_preds)
        val_f1 = f1_score(v_labels, v_preds, average="macro")

        # logging
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(
            {"mcc": train_mcc, "macro_f1": train_f1}
        )
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            {"mcc": val_mcc, "macro_f1": val_f1}
        )
        print(
            f"Epoch {epoch}/{max_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_MCC={val_mcc:.4f}, val_macroF1={val_f1:.4f}"
        )

        # early stop
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_state = model.state_dict()
        if es(val_mcc):
            print("Early stopping.")
            break

    # ---------------- test -------------------------------------------#
    model.load_state_dict(best_state)
    model.eval()
    t_preds, t_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"])
            t_preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy())
            t_labels.append(batch["labels"].cpu().numpy())
    t_preds = np.concatenate(t_preds)
    t_labels = np.concatenate(t_labels)
    test_mcc = matthews_corrcoef(t_labels, t_preds)
    test_f1 = f1_score(t_labels, t_preds, average="macro")
    print(f"Test MCC={test_mcc:.4f}, Test macroF1={test_f1:.4f}")

    experiment_data["SPR_BENCH"]["predictions"].append(t_preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(t_labels)
    experiment_data["SPR_BENCH"]["epoch_budgets"].append(max_epochs)


for ep_budget in [10, 20, 30]:
    start = time.time()
    print(f"\n=== Training with max_epochs={ep_budget} ===")
    run_training(max_epochs=ep_budget, patience=5, lr=3e-4)
    print(f"Elapsed: {time.time()-start:.1f}s")

# ----------------- Save experiment data --------------------------#
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved all results to", os.path.join(working_dir, "experiment_data.npy"))
