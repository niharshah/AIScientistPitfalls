import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, DatasetDict

# ---------------- Reproducibility ---------------- #
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- Device ------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Data loading ------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):  # helper
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr_bench(DATA_PATH)


# ---------------- Vocab & utils ------------------ #
def build_vocab(dsets):
    chars = set()
    for split in dsets.values():
        for s in split["sequence"]:
            chars.update(s)
    return {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 = PAD


vocab = build_vocab(spr)
vocab_size = len(vocab) + 1
max_len = max(max(len(s) for s in split["sequence"]) for split in spr.values())


def encode(seq: str):
    return [vocab[ch] for ch in seq]


def pad(ids, L):
    ids = ids[:L]
    return ids + [0] * (L - len(ids))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, max_len):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = pad(encode(self.seqs[idx]), self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], max_len), batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"], max_len), batch_size)
test_loader = DataLoader(SPRTorchDataset(spr["test"], max_len), batch_size)


# ---------------- Transformer model -------------- #
class TransformerSPR(nn.Module):
    def __init__(
        self, vocab_sz, max_len, emb_dim=128, nhead=4, nlayers=2, dff=256, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=dff,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, input_ids):
        pos_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(
            0
        )
        x = self.embed(input_ids) + self.pos(pos_ids)
        pad_mask = input_ids == 0
        x = self.enc(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(pooled).squeeze(1)


# ------------- Loss weighting for imbalance ------ #
labels_np = np.array(spr["train"]["label"])
pos_cnt = labels_np.sum()
neg_cnt = len(labels_np) - pos_cnt
pos_weight_tensor = torch.tensor(neg_cnt / pos_cnt, dtype=torch.float32, device=device)

# ------------- Experiment data container --------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "configs": [],
    }
}


# ------------- Evaluation ------------------------ #
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, preds, golds = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.append((logits.sigmoid() > 0.5).cpu().numpy())
        golds.append(batch["labels"].cpu().numpy())
    preds = np.concatenate(preds)
    golds = np.concatenate(golds)
    loss = tot_loss / len(loader.dataset)
    mcc = matthews_corrcoef(golds, preds)
    f1 = f1_score(golds, preds, average="macro")
    return loss, mcc, f1, preds, golds


class EarlyStop:
    def __init__(self, patience=3, mode="max", delta=1e-4):
        self.best = None
        self.wait = 0
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return False
        improve = (metric - self.best) if self.mode == "max" else (self.best - metric)
        if improve > self.delta:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True
        return self.stop


# ------------- Training loop --------------------- #
def train_run(lr=1e-3, epochs=10, patience=3):
    model = TransformerSPR(vocab_size, max_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.5, patience=2
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    stopper = EarlyStop(patience=patience, mode="max")
    best_state, best_f1 = None, -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
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
            running += loss.item() * batch["labels"].size(0)
        train_loss = running / len(train_loader.dataset)
        _, train_mcc, train_f1, _, _ = evaluate(model, train_loader, criterion)
        val_loss, val_mcc, val_f1, _, _ = evaluate(model, dev_loader, criterion)
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macro_f1 = {val_f1:.4f}"
        )
        scheduler.step(val_f1)

        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_f1)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
        if stopper(val_f1):
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    test_loss, test_mcc, test_f1, preds, golds = evaluate(model, test_loader, criterion)
    print(f"Test macro_F1 = {test_f1:.4f} | Test MCC = {test_mcc:.4f}")
    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(golds)
    experiment_data["SPR_BENCH"]["configs"].append(
        {"lr": lr, "epochs": epochs, "patience": patience}
    )


# ------------- Hyper-param trials ---------------- #
for lr in [1e-3, 5e-4]:
    print(f"\n=== Training run with lr={lr} ===")
    train_run(lr=lr, epochs=10, patience=3)

# ------------- Save experiment data -------------- #
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
