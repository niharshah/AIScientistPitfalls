import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
from datasets import load_dataset, DatasetDict

# ---------------- reproducibility ------------------------------------------------
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- data -----------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr_bench(DATA_PATH)

# build vocabulary
chars = set("".join("".join(spr[split]["sequence"]) for split in spr.keys()))
vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 1..|V|
PAD_ID = 0
CLS_ID = len(vocab) + 1
vocab_size = CLS_ID + 1
max_len = max(max(len(s) for s in spr[sp]["sequence"]) for sp in spr) + 1  # +CLS


def encode(seq: str):
    return [CLS_ID] + [vocab[c] for c in seq][: max_len - 1]


def pad(seq_ids):
    return seq_ids + [PAD_ID] * (max_len - len(seq_ids))


feat_dim = len(vocab) * 2 + 1  # counts + parity + length


def feature_vector(seq: str):
    counts = np.zeros(len(vocab), dtype=np.float32)
    for c in seq:
        counts[vocab[c] - 1] += 1
    parity = counts % 2
    length = np.array([len(seq)], dtype=np.float32)
    return np.concatenate([counts, parity, length])


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        # precompute to speed up
        self.ids_list = [
            torch.tensor(pad(encode(s)), dtype=torch.long) for s in self.seqs
        ]
        self.feats_list = [
            torch.tensor(feature_vector(s), dtype=torch.float32) for s in self.seqs
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.ids_list[idx],
            "features": self.feats_list[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


batch_size = 128
train_loader = DataLoader(SPRTorchDataset(spr["train"]), batch_size, shuffle=True)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"]), batch_size)
test_loader = DataLoader(SPRTorchDataset(spr["test"]), batch_size)


# ---------------- model ----------------------------------------------------------
class HybridSPR(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(2 * d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, ids, feats):
        x = self.embed(ids) + self.pos[: ids.size(1)]
        h = self.encoder(x)[:, 0]  # CLS token
        f = self.feat_proj(feats)
        out = self.cls_head(torch.cat([h, f], dim=-1)).squeeze(1)
        return out


# ---------------- utils ----------------------------------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["features"])
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)
            preds.append((logits.sigmoid() > 0.5).cpu().numpy())
            gts.append(batch["labels"].cpu().numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    mcc = matthews_corrcoef(gts, preds)
    return total_loss / len(loader.dataset), mcc, preds, gts


train_labels = np.array(spr["train"]["label"])
pos_weight = torch.tensor(
    (len(train_labels) - train_labels.sum()) / train_labels.sum(),
    dtype=torch.float32,
    device=device,
)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "configs": [],
    }
}


class EarlyStop:
    def __init__(self, patience=4):
        self.patience = patience
        self.best = -1
        self.count = 0

    def step(self, score):
        if score > self.best:
            self.best = score
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


def run(epochs=20, lr=1e-3):
    model = HybridSPR(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    stopper = EarlyStop(5)
    best_state, best_mcc = None, -1

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["features"])
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * batch["labels"].size(0)
        scheduler.step()
        tr_loss = running / len(train_loader.dataset)
        tr_mcc = evaluate(model, train_loader, criterion)[1]

        # ---- validation ----
        val_loss, val_mcc, _, _ = evaluate(model, dev_loader, criterion)
        print(f"Epoch {ep}: validation_loss = {val_loss:.4f} | val_MCC = {val_mcc:.4f}")

        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_mcc)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_mcc)

        if val_mcc > best_mcc:
            best_mcc, best_state = val_mcc, model.state_dict()

        if stopper.step(val_mcc):
            print("Early stopping triggered")
            break

    # ---- test ----
    model.load_state_dict(best_state)
    test_loss, test_mcc, preds, gts = evaluate(model, test_loader, criterion)
    print(f"Test MCC = {test_mcc:.4f}")

    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(gts)


for lr in (1e-3, 5e-4):
    print(f"\n===== Running experiment: lr={lr} =====")
    run(epochs=20, lr=lr)
    experiment_data["SPR_BENCH"]["configs"].append({"epochs": 20, "lr": lr})

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics to working/experiment_data.npy")
