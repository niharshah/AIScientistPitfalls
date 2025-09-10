import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- basic imports -----------------
import pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, f1_score

# ----------------- device / seeds ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------- data loading -----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(fname):
        return load_dataset(
            "csv",
            data_files=str(root / fname),
            split="train",  # treat each csv as one split
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr_bench(DATA_PATH)

# ----------------- vocab / encoding -------------
chars = set("".join("".join(spr[split]["sequence"]) for split in spr))
vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
PAD_ID = 0
vocab_size = len(vocab) + 1
max_len = max(max(len(s) for s in spr[split]["sequence"]) for split in spr)


def encode(seq: str):
    return [vocab[c] for c in seq][:max_len]


def pad(ids):
    return ids + [PAD_ID] * (max_len - len(ids))


# ----------------- torch dataset ---------------
class SPRTorchDS(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = pad(encode(self.seqs[idx]))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


batch_size = 128
train_loader = DataLoader(SPRTorchDS(spr["train"]), batch_size, shuffle=True)
dev_loader = DataLoader(SPRTorchDS(spr["dev"]), batch_size)
test_loader = DataLoader(SPRTorchDS(spr["test"]), batch_size)


# ----------------- model (bug-fixed) ------------
class LightTransformerNoCLS(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, ids):
        # padding mask: True where PAD
        pad_mask = ids.eq(PAD_ID)
        x = self.embed(ids) + self.pos[: ids.size(1)]
        out = self.encoder(x, src_key_padding_mask=pad_mask)  # <-- FIX HERE
        valid_mask = (~pad_mask).unsqueeze(-1)
        pooled = (out * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1)
        return self.fc(pooled).squeeze(1)


# ----------------- utils ------------------------
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.append((logits.sigmoid() > 0.5).cpu().numpy())
            gts.append(batch["labels"].cpu().numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return (
        tot_loss / len(loader.dataset),
        matthews_corrcoef(gts, preds),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


class EarlyStop:
    def __init__(self, patience=3):
        self.patience, self.best, self.count, self.stop = patience, -1.0, 0, False

    def __call__(self, metric):
        if metric > self.best:
            self.best, self.count = metric, 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True
        return self.stop


# -------------- imbalance weight ---------------
train_labels = np.array(spr["train"]["label"])
pos_weight = torch.tensor(
    (len(train_labels) - train_labels.sum()) / train_labels.sum(),
    dtype=torch.float32,
    device=device,
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# -------------- experiment log ------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "configs": [],
    }
}


# -------------- training routine ---------------
def run(epochs=10, lr=1e-3):
    model = LightTransformerNoCLS(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stopper = EarlyStop(3)
    best_state = None
    best_val = -1.0

    for ep in range(1, epochs + 1):
        # ---- train ----
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
        scheduler.step()
        tr_loss = running / len(train_loader.dataset)
        tr_metric = evaluate(model, train_loader, criterion)[1]

        # ---- val ----
        val_loss, val_mcc, _, _, _ = evaluate(model, dev_loader, criterion)
        print(f"Epoch {ep}: val_loss={val_loss:.4f} | val_MCC={val_mcc:.4f}")

        ed = experiment_data["SPR_BENCH"]
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train"].append(tr_metric)
        ed["metrics"]["val"].append(val_mcc)

        if val_mcc > best_val:
            best_val, best_state = val_mcc, model.state_dict()
        if stopper(val_mcc):
            print("Early stopping")
            break

    # ---- test ----
    model.load_state_dict(best_state)
    test_loss, test_mcc, test_f1, preds, gts = evaluate(model, test_loader, criterion)
    print(f"Test MCC={test_mcc:.4f} | Test MacroF1={test_f1:.4f}")

    ed["predictions"].append(preds)
    ed["ground_truth"].append(gts)
    ed["configs"].append({"epochs": epochs, "lr": lr, "best_val_mcc": best_val})


# -------------- small grid search --------------
for ep in (10, 12):
    for lr in (1e-3, 5e-4):
        print(f"\n=== run: epochs={ep}, lr={lr} ===")
        run(epochs=ep, lr=lr)

# -------------- save everything ----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics to working/experiment_data.npy")
