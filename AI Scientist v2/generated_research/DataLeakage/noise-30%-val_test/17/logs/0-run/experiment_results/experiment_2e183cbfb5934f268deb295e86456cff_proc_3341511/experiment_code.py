# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, DatasetDict

# --------------------- reproducibility & device ---------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- experiment dict ------------------------------
experiment_data = {
    "Constant_LR": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "configs": [],
        }
    }
}

# --------------------- data ----------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr_bench(DATA_PATH)

# --------------------- tokenisation ---------------------------------
chars = set("".join("".join(spr[sp]["sequence"]) for sp in spr))
vocab = {ch: i + 1 for i, ch in enumerate(sorted(chars))}
PAD_ID = 0
CLS_ID = len(vocab) + 1
vocab_size = CLS_ID + 1
max_len = max(max(len(s) for s in spr[sp]["sequence"]) for sp in spr) + 1


def encode(seq: str):
    return [CLS_ID] + [vocab[c] for c in seq][: max_len - 1]


def pad(seq_ids):
    return seq_ids + [PAD_ID] * (max_len - len(seq_ids))


class SPRTorchDataset(Dataset):
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
train_loader = DataLoader(SPRTorchDataset(spr["train"]), batch_size, shuffle=True)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"]), batch_size)
test_loader = DataLoader(SPRTorchDataset(spr["test"]), batch_size)


# --------------------- model ----------------------------------------
class LightTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, ids):
        x = self.embed(ids) + self.pos[: ids.size(1)]
        x = self.encoder(x)
        return self.fc(x[:, 0]).squeeze(1)


# --------------------- helpers --------------------------------------
class EarlyStop:
    def __init__(self, patience=3):
        self.best = None
        self.pat = patience
        self.cnt = 0
        self.stop = False

    def __call__(self, score):
        if self.best is None or score > self.best:
            self.best = score
            self.cnt = 0
        else:
            self.cnt += 1
            if self.cnt >= self.pat:
                self.stop = True
        return self.stop


def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.append((logits.sigmoid() > 0.5).cpu().numpy())
            gts.append(batch["labels"].cpu().numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    mcc = matthews_corrcoef(gts, preds)
    f1 = f1_score(gts, preds, average="macro")
    return tot_loss / len(loader.dataset), mcc, f1, preds, gts


train_labels = np.array(spr["train"]["label"])
pos_weight = torch.tensor(
    (len(train_labels) - train_labels.sum()) / train_labels.sum(),
    dtype=torch.float32,
    device=device,
)


# --------------------- experiment loop ------------------------------
def run_experiment(epochs=12, lr=1e-3):
    model = LightTransformer(vocab_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    stopper = EarlyStop(3)
    best_state, best_mcc = None, -1

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = criterion(model(batch["input_ids"]), batch["labels"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * batch["labels"].size(0)
        tr_loss = running / len(train_loader.dataset)
        _, tr_mcc, _, _, _ = evaluate(model, train_loader, criterion)
        val_loss, val_mcc, _, _, _ = evaluate(model, dev_loader, criterion)

        print(f"Ep {ep}: val_loss={val_loss:.4f} | val_MCC={val_mcc:.4f}")

        experiment_data["Constant_LR"]["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["Constant_LR"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["Constant_LR"]["SPR_BENCH"]["metrics"]["train"].append(tr_mcc)
        experiment_data["Constant_LR"]["SPR_BENCH"]["metrics"]["val"].append(val_mcc)

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_state = model.state_dict()
        if stopper(val_mcc):
            print("Early stopping")
            break

    # ----- testing -----
    model.load_state_dict(best_state)
    tst_loss, tst_mcc, tst_f1, preds, gts = evaluate(model, test_loader, criterion)
    print(f"TEST | MCC={tst_mcc:.4f} | MacroF1={tst_f1:.4f}")

    ed = experiment_data["Constant_LR"]["SPR_BENCH"]
    ed["predictions"].append(preds)
    ed["ground_truth"].append(gts)
    ed["configs"].append({"epochs": epochs, "lr": lr})


for ep in (10, 12):
    for lr in (1e-3, 5e-4):
        print(f"\n=== Constant LR Run: epochs={ep}, lr={lr} ===")
        run_experiment(epochs=ep, lr=lr)

# --------------------- save -----------------------------------------
os.makedirs("working", exist_ok=True)
np.save("working/experiment_data.npy", experiment_data, allow_pickle=True)
print("Saved metrics to working/experiment_data.npy")
