# No-Weight-Decay Optimiser Ablation ─────────────────────────────────────────────
import os, pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, DatasetDict

# ---------------- reproducibility ------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- paths ----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")  # expected path


# ---------------- data -----------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    ld = lambda name: load_dataset(
        "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
    )
    return DatasetDict(train=ld("train.csv"), dev=ld("dev.csv"), test=ld("test.csv"))


spr = load_spr_bench(DATA_PATH)

# build vocabulary ----------------------------------------------------------------
chars = set("".join("".join(spr[sp]["sequence"]) for sp in spr))
vocab = {ch: i + 1 for i, ch in enumerate(sorted(chars))}
PAD_ID, CLS_ID = 0, len(vocab) + 1
vocab_size, max_len = CLS_ID + 1, max(len(s) for s in spr["train"]["sequence"]) + 1


def encode(seq: str):  # prepend CLS
    return [CLS_ID] + [vocab[c] for c in seq][: max_len - 1]


def pad(ids):
    return ids + [PAD_ID] * (max_len - len(ids))


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = torch.tensor(pad(encode(self.seqs[idx])), dtype=torch.long)
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"input_ids": ids, "labels": lbl}


batch_size = 128
train_loader = DataLoader(SPRTorchDataset(spr["train"]), batch_size, shuffle=True)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"]), batch_size)
test_loader = DataLoader(SPRTorchDataset(spr["test"]), batch_size)


# ---------------- model ----------------------------------------------------------
class LightTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, layers=2, drop=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 256, dropout=drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, ids):
        x = self.embed(ids) + self.pos[: ids.size(1)]
        h = self.encoder(x)
        return self.fc(h[:, 0]).squeeze(1)


# ---------------- utils ----------------------------------------------------------
class EarlyStop:
    def __init__(self, patience=3):
        self.p, self.best, self.cnt, self.flag = patience, None, 0, False

    def __call__(self, score):
        if self.best is None or score > self.best:
            self.best, self.cnt = score, 0
        else:
            self.cnt += 1
            if self.cnt >= self.p:
                self.flag = True
        return self.flag


def evaluate(model, loader, crit):
    model.eval()
    tot, preds, gts = 0.0, [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            logit = model(b["input_ids"])
            loss = crit(logit, b["labels"])
            tot += loss.item() * b["labels"].size(0)
            preds.append((logit.sigmoid() > 0.5).cpu().numpy())
            gts.append(b["labels"].cpu().numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    return (
        tot / len(loader.dataset),
        matthews_corrcoef(gts, preds),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# class imbalance for BCEWithLogitsLoss ------------------------------------------
train_labels = np.array(spr["train"]["label"])
pos_weight = torch.tensor(
    (len(train_labels) - train_labels.sum()) / train_labels.sum(),
    dtype=torch.float32,
    device=device,
)

# ---------------- experiment dict ------------------------------------------------
experiment_data = {
    "NoWeightDecay": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "configs": [],
        }
    }
}
EXP = experiment_data["NoWeightDecay"]["SPR_BENCH"]


def run_experiment(epochs=12, lr=1e-3):
    model, best_state, best_mcc = LightTransformer(vocab_size).to(device), None, -1
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)  # key change
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    stopper = EarlyStop(3)

    for ep in range(1, epochs + 1):
        # ---- training ----
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = crit(model(batch["input_ids"]), batch["labels"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * batch["labels"].size(0)
        sched.step()
        tr_loss = running / len(train_loader.dataset)
        _, tr_mcc, _, _, _ = evaluate(model, train_loader, crit)
        # ---- validation --
        val_loss, val_mcc, _, _, _ = evaluate(model, dev_loader, crit)
        print(f"Epoch {ep}: val_loss={val_loss:.4f} | val_MCC={val_mcc:.4f}")

        # log
        EXP["losses"]["train"].append(tr_loss)
        EXP["losses"]["val"].append(val_loss)
        EXP["metrics"]["train"].append(tr_mcc)
        EXP["metrics"]["val"].append(val_mcc)

        if val_mcc > best_mcc:
            best_mcc, best_state = val_mcc, model.state_dict()
        if stopper(val_mcc):
            print("Early stopping triggered")
            break

    # ---- test -------------------------------------------------------------------
    model.load_state_dict(best_state)
    test_loss, test_mcc, test_f1, preds, gts = evaluate(model, test_loader, crit)
    print(f"TEST: loss={test_loss:.4f} | MCC={test_mcc:.4f} | MacroF1={test_f1:.4f}")

    EXP["predictions"].append(preds)
    EXP["ground_truth"].append(gts)
    EXP["configs"].append({"epochs": epochs, "lr": lr})
    return test_mcc


# ---------------- grid search ----------------------------------------------------
for ep in (10, 12):
    for lr in (1e-3, 5e-4):
        print(f"\n===== RUN: epochs={ep}, lr={lr} =====")
        run_experiment(epochs=ep, lr=lr)

# ---------------- save -----------------------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved results to working/experiment_data.npy")
