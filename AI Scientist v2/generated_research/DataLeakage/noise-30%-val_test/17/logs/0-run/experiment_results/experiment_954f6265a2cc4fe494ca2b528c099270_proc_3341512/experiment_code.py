import os, pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, DatasetDict

# ---------------- paths / reproducibility ---------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# ---------------- data ----------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    _ld = lambda name: load_dataset(
        "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
    )
    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr_bench(DATA_PATH)

chars = set("".join("".join(spr[sp]["sequence"]) for sp in spr))
vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
PAD_ID, CLS_ID = 0, len(vocab) + 1
vocab_size, max_len = (
    CLS_ID + 1,
    max(len(s) for sp in spr for s in spr[sp]["sequence"]) + 1,
)
encode = lambda s: [CLS_ID] + [vocab[c] for c in s][: max_len - 1]


def pad(x):
    return x + [PAD_ID] * (max_len - len(x))


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        ids = torch.tensor(pad(encode(self.seqs[i])), dtype=torch.long)
        lab = torch.tensor(self.labels[i], dtype=torch.float32)
        return {"input_ids": ids, "labels": lab}


batch_size = 128
train_loader = DataLoader(SPRTorchDataset(spr["train"]), batch_size, shuffle=True)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"]), batch_size)
test_loader = DataLoader(SPRTorchDataset(spr["test"]), batch_size)


# ---------------- model: fixed sinusoidal PE ------------------------------------
def sinusoid_table(max_len, dim):
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / dim)
    )
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class LightTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        pe = sinusoid_table(max_len, d_model)
        self.register_buffer("pos", pe)  # non-trainable
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 256, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, ids):
        x = self.embed(ids) + self.pos[: ids.size(1)]
        out = self.encoder(x)
        return self.fc(out[:, 0]).squeeze(1)


# ---------------- utils ----------------------------------------------------------
class EarlyStop:
    def __init__(self, patience=3):
        self.pat, self.best, self.cnt, self.stop = patience, None, 0, False

    def __call__(self, score):
        if self.best is None or score > self.best:
            self.best, self.cnt = score, 0
        else:
            self.cnt += 1
            if self.cnt >= self.pat:
                self.stop = True
        return self.stop


def evaluate(model, loader, loss_fn):
    model.eval()
    tot, preds, gts = 0.0, [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            logit = model(b["input_ids"])
            loss = loss_fn(logit, b["labels"])
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


train_labels = np.array(spr["train"]["label"])
pos_weight = torch.tensor(
    (len(train_labels) - train_labels.sum()) / train_labels.sum(), device=device
)

experiment_data = {
    "fixed_sinusoidal": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "configs": [],
        }
    }
}


# ---------------- train loop -----------------------------------------------------
def run_experiment(epochs=12, lr=1e-3):
    model, opt = LightTransformer(vocab_size).to(device), torch.optim.AdamW
    model, optim = model, opt(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn, stopper, best_mcc, best = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        EarlyStop(3),
        -1,
        None,
    )
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(batch["input_ids"])
            loss = loss_fn(out, batch["labels"])
            loss.backward()
            optim.step()
            run += loss.item() * batch["labels"].size(0)
        sched.step()
        tr_loss = run / len(train_loader.dataset)
        _, tr_mcc, _, _, _ = evaluate(model, train_loader, loss_fn)
        val_loss, val_mcc, _, _, _ = evaluate(model, dev_loader, loss_fn)
        print(f"Epoch {ep}: val_loss={val_loss:.4f} val_MCC={val_mcc:.4f}")
        ed = experiment_data["fixed_sinusoidal"]["SPR_BENCH"]
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train"].append(tr_mcc)
        ed["metrics"]["val"].append(val_mcc)
        if val_mcc > best_mcc:
            best_mcc, best = val_mcc, model.state_dict()
        if stopper(val_mcc):
            print("Early stopping")
            break
    model.load_state_dict(best)
    tst_loss, tst_mcc, tst_f1, pred, gt = evaluate(model, test_loader, loss_fn)
    print(f"Test MCC={tst_mcc:.4f}  Test MacroF1={tst_f1:.4f}")
    ed["predictions"].append(pred)
    ed["ground_truth"].append(gt)
    ed["configs"].append({"epochs": epochs, "lr": lr})


# ---------------- run grid search ------------------------------------------------
for ep in (10, 12):
    for lr in (1e-3, 5e-4):
        print(f"\n=== epochs={ep} lr={lr} ===")
        run_experiment(ep, lr)

# ---------------- save -----------------------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved to working/experiment_data.npy")
