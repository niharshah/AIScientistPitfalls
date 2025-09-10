import os, pathlib, random, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, f1_score
import matplotlib.pyplot as plt

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- data helpers ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def get_dataset() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Found real SPR_BENCH at", p)
            return load_spr_bench(p)
    print("Real SPR_BENCH not found – generating synthetic toy data")

    def synth(n):
        rows, shapes = "ABCD", []
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 15)))
            label = int(seq.count("A") % 2 == 0)
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    def to_ds(rows):
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    return DatasetDict(
        train=to_ds(synth(4000)), dev=to_ds(synth(800)), test=to_ds(synth(800))
    )


dsets = get_dataset()

# ---------- vocabulary ----------
all_text = "".join(dsets["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 2 for i, ch in enumerate(vocab)}  # 0 PAD, 1 CLS
itos = {i: ch for ch, i in stoi.items()}
CLS_ID = 1
PAD_ID = 0
vocab_size = len(stoi) + 2
max_len = min(120, max(map(len, dsets["train"]["sequence"])) + 1)  # +1 for CLS


def encode(seq: str):
    ids = [CLS_ID] + [stoi.get(ch, 0) for ch in seq[: max_len - 1]]
    return ids + [PAD_ID] * (max_len - len(ids))


def sym_features(seq: str):
    # parity for each symbol, first&last symbol one-hot, length bucket (<=8,<=12,else)
    feats = []
    for ch in vocab:
        feats.append(int(seq.count(ch) % 2))  # parity
    first, last = seq[0], seq[-1]
    feats += [int(first == ch) for ch in vocab]
    feats += [int(last == ch) for ch in vocab]
    L = len(seq)
    feats += [int(L <= 8), int(8 < L <= 12), int(L > 12)]
    return feats


feat_dim = len(vocab) * 3 + 3


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seq, self.y = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "sym_feats": torch.tensor(sym_features(self.seq[idx]), dtype=torch.float),
            "label": torch.tensor(int(self.y[idx]), dtype=torch.float),
        }


def loader(name, batch=128, shuffle=False):
    return DataLoader(
        SPRDataset(dsets[name]),
        batch_size=batch,
        shuffle=shuffle,
        collate_fn=lambda b: {k: torch.stack([x[k] for x in b]) for k in b[0]},
    )


train_loader = lambda: loader("train", 128, True)
dev_loader = lambda: loader("dev", 256, False)
test_loader = lambda: loader("test", 256, False)


# ---------- model ----------
class HybridTransformer(nn.Module):
    def __init__(
        self, vocab_sz, feat_dim, d_model=96, nhead=4, num_layers=2, dropout=0.1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, d_model, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls_proj = nn.Linear(d_model + feat_dim, 1)

    def forward(self, ids, feats):
        x = self.emb(ids) + self.pos[:, : ids.size(1), :]
        h = self.enc(x)[:, 0]  # CLS token
        cat = torch.cat([h, feats], dim=1)
        return self.cls_proj(cat).squeeze(1)


# ---------- experiment storage ----------
experiment_data = {
    "hybrid_transformer": {
        "metrics": {"train_MCC": [], "val_MCC": [], "train_F1": [], "val_F1": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}

dropouts = [0.1, 0.3]
epochs = 6
best_val_mcc = -1
for dr in dropouts:
    print(f"\n==== Training HybridTransformer dropout={dr} ====")
    model = HybridTransformer(vocab_size, feat_dim, dropout=dr).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    exp = experiment_data["hybrid_transformer"]
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = []
        tr_pred = []
        tr_lbl = []
        for batch in train_loader():
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            logits = model(batch["input_ids"], batch["sym_feats"])
            loss = crit(logits, batch["label"])
            loss.backward()
            opt.step()
            tr_loss.append(loss.item())
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            tr_pred.extend(preds)
            tr_lbl.extend(batch["label"].int().cpu().numpy())
        train_mcc = matthews_corrcoef(tr_lbl, tr_pred)
        train_f1 = f1_score(tr_lbl, tr_pred, average="macro")

        # ---- validate ----
        model.eval()
        val_loss = []
        val_pred = []
        val_lbl = []
        with torch.no_grad():
            for batch in dev_loader():
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"], batch["sym_feats"])
                val_loss.append(crit(logits, batch["label"]).item())
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                val_pred.extend(preds)
                val_lbl.extend(batch["label"].int().cpu().numpy())
        val_mcc = matthews_corrcoef(val_lbl, val_pred)
        val_f1 = f1_score(val_lbl, val_pred, average="macro")
        print(
            f"Epoch {ep}: validation_loss = {np.mean(val_loss):.4f} | val_MCC={val_mcc:.3f}"
        )

        exp["metrics"]["train_MCC"].append(train_mcc)
        exp["metrics"]["val_MCC"].append(val_mcc)
        exp["metrics"]["train_F1"].append(train_f1)
        exp["metrics"]["val_F1"].append(val_f1)
        exp["losses"]["train"].append(np.mean(tr_loss))
        exp["losses"]["val"].append(np.mean(val_loss))
        exp["epochs"].append(ep)

        if val_mcc > best_val_mcc:
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_val_mcc = val_mcc

    # ---- plot losses ----
    plt.figure()
    plt.plot(exp["epochs"], exp["losses"]["train"], label="train")
    plt.plot(exp["epochs"], exp["losses"]["val"], label="val")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss (dropout={dr})")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"loss_dropout_{dr}.png"))
    plt.close()

# ---------- test evaluation on best model ----------
model.load_state_dict(best_model_state)
model.to(device)
model.eval()
tst_pred, tst_lbl = [], []
with torch.no_grad():
    for batch in test_loader():
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["sym_feats"])
        preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        tst_pred.extend(preds)
        tst_lbl.extend(batch["label"].int().cpu().numpy())
test_mcc = matthews_corrcoef(tst_lbl, tst_pred)
test_f1 = f1_score(tst_lbl, tst_pred, average="macro")
print(f"\n=== Test Results: MCC={test_mcc:.3f} | Macro-F1={test_f1:.3f} ===")

experiment_data["hybrid_transformer"]["predictions"] = tst_pred
experiment_data["hybrid_transformer"]["ground_truth"] = tst_lbl

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to working directory")
