import os, random, math, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, accuracy_score

# ------------------------------------------------------------------- paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------- device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------- experiment recorder
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {
            "train_MCC": [],
            "val_MCC": [],
            "train_RMA": [],
            "val_RMA": [],
            "test_MCC": None,
            "test_RMA": None,
        },
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "cfg": {},
    }
}

# ---------------------------------------------------------------- reproducibility
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# ---------------------------------------------------------------- load dataset
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


data_path = pathlib.Path(
    os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
if not data_path.exists():
    raise RuntimeError(f"SPR_BENCH not found at {data_path}")
ds = load_spr_bench(data_path)

# ---------------------------------------------------------------- vocabulary / encoding
all_chars = sorted({c for s in ds["train"]["sequence"] for c in s})
stoi = {c: i + 1 for i, c in enumerate(all_chars)}  # 0 = PAD
pad_id = 0
max_len = min(60, max(len(s) for s in ds["train"]["sequence"]))


def encode(seq: str):
    ids = [stoi.get(c, pad_id) for c in seq[:max_len]]
    ids += [pad_id] * (max_len - len(ids))
    return ids


# ---------------------------------------------------------------- torch dataset
class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.X = [encode(s) for s in hf_split["sequence"]]
        self.y = hf_split["label"]
        self.rules = hf_split["rule"] if "rule" in hf_split.column_names else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            "x": torch.tensor(self.X[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }
        if self.rules is not None:
            sample["rule"] = self.rules[idx]
        return sample


train_ds, dev_ds, test_ds = (SPRTorch(ds[s]) for s in ["train", "dev", "test"])


def collate(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    out = {"x": xs, "y": ys}
    if "rule" in batch[0]:
        out["rule"] = [b["rule"] for b in batch]
    return out


# ---------------------------------------------------------------- model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # x: (B,L,D)
        return x + self.pe[:, : x.size(1)]


class SPRTransformer(nn.Module):
    def __init__(self, vocab, d_model=64, nhead=4, nlayers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        mask = x == pad_id
        h = self.emb(x)
        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        h = h.mean(1)
        return self.head(h).squeeze(1)


# ---------------------------------------------------------------- metrics
def rule_macro_accuracy(preds, gts, rules=None):
    if rules is None:
        return accuracy_score(gts, preds)
    groups = {}
    for p, g, r in zip(preds, gts, rules):
        groups.setdefault(r, {"p": [], "g": []})
        groups[r]["p"].append(p)
        groups[r]["g"].append(g)
    accs = [accuracy_score(d["g"], d["p"]) for d in groups.values()]
    return float(np.mean(accs))


# ---------------------------------------------------------------- training routine (bug-fixed)
def train_cfg(hidden_size, epochs=8, lr=5e-4, batch_size=128):
    model = SPRTransformer(len(all_chars), d_model=hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * (len(train_ds) // batch_size)
    )
    criterion = nn.BCEWithLogitsLoss()

    tr_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    va_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

    best_val_mcc, best_state = -1, None

    for epoch in range(1, epochs + 1):
        # ----------------- train -----------------
        model.train()
        tot_loss, preds, gts, tr_rules = 0, [], [], []
        for batch in tr_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            scheduler.step()

            tot_loss += loss.item() * batch["x"].size(0)
            batch_pred = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(
                int
            )
            preds.extend(batch_pred)
            gts.extend(batch["y"].cpu().numpy())
            if "rule" in batch:
                tr_rules.extend(batch["rule"])

        train_loss = tot_loss / len(train_ds)
        train_mcc = matthews_corrcoef(gts, preds)
        train_rma = rule_macro_accuracy(preds, gts, tr_rules if tr_rules else None)

        # ----------------- validation -----------------
        model.eval()
        val_loss, v_preds, v_gts, v_rules = 0, [], [], []
        with torch.no_grad():
            for batch in va_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["x"])
                loss = criterion(logits, batch["y"])
                val_loss += loss.item() * batch["x"].size(0)

                batch_pred = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)
                v_preds.extend(batch_pred)
                v_gts.extend(batch["y"].cpu().numpy())
                if "rule" in batch:
                    v_rules.extend(batch["rule"])

        val_loss /= len(dev_ds)
        val_mcc = matthews_corrcoef(v_gts, v_preds)
        val_rma = rule_macro_accuracy(v_preds, v_gts, v_rules if v_rules else None)

        # ----------- bookkeeping -----------
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train_MCC"].append(train_mcc)
        experiment_data["SPR_BENCH"]["metrics"]["val_MCC"].append(val_mcc)
        experiment_data["SPR_BENCH"]["metrics"]["train_RMA"].append(train_rma)
        experiment_data["SPR_BENCH"]["metrics"]["val_RMA"].append(val_rma)
        experiment_data["SPR_BENCH"]["epochs"].append(epoch)

        print(
            f"Epoch {epoch}: val_loss={val_loss:.4f}  val_MCC={val_mcc:.3f}  val_RMA={val_rma:.3f}"
        )

        if val_mcc > best_val_mcc:
            best_val_mcc, best_state = val_mcc, model.state_dict()

    return best_state, best_val_mcc


# ---------------------------------------------------------------- hyper-parameter sweep
hidden_choices = [64, 128]
best_hidden, best_state, best_val = None, None, -1
for h in hidden_choices:
    print(f"\n=== Training model with d_model={h} ===")
    state, val = train_cfg(h)
    if val > best_val:
        best_val, best_hidden, best_state = val, h, state
print(f"\nBest model uses d_model={best_hidden}  (dev MCC={best_val:.4f})")
experiment_data["SPR_BENCH"]["cfg"]["best_hidden"] = best_hidden

# ---------------------------------------------------------------- test evaluation
test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate)
model = SPRTransformer(len(all_chars), d_model=best_hidden).to(device)
model.load_state_dict(best_state)
model.eval()

test_preds, test_gts, test_rules = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["x"])
        batch_pred = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)
        test_preds.extend(batch_pred)
        test_gts.extend(batch["y"].cpu().numpy())
        if "rule" in batch:
            test_rules.extend(batch["rule"])

test_mcc = matthews_corrcoef(test_gts, test_preds)
test_rma = rule_macro_accuracy(test_preds, test_gts, test_rules if test_rules else None)
print(f"Test MCC={test_mcc:.4f}  Test RMA={test_rma:.4f}")

experiment_data["SPR_BENCH"]["metrics"]["test_MCC"] = test_mcc
experiment_data["SPR_BENCH"]["metrics"]["test_RMA"] = test_rma
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts

# ---------------------------------------------------------------- save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
