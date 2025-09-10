import os, random, pathlib, math, time, json
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ──────────────────── bookkeeping ─────────────────── #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
ABLT = "TokenOnlyTransformer"
DSNAME = "SPR_BENCH"
experiment_data = {
    ABLT: {
        DSNAME: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# ───────────────── reproducibility / device ───────── #
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ───────────── dataset helpers (real or toy) ─────── #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


def maybe_dataset() -> DatasetDict:
    root = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    if root.exists():
        print("Found real SPR_BENCH at", root)
        return load_spr_bench(root)
    print("⚠️  SPR_BENCH not found – generating synthetic data.")
    syms = list("ABCDEFGH")

    def synth(n):
        seqs, labs = [], []
        for i in range(n):
            ln = random.randint(5, 15)
            seq = "".join(random.choice(syms) for _ in range(ln))
            labs.append(int(seq.count("A") % 2 == 0))
            seqs.append(seq)
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    return DatasetDict(
        train=HFDataset.from_dict(synth(3000)),
        dev=HFDataset.from_dict(synth(800)),
        test=HFDataset.from_dict(synth(800)),
    )


spr = maybe_dataset()
print("Split sizes:", {k: len(v) for k, v in spr.items()})

# ─────────────── tokenisation utilities ──────────── #
PAD, CLS = 0, 1
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 2 for i, ch in enumerate(vocab)}
vocab_size = len(stoi) + 2
itos = {0: "<pad>", 1: "<cls>", **{i + 2: c for i, c in enumerate(vocab)}}
max_len = min(48, max(len(s) for s in spr["train"]["sequence"])) + 1  # +1 CLS


def encode_tokens(seq: str):
    ids = [CLS] + [stoi.get(c, PAD) for c in seq][: max_len - 1]
    ids += [PAD] * (max_len - len(ids))
    return ids[:max_len]


# keep encode_counts only to satisfy dataset but it won't be used
def encode_counts(seq: str):
    vec = np.zeros(len(vocab) + 1, dtype=np.float32)
    return vec


# ────────────────── torch Dataset ─────────────────── #
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seq, self.lab, self.ids = hf_ds["sequence"], hf_ds["label"], hf_ds["id"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode_tokens(self.seq[idx]), dtype=torch.long),
            "y": torch.tensor(self.lab[idx], dtype=torch.float32),
            "rid": str(self.ids[idx]),
        }


train_ds, val_ds, test_ds = (
    SPRTorch(spr["train"]),
    SPRTorch(spr["dev"]),
    SPRTorch(spr["test"]),
)


# ───────────── Token-Only Transformer model ────────── #
class TokenOnlyTransformer(nn.Module):
    def __init__(self, vocab_sz, emb=64, nhead=8, nlayers=2, ff=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, max_len, emb))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.classifier = nn.Sequential(
            nn.Linear(emb, emb), nn.ReLU(), nn.Dropout(dropout), nn.Linear(emb, 1)
        )

    def forward(self, tok):
        h = self.emb(tok) + self.pos[:, : tok.size(1), :]
        h = self.transformer(h)
        cls = h[:, 0]
        return self.classifier(cls).squeeze(1)


model = TokenOnlyTransformer(
    vocab_size, emb=96, nhead=8, nlayers=3, ff=256, dropout=0.15
).to(device)


# ─────────────── utilities / evaluation ───────────── #
def rule_macro_accuracy(preds, gts, ids):
    d = {}
    for p, g, i in zip(preds, gts, ids):
        key = str(i).split("-")[0]
        c, t = d.get(key, (0, 0))
        d[key] = (c + int(p == g), t + 1)
    return np.mean([c / t for c, t in d.values()]) if d else 0.0


def evaluate(loader):
    model.eval()
    tot_loss = 0
    logits_all = []
    y_all = []
    id_all = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["rid"]
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logit = model(x)
            loss = criterion(logit, y)
            tot_loss += loss.item() * y.size(0)
            logits_all.append(logit.sigmoid().cpu())
            y_all.append(y.cpu())
            id_all += ids
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    preds = (logits > 0.5).int().numpy()
    y_np = y.int().numpy()
    acc = (preds == y_np).mean()
    mcc = matthews_corrcoef(y_np, preds) if len(np.unique(y_np)) > 1 else 0.0
    rma = rule_macro_accuracy(preds, y_np, id_all)
    return tot_loss / len(loader.dataset), acc, mcc, rma, preds, y_np, id_all


# ─────────────── DataLoaders & training setup ─────── #
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

label_smooth = 0.04


def smooth_labels(y):
    return y * (1 - label_smooth) + 0.5 * label_smooth


criterion = lambda logits, y: nn.BCEWithLogitsLoss()(logits, smooth_labels(y))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
total_steps = len(train_loader) * 8
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, total_steps=total_steps
)

# ───────────────────── training loop ──────────────── #
epochs = 8
for epoch in range(1, epochs + 1):
    model.train()
    tr_loss_sum = 0
    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tr_loss_sum += loss.item() * y.size(0)
    train_loss = tr_loss_sum / len(train_loader.dataset)
    with torch.no_grad():
        p = (out.sigmoid() > 0.5).int().cpu().numpy()
        yb = batch["y"].cpu().int().numpy()
        tr_acc = (p == yb).mean()
        tr_mcc = matthews_corrcoef(yb, p) if len(np.unique(yb)) > 1 else 0.0
        tr_rma = rule_macro_accuracy(p, yb, batch["rid"])
    val_loss, val_acc, val_mcc, val_rma, *_ = evaluate(val_loader)

    ed = experiment_data[ABLT][DSNAME]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append({"acc": tr_acc, "MCC": tr_mcc, "RMA": tr_rma})
    ed["metrics"]["val"].append({"acc": val_acc, "MCC": val_mcc, "RMA": val_rma})
    ed["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | acc={val_acc:.3f} | MCC={val_mcc:.3f} | RMA={val_rma:.3f}"
    )

# ───────────────────── final evaluation ───────────── #
test_loss, test_acc, test_mcc, test_rma, preds, gts, ids = evaluate(test_loader)
print("\n===== TEST RESULTS =====")
print(
    f"loss={test_loss:.4f} | acc={test_acc:.3f} | MCC={test_mcc:.3f} | RMA={test_rma:.3f}"
)

ed = experiment_data[ABLT][DSNAME]
ed["predictions"] = preds.tolist()
ed["ground_truth"] = gts.tolist()
ed["test_metrics"] = {
    "loss": test_loss,
    "acc": test_acc,
    "MCC": test_mcc,
    "RMA": test_rma,
}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
