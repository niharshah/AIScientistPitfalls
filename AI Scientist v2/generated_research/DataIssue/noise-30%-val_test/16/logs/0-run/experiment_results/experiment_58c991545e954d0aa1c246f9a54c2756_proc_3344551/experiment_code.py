# No-Label-Smoothing ablation – stand-alone script
import os, random, pathlib, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ─────────────────── bookkeeping dict ─────────────────── #
experiment_data = {
    "no_label_smoothing": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# ───────────────── reproducibility / device ───────────── #
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ──────────────── dataset helpers (real or toy) ───────── #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(csv):  # tiny helper
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {"train": _l("train.csv"), "dev": _l("dev.csv"), "test": _l("test.csv")}
    )


def maybe_dataset() -> DatasetDict:
    root = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    if root.exists():
        print("Found real SPR_BENCH at", root)
        return load_spr_bench(root)
    print("⚠️  SPR_BENCH not found – using synthetic toy data")
    syms = list("ABCDEFGH")

    def synth(n):
        seqs, labs = [], []
        for i in range(n):
            ln = random.randint(5, 15)
            seq = "".join(random.choice(syms) for _ in range(ln))
            labs.append(int(seq.count("A") % 2 == 0))
            seqs.append(seq)
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    dd = DatasetDict()
    for sp, n in [("train", 3000), ("dev", 800), ("test", 800)]:
        dd[sp] = HFDataset.from_dict(synth(n))
    return dd


spr = maybe_dataset()
print({k: len(v) for k, v in spr.items()})

# ───────────────── tokenisation utilities ─────────────── #
PAD, CLS = 0, 1
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 2 for i, ch in enumerate(vocab)}
vocab_size = len(stoi) + 2
itos = {i: ch for ch, i in enumerate(["<pad>", "<cls>"] + vocab)}

max_len = min(48, max(len(s) for s in spr["train"]["sequence"])) + 1  # +1 for CLS


def encode_tokens(seq: str):
    ids = [CLS] + [stoi.get(c, PAD) for c in seq][: max_len - 1]
    ids += [PAD] * (max_len - len(ids))
    return ids


def encode_counts(seq: str):
    vec = np.zeros(len(vocab) + 1, dtype=np.float32)
    for ch in seq:
        if ch in stoi:
            vec[stoi[ch] - 2] += 1.0
    vec[:-1] /= max(len(seq), 1)
    vec[-1] = len(seq) / max_len
    return vec


# ───────────────── torch dataset class ───────────────── #
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seq, self.lab, self.ids = hf_ds["sequence"], hf_ds["label"], hf_ds["id"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode_tokens(self.seq[idx]), dtype=torch.long),
            "feat": torch.tensor(encode_counts(self.seq[idx])),
            "y": torch.tensor(self.lab[idx], dtype=torch.float32),
            "rid": str(self.ids[idx]),
        }


train_ds, val_ds, test_ds = (
    SPRTorch(spr["train"]),
    SPRTorch(spr["dev"]),
    SPRTorch(spr["test"]),
)


# ───────────────── model definition ──────────────────── #
class CountAwareTransformer(nn.Module):
    def __init__(
        self, vocab_sz, emb=64, nhead=8, nlayers=2, ff=128, extra_dim=0, dr=0.1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, max_len, emb))
        enc = nn.TransformerEncoderLayer(emb, nhead, ff, dr, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, nlayers)
        self.feat_proj = nn.Linear(extra_dim, emb)
        self.classifier = nn.Sequential(
            nn.Linear(emb * 2, emb), nn.ReLU(), nn.Dropout(dr), nn.Linear(emb, 1)
        )

    def forward(self, tok, feats):
        h = self.emb(tok) + self.pos[:, : tok.size(1)]
        h = self.transformer(h)
        cls = h[:, 0]
        f = self.feat_proj(feats)
        return self.classifier(torch.cat([cls, f], -1)).squeeze(1)


model = CountAwareTransformer(
    vocab_size, emb=96, nhead=8, nlayers=3, ff=256, extra_dim=len(vocab) + 1, dr=0.15
).to(device)

# ───────────────── dataloaders / loss / opt ──────────── #
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, 256)
test_loader = DataLoader(test_ds, 256)

criterion = nn.BCEWithLogitsLoss()  # <-- ablation: NO LABEL SMOOTHING
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
total_steps = len(train_loader) * 8
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, total_steps=total_steps
)


# ─────────────────── metric helpers ─────────────────── #
def rule_macro_accuracy(preds, gts, ids):
    d = {}
    for p, g, i in zip(preds, gts, ids):
        k = str(i).split("-")[0]
        c, t = d.get(k, (0, 0))
        d[k] = (c + int(p == g), t + 1)
    return np.mean([c / t for c, t in d.values()]) if d else 0.0


@torch.no_grad()
def evaluate(loader):
    model.eval()
    tots, logits_all, y_all, id_all = 0, [], [], []
    for batch in loader:
        ids = batch["rid"]
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logit = model(batch["x"], batch["feat"])
        y = batch["y"]
        tots += criterion(logit, y).item() * y.size(0)
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
    return tots / len(loader.dataset), acc, mcc, rma, preds, y_np, id_all


# ───────────────────── training loop ─────────────────── #
epochs = 8
for ep in range(1, epochs + 1):
    model.train()
    tr_loss = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["x"], batch["feat"])
        loss = criterion(out, batch["y"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tr_loss += loss.item() * batch["y"].size(0)
    train_loss = tr_loss / len(train_loader.dataset)
    with torch.no_grad():
        p = (out.sigmoid() > 0.5).int().cpu().numpy()
        yb = batch["y"].cpu().int().numpy()
        tr_acc = (p == yb).mean()
        tr_mcc = matthews_corrcoef(yb, p) if len(np.unique(yb)) > 1 else 0.0
        tr_rma = rule_macro_accuracy(p, yb, batch["rid"])
    val_loss, val_acc, val_mcc, val_rma, *_ = evaluate(val_loader)
    # log
    ex = experiment_data["no_label_smoothing"]["SPR_BENCH"]
    ex["losses"]["train"].append(train_loss)
    ex["losses"]["val"].append(val_loss)
    ex["metrics"]["train"].append({"acc": tr_acc, "MCC": tr_mcc, "RMA": tr_rma})
    ex["metrics"]["val"].append({"acc": val_acc, "MCC": val_mcc, "RMA": val_rma})
    ex["epochs"].append(ep)
    print(
        f"Epoch {ep}: val_loss={val_loss:.4f} | acc={val_acc:.3f} | MCC={val_mcc:.3f} | RMA={val_rma:.3f}"
    )

# ─────────────────── final test eval ─────────────────── #
test_loss, test_acc, test_mcc, test_rma, preds, gts, ids = evaluate(test_loader)
print("\n==== TEST ====")
print(
    f"loss={test_loss:.4f} | acc={test_acc:.3f} | MCC={test_mcc:.3f} | RMA={test_rma:.3f}"
)

ex = experiment_data["no_label_smoothing"]["SPR_BENCH"]
ex["predictions"] = preds.tolist()
ex["ground_truth"] = gts.tolist()
ex["test_metrics"] = {
    "loss": test_loss,
    "acc": test_acc,
    "MCC": test_mcc,
    "RMA": test_rma,
}

# ─────────────────── save data for plots ─────────────── #
os.makedirs("working", exist_ok=True)
np.save("working/experiment_data.npy", experiment_data)
