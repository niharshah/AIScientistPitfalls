# Raw-Count Concat (No Feature-Projection) – self-contained script
import os, random, pathlib, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ────────────────── bookkeeping / reproducibility / device ───────────────── #
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

experiment_data = {
    "RawCount_NoProj": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}


# ───────────────────── dataset helpers (real or synthetic) ───────────────── #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(csv):  # load split csv into hf dataset
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _l(f"{s}.csv")
    return d


def maybe_dataset() -> DatasetDict:
    root = pathlib.Path(os.getenv("SPR_PATH", "./SPR_BENCH/"))
    if root.exists():
        print("Found SPR_BENCH at", root)
        return load_spr_bench(root)
    print("⚠️  SPR_BENCH not found – using synthetic toy data.")
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
    for split, n in [("train", 3000), ("dev", 800), ("test", 800)]:
        dd[split] = HFDataset.from_dict(synth(n))
    return dd


spr = maybe_dataset()
print("Split sizes:", {k: len(v) for k, v in spr.items()})

# ─────────────────────────── tokenisation utils ──────────────────────────── #
PAD, CLS = 0, 1
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 2 for i, ch in enumerate(vocab)}  # reserve 0/1
itos = {i: ch for ch, i in enumerate(["<pad>", "<cls>"] + vocab)}
vocab_size = len(stoi) + 2
max_len = min(48, max(len(s) for s in spr["train"]["sequence"])) + 1  # include CLS


def encode_tokens(seq: str):
    ids = [CLS] + [stoi.get(c, PAD) for c in seq][: max_len - 1]
    ids += [PAD] * (max_len - len(ids))
    return ids[:max_len]


def encode_counts(seq: str):
    vec = np.zeros(len(vocab) + 1, dtype=np.float32)  # +1 for length feature
    for ch in seq:
        if ch in stoi:
            vec[stoi[ch] - 2] += 1.0
    vec[:-1] /= max(len(seq), 1)  # normalised counts
    vec[-1] = len(seq) / max_len  # length fraction
    return vec


# ───────────────────────────── torch Dataset ─────────────────────────────── #
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


# ───────────────────────── hybrid model (no feat_proj) ───────────────────── #
class RawCountConcatTransformer(nn.Module):
    def __init__(
        self, vocab_sz, emb=64, nhead=8, nlayers=2, ff=128, extra_dim=0, dropout=0.1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, max_len, emb))
        enc = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=nlayers)
        # classifier now expects emb + extra_dim
        self.classifier = nn.Sequential(
            nn.Linear(emb + extra_dim, emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb, 1),
        )

    def forward(self, tok, feats):
        h = self.emb(tok) + self.pos[:, : tok.size(1), :]
        h = self.transformer(h)
        cls = h[:, 0]
        cat = torch.cat([cls, feats], dim=-1)
        return self.classifier(cat).squeeze(1)


model = RawCountConcatTransformer(
    vocab_size,
    emb=96,
    nhead=8,
    nlayers=3,
    ff=256,
    extra_dim=len(vocab) + 1,
    dropout=0.15,
).to(device)


# ───────────────────────── training / evaluation utils ───────────────────── #
def rule_macro_accuracy(preds, gts, ids):
    bucket = {}
    for p, g, i in zip(preds, gts, ids):
        k = str(i).split("-")[0]
        c, t = bucket.get(k, (0, 0))
        bucket[k] = (c + int(p == g), t + 1)
    return np.mean([c / t for c, t in bucket.values()]) if bucket else 0.0


def evaluate(loader):
    model.eval()
    tot_loss = 0
    logits_all, y_all, id_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["rid"]
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"], batch["feat"])
            y = batch["y"]
            loss = criterion(logits, y)
            tot_loss += loss.item() * y.size(0)
            logits_all.append(logits.sigmoid().cpu())
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


# ───────────────────────────── data loaders ──────────────────────────────── #
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# ─────────────────────── loss, optimiser, scheduler ─────────────────────── #
label_smooth = 0.04
criterion = lambda logits, y: nn.BCEWithLogitsLoss()(
    logits, y * (1 - label_smooth) + 0.5 * label_smooth
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, total_steps=len(train_loader) * 8
)

# ───────────────────────────────── training ──────────────────────────────── #
epochs = 8
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
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
        epoch_loss += loss.item() * batch["y"].size(0)
    train_loss = epoch_loss / len(train_loader.dataset)
    with torch.no_grad():
        p = (out.sigmoid() > 0.5).int().cpu().numpy()
        yb = batch["y"].cpu().int().numpy()
        tr_acc = (p == yb).mean()
        tr_mcc = matthews_corrcoef(yb, p) if len(np.unique(yb)) > 1 else 0.0
        tr_rma = rule_macro_accuracy(p, yb, batch["rid"])
    val_loss, val_acc, val_mcc, val_rma, *_ = evaluate(val_loader)
    # log
    exp = experiment_data["RawCount_NoProj"]["SPR_BENCH"]
    exp["losses"]["train"].append(train_loss)
    exp["losses"]["val"].append(val_loss)
    exp["metrics"]["train"].append({"acc": tr_acc, "MCC": tr_mcc, "RMA": tr_rma})
    exp["metrics"]["val"].append({"acc": val_acc, "MCC": val_mcc, "RMA": val_rma})
    exp["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} acc={val_acc:.3f} "
        f"MCC={val_mcc:.3f} RMA={val_rma:.3f}"
    )

# ───────────────────────────── final test ───────────────────────────────── #
test_loss, test_acc, test_mcc, test_rma, preds, gts, ids = evaluate(test_loader)
print("\n===== TEST RESULTS =====")
print(
    f"loss={test_loss:.4f} | acc={test_acc:.3f} | MCC={test_mcc:.3f} | RMA={test_rma:.3f}"
)

exp = experiment_data["RawCount_NoProj"]["SPR_BENCH"]
exp["predictions"] = preds.tolist()
exp["ground_truth"] = gts.tolist()
exp["test_metrics"] = {
    "loss": test_loss,
    "acc": test_acc,
    "MCC": test_mcc,
    "RMA": test_rma,
}

# ───────────────────────────── save data ─────────────────────────────────── #
np.save("experiment_data.npy", experiment_data)
print("Saved experiment data to experiment_data.npy")
