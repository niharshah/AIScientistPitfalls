import os, pathlib, random, time, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ---------------------------------------------------------------------------#
# basic setup & bookkeeping
# ---------------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

rng_seed = 2024
random.seed(rng_seed)
np.random.seed(rng_seed)
torch.manual_seed(rng_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------#
# dataset helpers
# ---------------------------------------------------------------------------#
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _l(f"{sp}.csv")
    return d


def maybe_dataset() -> DatasetDict:
    root = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    if root.exists():
        print("Loading real SPR_BENCH …")
        return load_spr_bench(root)
    print("Fallback: synthetic toy data")
    syms = list("ABCDEFGH")

    def synth(n):
        seqs, labs = [], []
        for _ in range(n):
            ln = random.randint(5, 12)
            s = "".join(random.choice(syms) for _ in range(ln))
            labs.append(int(s.count("A") % 2 == 0))
            seqs.append(s)
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    d = DatasetDict()
    d["train"] = HFDataset.from_dict(synth(2000))
    d["dev"] = HFDataset.from_dict(synth(500))
    d["test"] = HFDataset.from_dict(synth(500))
    return d


spr = maybe_dataset()
print({k: len(v) for k, v in spr.items()})

# ---------------------------------------------------------------------------#
# vocabulary & encoding
# ---------------------------------------------------------------------------#
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
PAD, CLS = 0, 1
stoi = {ch: i + 2 for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_sz = len(stoi) + 2
max_len = min(40, max(len(s) for s in spr["train"]["sequence"])) + 1  # +CLS


def encode_seq(seq: str):
    ids = [CLS] + [stoi.get(c, PAD) for c in seq[: max_len - 1]]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    return ids[:max_len]


def count_vec(seq: str):
    vec = np.zeros(len(vocab), dtype=np.float32)
    for c in seq:
        if c in stoi:
            vec[stoi[c] - 2] += 1
    if len(seq) > 0:
        vec /= len(seq)
    return vec


# ---------------------------------------------------------------------------#
# torch Dataset
# ---------------------------------------------------------------------------#
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.sq = hf_ds["sequence"]
        self.lab = hf_ds["label"]
        self.ids = hf_ds["id"]

    def __len__(self):
        return len(self.sq)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode_seq(self.sq[idx]), dtype=torch.long),
            "cnt": torch.tensor(count_vec(self.sq[idx]), dtype=torch.float32),
            "y": torch.tensor(self.lab[idx], dtype=torch.float32),
            "rid": str(self.ids[idx]),
        }


train_ds, val_ds, test_ds = (
    SPRTorch(spr["train"]),
    SPRTorch(spr["dev"]),
    SPRTorch(spr["test"]),
)


# ---------------------------------------------------------------------------#
# hybrid model
# ---------------------------------------------------------------------------#
class HybridTransformer(nn.Module):
    def __init__(self, vocab_sz, emb=128, heads=8, layers=3, ff=256, dropout=0.1):
        super().__init__()
        self.emb_tok = nn.Embedding(vocab_sz, emb, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, max_len, emb))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
        )
        self.trf = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cnt_proj = nn.Sequential(
            nn.Linear(len(vocab), emb), nn.ReLU(), nn.Dropout(dropout)
        )
        self.out = nn.Linear(emb * 2, 1)

    def forward(self, x, cnt):
        z = self.emb_tok(x) + self.pos[:, : x.size(1), :]
        h = self.trf(z)[:, 0]  # CLS token
        c = self.cnt_proj(cnt)
        return self.out(torch.cat([h, c], dim=-1)).squeeze(1)


model = HybridTransformer(vocab_sz=vocab_sz).to(device)


# ---------------------------------------------------------------------------#
# Rule-Macro Accuracy
# ---------------------------------------------------------------------------#
def rule_macro_accuracy(preds, gts, ids):
    bucket = {}
    for p, g, i in zip(preds, gts, ids):
        r = i.split("-")[0]
        corr, tot = bucket.get(r, (0, 0))
        bucket[r] = (corr + int(p == g), tot + 1)
    return np.mean([c / t for c, t in bucket.values()])


# ---------------------------------------------------------------------------#
# train / eval loops
# ---------------------------------------------------------------------------#
def evaluate(dl):
    model.eval()
    all_logits, all_y, all_ids = [], [], []
    loss_tot = 0
    with torch.no_grad():
        for batch in dl:
            ids = batch["rid"]
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logit = model(batch["x"], batch["cnt"])
            loss = criterion(logit, batch["y"])
            loss_tot += loss.item() * batch["x"].size(0)
            all_logits.append(logit.cpu())
            all_y.append(batch["y"].cpu())
            all_ids += ids
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    preds = (torch.sigmoid(logits) > 0.5).int().numpy()
    y_np = y.int().numpy()
    acc = (preds == y_np).mean()
    mcc = matthews_corrcoef(y_np, preds) if len(np.unique(y_np)) > 1 else 0.0
    rma = rule_macro_accuracy(preds, y_np, all_ids)
    return loss_tot / len(dl.dataset), acc, mcc, rma, preds, y_np


batch_size = 128
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=256)
test_dl = DataLoader(test_ds, batch_size=256)

# label smoothing BCE
smooth_eps = 0.1


def smooth_targets(y):
    return y * (1 - smooth_eps) + 0.5 * smooth_eps


criterion = lambda logits, targets: nn.BCEWithLogitsLoss()(
    logits, smooth_targets(targets)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

epochs = 8
for epoch in range(1, epochs + 1):
    model.train()
    t0 = time.time()
    loss_sum = 0
    for batch in train_dl:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["x"], batch["cnt"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * batch["x"].size(0)
    scheduler.step()
    train_loss = loss_sum / len(train_dl.dataset)
    # quick last-batch metrics
    with torch.no_grad():
        pb = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        ty = batch["y"].cpu().int().numpy()
        tr_acc = (pb == ty).mean()
        tr_mcc = matthews_corrcoef(ty, pb)
        tr_rma = rule_macro_accuracy(pb, ty, batch["rid"])
    val_loss, val_acc, val_mcc, val_rma, *_ = evaluate(val_dl)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"acc": tr_acc, "MCC": tr_mcc, "RMA": tr_rma}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "MCC": val_mcc, "RMA": val_rma}
    )
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  val_MCC={val_mcc:.3f}  val_RMA={val_rma:.3f}  ({time.time()-t0:.1f}s)"
    )

# ---------------------------------------------------------------------------#
# final evaluation
# ---------------------------------------------------------------------------#
test_loss, test_acc, test_mcc, test_rma, preds, gts = evaluate(test_dl)
print("\n=== Test ===")
print(
    f"loss {test_loss:.4f} | acc {test_acc:.3f} | MCC {test_mcc:.3f} | RMA {test_rma:.3f}"
)

experiment_data["SPR_BENCH"]["predictions"] = preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = gts.tolist()
experiment_data["SPR_BENCH"]["test_metrics"] = {
    "loss": test_loss,
    "acc": test_acc,
    "MCC": test_mcc,
    "RMA": test_rma,
}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
