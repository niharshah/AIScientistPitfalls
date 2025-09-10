import os, pathlib, random, time, math, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ---------------------------------------------------------------------------#
# experiment bookkeeping & reproducibility
# ---------------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},  # will store dicts with acc/MCC/RMA
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

# ---------------------------------------------------------------------------#
# device
# ---------------------------------------------------------------------------#
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
    d["train"] = _l("train.csv")
    d["dev"] = _l("dev.csv")
    d["test"] = _l("test.csv")
    return d


def maybe_load_dataset() -> DatasetDict:
    root = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    if root.exists():
        print("Loading real SPR_BENCH from", root)
        return load_spr_bench(root)
    print("Real dataset not found - falling back to synthetic toy data.")
    syms = list("ABCDEFGH")

    def synth_split(n):
        seqs, labs = [], []
        for idx in range(n):
            ln = random.randint(5, 12)
            seq = "".join(random.choice(syms) for _ in range(ln))
            lab = int(seq.count("A") % 2 == 0)
            seqs.append(seq)
            labs.append(lab)
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    dd = DatasetDict()
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        dd[split] = HFDataset.from_dict(synth_split(n))
    return dd


spr = maybe_load_dataset()
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# ---------------------------------------------------------------------------#
# vocabulary & encoding
# ---------------------------------------------------------------------------#
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
PAD = 0
CLS = 1
stoi = {ch: i + 2 for i, ch in enumerate(vocab)}  # 0 PAD, 1 CLS, rest symbols
itos = {i: ch for ch, i in stoi.items()}
max_len = min(40, max(len(s) for s in spr["train"]["sequence"])) + 1  # +1 for CLS


def encode(seq: str):
    ids = [CLS] + [stoi.get(c, PAD) for c in seq[: max_len - 1]]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    return ids[:max_len]


# ---------------------------------------------------------------------------#
# torch Dataset
# ---------------------------------------------------------------------------#
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seq = hf_ds["sequence"]
        self.lab = hf_ds["label"]
        self.ids = hf_ds["id"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "y": torch.tensor(self.lab[idx], dtype=torch.float32),
            "rid": str(self.ids[idx]),
        }


train_ds = SPRTorch(spr["train"])
val_ds = SPRTorch(spr["dev"])
test_ds = SPRTorch(spr["test"])


# ---------------------------------------------------------------------------#
# model: tiny Transformer encoder
# ---------------------------------------------------------------------------#
class TinyTransformer(nn.Module):
    def __init__(self, vocab_sz, emb=64, nhead=8, nlayers=2, ff=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, max_len, emb))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.cls_head = nn.Linear(emb, 1)

    def forward(self, x):
        z = self.emb(x) + self.pos[:, : x.size(1), :]
        h = self.transformer(z)
        cls_h = h[:, 0]  # token 0 is CLS
        return self.cls_head(cls_h).squeeze(1)


model = TinyTransformer(vocab_sz=len(stoi) + 2).to(device)


# ---------------------------------------------------------------------------#
# training utilities
# ---------------------------------------------------------------------------#
def rule_macro_accuracy(preds, gts, ids):
    rule_stats = {}
    for p, g, i in zip(preds, gts, ids):
        rule_id = str(i).split("-")[0]  # heuristic extraction
        correct, total = rule_stats.get(rule_id, (0, 0))
        rule_stats[rule_id] = (correct + int(p == g), total + 1)
    return np.mean([c / t for c, t in rule_stats.values()])


def evaluate(dloader):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    all_logits, all_y, all_ids = [], [], []
    loss_total = 0
    with torch.no_grad():
        for batch in dloader:
            ids = batch["rid"]
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = crit(logits, batch["y"])
            loss_total += loss.item() * batch["x"].size(0)
            all_logits.append(logits.cpu())
            all_y.append(batch["y"].cpu())
            all_ids += ids
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    preds = (torch.sigmoid(logits) > 0.5).int().numpy()
    y_np = y.int().numpy()
    acc = (preds == y_np).mean()
    mcc = matthews_corrcoef(y_np, preds) if len(np.unique(y_np)) > 1 else 0.0
    rma = rule_macro_accuracy(preds, y_np, all_ids)
    return loss_total / len(dloader.dataset), acc, mcc, rma, preds, y_np, all_ids


# ---------------------------------------------------------------------------#
# train loop
# ---------------------------------------------------------------------------#
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 6
for epoch in range(1, epochs + 1):
    model.train()
    t0 = time.time()
    loss_sum = 0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * batch["x"].size(0)
    train_loss = loss_sum / len(train_loader.dataset)
    # metrics on train (quick)
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        tr_acc = (preds == batch["y"].cpu().int().numpy()).mean()
        tr_mcc = matthews_corrcoef(batch["y"].cpu().int().numpy(), preds)
        tr_rma = rule_macro_accuracy(
            preds, batch["y"].cpu().int().numpy(), batch["rid"]
        )
    # validation
    val_loss, val_acc, val_mcc, val_rma, *_ = evaluate(val_loader)

    # record
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
        f"Epoch {epoch}: "
        f"val_loss = {val_loss:.4f} | val_acc = {val_acc:.3f} | "
        f"val_MCC = {val_mcc:.3f} | val_RMA = {val_rma:.3f} "
        f"({time.time()-t0:.1f}s)"
    )

# ---------------------------------------------------------------------------#
# final test evaluation
# ---------------------------------------------------------------------------#
test_loss, test_acc, test_mcc, test_rma, preds, gts, ids = evaluate(test_loader)
print("\n=== Test results ===")
print(
    f"loss: {test_loss:.4f} | acc: {test_acc:.3f} | "
    f"MCC: {test_mcc:.3f} | RMA: {test_rma:.3f}"
)

experiment_data["SPR_BENCH"]["predictions"] = preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = gts.tolist()
experiment_data["SPR_BENCH"]["test_metrics"] = {
    "loss": test_loss,
    "acc": test_acc,
    "MCC": test_mcc,
    "RMA": test_rma,
}

# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
