import os, time, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ----------------------------- dirs / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ----------------------------- experiment dict
experiment_data = {
    "no_transformer_mean_pool": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}


# ----------------------------- helper metrics
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def color_weighted_accuracy(seqs, y, yhat):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if a == b else 0 for wi, a, b in zip(w, y, yhat)) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y, yhat):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if a == b else 0 for wi, a, b in zip(w, y, yhat)) / max(sum(w), 1)


def composite_variety_accuracy(seqs, y, yhat):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi if a == b else 0 for wi, a, b in zip(w, y, yhat)) / max(sum(w), 1)


# ----------------------------- synthetic fallback
def synth_dataset(n_train=5000, n_dev=1000, n_test=1000, n_cls=4):
    def rand_tok():
        return random.choice("ABCD") + random.choice("0123")

    def rand_seq():
        return " ".join(rand_tok() for _ in range(random.randint(4, 12)))

    def lab(s):
        return (count_color_variety(s) + count_shape_variety(s)) % n_cls

    def make(n):
        seqs = [rand_seq() for _ in range(n)]
        return {"sequence": seqs, "label": [lab(s) for s in seqs]}

    d = DatasetDict()
    for split, n in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        d[split] = load_dataset("json", split=[], data=make(n))
    return d


# ----------------------------- data load
try:
    DATA_ROOT = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_ROOT)
    print("Loaded SPR_BENCH")
except Exception as e:
    print("Falling back to synthetic data", e)
    spr = synth_dataset()

num_classes = len(set(spr["train"]["label"]))
print("#classes:", num_classes)

# ----------------------------- vocab
shapes = sorted({tok[0] for seq in spr["train"]["sequence"] for tok in seq.split()})
colors = sorted(
    {tok[1] for seq in spr["train"]["sequence"] for tok in seq.split() if len(tok) > 1}
)
shape2id = {s: i + 1 for i, s in enumerate(shapes)}  # 0 = PAD
color2id = {c: i + 1 for i, c in enumerate(colors)}  # 0 = PAD


def encode(seq):
    s_ids, c_ids = [], []
    for tok in seq.split():
        s_ids.append(shape2id.get(tok[0], 0))
        c_ids.append(color2id.get(tok[1], 0) if len(tok) > 1 else 0)
    return s_ids, c_ids


class SPRDataset(Dataset):
    def __init__(self, sequences, labels, max_len=None):
        enc = [encode(s) for s in sequences]
        self.max_len = max_len or max(len(e[0]) for e in enc)
        self.shapes = [e[0][: self.max_len] for e in enc]
        self.colors = [e[1][: self.max_len] for e in enc]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "shape": torch.tensor(self.shapes[idx], dtype=torch.long),
            "color": torch.tensor(self.colors[idx], dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    maxlen = max(len(b["shape"]) for b in batch)

    def pad(x):
        return torch.cat([x, torch.zeros(maxlen - len(x), dtype=torch.long)])

    shape = torch.stack([pad(b["shape"]) for b in batch])
    color = torch.stack([pad(b["color"]) for b in batch])
    y = torch.stack([b["y"] for b in batch])
    return {"shape": shape, "color": color, "y": y}


train_ds = SPRDataset(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRDataset(spr["dev"]["sequence"], spr["dev"]["label"], train_ds.max_len)
test_ds = SPRDataset(spr["test"]["sequence"], spr["test"]["label"], train_ds.max_len)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate)


# ----------------------------- Mean-Pooling model
class MeanPoolingBaseline(nn.Module):
    def __init__(self, n_shape, n_color, d_model=64, num_cls=2):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(n_color, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(256, d_model)  # still learned but not contextual
        self.cls = nn.Linear(d_model, num_cls)

    def forward(self, s_ids, c_ids):
        B, L = s_ids.shape
        pos = torch.arange(L, device=s_ids.device).unsqueeze(0).expand(B, L)
        x = self.shape_emb(s_ids) + self.color_emb(c_ids) + self.pos_emb(pos)
        mask = s_ids == 0
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        h = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)  # mean over non-pad
        return self.cls(h)


model = MeanPoolingBaseline(
    len(shape2id) + 1, len(color2id) + 1, d_model=64, num_cls=num_classes
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

tag = "no_transformer_mean_pool"
epochs = 12
best_cva, best_state = -1, None


# ----------------------------- evaluation
def evaluate(loader, split):
    model.eval()
    tot_loss, seqs, ys, yh = 0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["shape"], batch["color"])
            loss = criterion(logits, batch["y"])
            tot_loss += loss.item() * batch["y"].size(0)
            preds = logits.argmax(-1).cpu().tolist()
            ys.extend(batch["y"].cpu().tolist())
            yh.extend(preds)
            start = len(seqs)
            if split == "train":
                seqs.extend(spr["train"]["sequence"][start : start + len(preds)])
            elif split == "dev":
                seqs.extend(spr["dev"]["sequence"][start : start + len(preds)])
            else:
                seqs.extend(spr["test"]["sequence"][start : start + len(preds)])
    loss = tot_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(seqs, ys, yh)
    swa = shape_weighted_accuracy(seqs, ys, yh)
    cva = composite_variety_accuracy(seqs, ys, yh)
    return loss, cwa, swa, cva, yh, ys


# ----------------------------- training loop
for ep in range(1, epochs + 1):
    model.train()
    run_loss = 0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["shape"], batch["color"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["y"].size(0)
    tr_loss = run_loss / len(train_loader.dataset)
    experiment_data[tag]["SPR"]["losses"]["train"].append(tr_loss)

    val_loss, cwa, swa, cva, _, _ = evaluate(dev_loader, "dev")
    experiment_data[tag]["SPR"]["losses"]["val"].append(val_loss)
    experiment_data[tag]["SPR"]["metrics"]["val"].append(
        {"cwa": cwa, "swa": swa, "cva": cva}
    )
    experiment_data[tag]["SPR"]["timestamps"].append(time.time())
    print(
        f"Epoch {ep}: val_loss={val_loss:.4f} | CWA={cwa:.4f} | SWA={swa:.4f} | CVA={cva:.4f}"
    )
    if cva > best_cva:
        best_cva, best_state = cva, {k: v.cpu() for k, v in model.state_dict().items()}

# ----------------------------- test
if best_state:
    model.load_state_dict(best_state)
test_loss, cwa, swa, cva, preds, gts = evaluate(test_loader, "test")
print(f"\nTEST: loss={test_loss:.4f} | CWA={cwa:.4f} | SWA={swa:.4f} | CVA={cva:.4f}")
exp = experiment_data[tag]["SPR"]
exp["metrics"]["test"] = {"cwa": cwa, "swa": swa, "cva": cva}
exp["predictions"] = preds
exp["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
