import os, pathlib, random, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# ------------------------------ working dir / device ------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------------------------- metrics -------------------------------------- #
def count_color_variety(sequence):
    return len(set(tok[1] for tok in sequence.split()))


def count_shape_variety(sequence):
    return len(set(tok[0] for tok in sequence.split()))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    tot = max(sum(w), 1)
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / tot


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    tot = max(sum(w), 1)
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / tot


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    tot = max(sum(w), 1)
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / tot


# --------------------------------- data loading ----------------------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr(root):
    if root.exists():

        def _ld(csv_name):
            return load_dataset(
                "csv",
                data_files=str(root / csv_name),
                split="train",
                cache_dir=".cache_dsets",
            )

        return DatasetDict({sp: _ld(f"{sp}.csv") for sp in ["train", "dev", "test"]})
    shapes, colors = list("ABCD"), list("1234")

    def gen(n):
        rows = []
        for i in range(n):
            ln = random.randint(3, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(ln)
            )
            rows.append({"id": i, "sequence": seq, "label": random.randint(0, 3)})
        return rows

    d = DatasetDict()
    for sp, n in [("train", 600), ("dev", 150), ("test", 150)]:
        tmp = os.path.join(working_dir, f"{sp}.jsonl")
        with open(tmp, "w") as f:
            for r in gen(n):
                f.write(json.dumps(r) + "\n")
        d[sp] = load_dataset("json", data_files=tmp, split="train")
    return d


spr = load_spr(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))

# --------------------------- glyph vocab & k-means (unused) ------------------------ #
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}
color2id = {c: i + 1 for i, c in enumerate(colors)}


# -------------------------------- dataset / collate ------------------------------- #
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seq = spr[split]["sequence"]
        self.lab = spr[split]["label"]

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        tokens = self.seq[idx].split()
        return {
            "shape": [shape2id[t[0]] for t in tokens],
            "color": [color2id[t[1]] for t in tokens],
            "label": self.lab[idx],
            "seq_str": self.seq[idx],
        }


def collate(batch):
    maxlen = max(len(b["shape"]) for b in batch)

    def pad(key):
        return torch.tensor(
            [b[key] + [0] * (maxlen - len(b[key])) for b in batch], dtype=torch.long
        )

    shape = pad("shape")
    color = pad("color")
    mask = (shape != 0).float()
    return {
        "shape": shape,
        "color": color,
        "mask": mask,
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "seqs": [b["seq_str"] for b in batch],
    }


batch_size = 128
train_loader = DataLoader(
    SPRTorch("train"), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch("dev"), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch("test"), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# --------------------------------- model (No-Cluster) ----------------------------- #
class BiLSTMClassifierNoCluster(nn.Module):
    def __init__(self, n_shape, n_color, num_classes, emb_dim=32, hidden=64, dropp=0.2):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, emb_dim, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim * 2,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropp)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropp),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, sh, co, mask):
        x = torch.cat([self.shape_emb(sh), self.color_emb(co)], dim=-1)
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, enforce_sorted=False, batch_first=True
        )
        output, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        masked = unpacked * mask.unsqueeze(-1)
        pooled = masked.sum(1) / mask.sum(1, keepdim=True)
        out = self.dropout(pooled)
        return self.fc(out)


model = BiLSTMClassifierNoCluster(len(shapes), len(colors), num_classes).to(device)

# -------------------------------- training setup ---------------------------------- #
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10
experiment_data = {
    "NoClusterEmbedding": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# -------------------------------- evaluate helper --------------------------------- #
def evaluate(net, loader):
    net.eval()
    all_preds, all_tgts, all_seqs = [], [], []
    loss_total = 0.0
    with torch.no_grad():
        for batch in loader:
            bt = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = net(bt["shape"], bt["color"], bt["mask"])
            loss = criterion(logits, bt["labels"])
            loss_total += loss.item() * bt["labels"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_tgts.extend(bt["labels"].cpu().tolist())
            all_seqs.extend(batch["seqs"])
    avg_loss = loss_total / len(loader.dataset)
    metrics = {
        "CWA": color_weighted_accuracy(all_seqs, all_tgts, all_preds),
        "SWA": shape_weighted_accuracy(all_seqs, all_tgts, all_preds),
        "GCWA": glyph_complexity_weighted_accuracy(all_seqs, all_tgts, all_preds),
    }
    return avg_loss, metrics, all_preds, all_tgts


# -------------------------------- training loop ----------------------------------- #
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        bt = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(bt["shape"], bt["color"], bt["mask"])
        loss = criterion(logits, bt["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * bt["labels"].size(0)
    train_loss = running_loss / len(train_loader.dataset)
    val_loss, val_metrics, _, _ = evaluate(model, dev_loader)
    experiment_data["NoClusterEmbedding"]["SPR_BENCH"]["losses"]["train"].append(
        train_loss
    )
    experiment_data["NoClusterEmbedding"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["NoClusterEmbedding"]["SPR_BENCH"]["metrics"]["train"].append({})
    experiment_data["NoClusterEmbedding"]["SPR_BENCH"]["metrics"]["val"].append(
        val_metrics
    )
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | CWA={val_metrics['CWA']:.3f} "
        f"| SWA={val_metrics['SWA']:.3f} | GCWA={val_metrics['GCWA']:.3f}"
    )

# -------------------------------- final test -------------------------------------- #
test_loss, test_metrics, test_preds, test_tgts = evaluate(model, test_loader)
d = experiment_data["NoClusterEmbedding"]["SPR_BENCH"]
d["losses"]["test"] = test_loss
d["metrics"]["test"] = test_metrics
d["predictions"] = test_preds
d["ground_truth"] = test_tgts
print(
    f"Test: loss={test_loss:.4f} | CWA={test_metrics['CWA']:.3f} | "
    f"SWA={test_metrics['SWA']:.3f} | GCWA={test_metrics['GCWA']:.3f}"
)

# -------------------------------- save -------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
