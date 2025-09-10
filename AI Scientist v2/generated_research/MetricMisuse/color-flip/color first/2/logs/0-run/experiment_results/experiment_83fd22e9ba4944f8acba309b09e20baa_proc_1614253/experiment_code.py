# ---------------------------- Sum-Fusion Ablation : single-file script ---------------------------- #
import os, pathlib, random, json, numpy as np, torch, warnings
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# -------------------------------- working / device -------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ------------------------------ metric helpers ------------------------------------ #
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ------------------------------ data loading ------------------------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr(root: pathlib.Path):
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
            for row in gen(n):
                f.write(json.dumps(row) + "\n")
        d[sp] = load_dataset("json", data_files=tmp, split="train")
    return d


spr = load_spr(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))

# ------------------------------ glyph clustering ------------------------------- #
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}
color2id = {c: i + 1 for i, c in enumerate(colors)}

token_set = sorted(set(all_tokens))
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float
)
n_clusters = min(max(6, len(token_vecs) // 2), 40)
print(f"Clustering {len(token_vecs)} glyphs into {n_clusters} clusters")
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(token_vecs)
tok2cluster = {tok: int(cl) + 1 for tok, cl in zip(token_set, kmeans.labels_)}


# ------------------------------ dataset / loader ------------------------------- #
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seq = spr[split]["sequence"]
        self.lab = spr[split]["label"]

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        toks = self.seq[idx].split()
        return {
            "shape": [shape2id[t[0]] for t in toks],
            "color": [color2id[t[1]] for t in toks],
            "cluster": [tok2cluster[t] for t in toks],
            "label": self.lab[idx],
            "seq_str": self.seq[idx],
        }


def collate(batch):
    maxlen = max(len(b["shape"]) for b in batch)

    def pad(key):
        return torch.tensor(
            [b[key] + [0] * (maxlen - len(b[key])) for b in batch], dtype=torch.long
        )

    out = {
        "shape": pad("shape"),
        "color": pad("color"),
        "cluster": pad("cluster"),
        "mask": (pad("shape") != 0).float(),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "seqs": [b["seq_str"] for b in batch],
    }
    return out


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


# ------------------------------ Sum-Fusion BiLSTM ------------------------------ #
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        n_shape,
        n_color,
        n_cluster,
        num_classes,
        emb_dim=32,
        hidden=64,
        dropp=0.2,
        fusion="sum",
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, emb_dim, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, emb_dim, padding_idx=0)
        self.clus_emb = nn.Embedding(n_cluster + 1, emb_dim, padding_idx=0)
        self.fusion = fusion
        input_size = emb_dim if fusion == "sum" else emb_dim * 3
        self.lstm = nn.LSTM(
            input_size=input_size,
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

    def forward(self, sh, co, cl, mask):
        if self.fusion == "sum":
            x = self.shape_emb(sh) + self.color_emb(co) + self.clus_emb(cl)
        else:
            x = torch.cat(
                [self.shape_emb(sh), self.color_emb(co), self.clus_emb(cl)], dim=-1
            )
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        masked = unpacked * mask.unsqueeze(-1)
        pooled = masked.sum(1) / mask.sum(1, keepdim=True)
        return self.fc(self.dropout(pooled))


# ------------------------------ training setup ------------------------------- #
model = BiLSTMClassifier(
    len(shapes), len(colors), n_clusters, num_classes, fusion="sum"
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

experiment_data = {
    "SumFusion": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# ------------------------------ helper eval ---------------------------------- #
def evaluate(net, loader):
    net.eval()
    all_preds, all_tgts, all_seqs = [], [], []
    loss_total = 0.0
    with torch.no_grad():
        for batch in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = net(bt["shape"], bt["color"], bt["cluster"], bt["mask"])
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


# ------------------------------ training loop -------------------------------- #
for ep in range(1, epochs + 1):
    model.train()
    running = 0.0
    for batch in train_loader:
        bt = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(bt["shape"], bt["color"], bt["cluster"], bt["mask"])
        loss = criterion(logits, bt["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item() * bt["labels"].size(0)
    train_loss = running / len(train_loader.dataset)
    val_loss, val_metrics, _, _ = evaluate(model, dev_loader)
    experiment_data["SumFusion"]["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SumFusion"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SumFusion"]["SPR_BENCH"]["metrics"]["train"].append({})
    experiment_data["SumFusion"]["SPR_BENCH"]["metrics"]["val"].append(val_metrics)
    print(
        f"Epoch {ep}: val_loss={val_loss:.4f} | CWA={val_metrics['CWA']:.3f} | "
        f"SWA={val_metrics['SWA']:.3f} | GCWA={val_metrics['GCWA']:.3f}"
    )

# ------------------------------ final test ----------------------------------- #
test_loss, test_metrics, test_preds, test_tgts = evaluate(model, test_loader)
ed = experiment_data["SumFusion"]["SPR_BENCH"]
ed["losses"]["test"] = test_loss
ed["metrics"]["test"] = test_metrics
ed["predictions"] = test_preds
ed["ground_truth"] = test_tgts
print(
    f"Test: loss={test_loss:.4f} | CWA={test_metrics['CWA']:.3f} | "
    f"SWA={test_metrics['SWA']:.3f} | GCWA={test_metrics['GCWA']:.3f}"
)

# ------------------------------ save ----------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
