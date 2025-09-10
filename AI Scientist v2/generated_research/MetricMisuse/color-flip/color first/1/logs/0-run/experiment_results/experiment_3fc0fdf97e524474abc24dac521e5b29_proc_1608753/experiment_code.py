import os, time, pathlib, random, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# ------------------------------------------------------------------- paths / working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------- gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------- experiment data dict
experiment_data = {}


# ------------------------------------------------------------------- SPR loading helpers
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


def count_color_variety(seq: str) -> int:
    return len(set(t[1] for t in seq.strip().split() if len(t) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(t[0] for t in seq.strip().split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def composite_variety_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ------------------------------------------------------------------- fallback synthetic data
def synth_dataset(n_train=4000, n_dev=800, n_test=800, n_classes=4):
    def rand_seq():
        toks = [
            random.choice("ABCD") + random.choice("0123")
            for _ in range(random.randint(4, 10))
        ]
        return " ".join(toks)

    def lab(s):
        return (count_color_variety(s) + count_shape_variety(s)) % n_classes

    def make(n):
        seqs = [rand_seq() for _ in range(n)]
        return {"sequence": seqs, "label": [lab(s) for s in seqs]}

    d = DatasetDict()
    d["train"] = load_dataset("json", split=[], data=make(n_train))
    d["dev"] = load_dataset("json", split=[], data=make(n_dev))
    d["test"] = load_dataset("json", split=[], data=make(n_test))
    return d


# ------------------------------------------------------------------- obtain dataset
try:
    DATA_ROOT = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_ROOT)
    print("Loaded official SPR_BENCH dataset.")
except Exception as e:
    print("Official dataset not found, creating synthetic one.")
    spr = synth_dataset()

num_classes = len(set(spr["train"]["label"]))
print(f"#Classes = {num_classes}")

# ------------------------------------------------------------------- glyph vocabulary
train_glyphs = [g for seq in spr["train"]["sequence"] for g in seq.strip().split()]
glyph_set = sorted(set(train_glyphs))
glyph2idx = {g: i for i, g in enumerate(glyph_set)}

# ------------------------------------------------------------------- KMeans clustering of glyphs
glyph_features = np.array(
    [
        [
            ord(g[0]) - 65 if len(g) > 1 else 0,
            int(g[1]) if len(g) > 1 and g[1].isdigit() else 0,
        ]
        for g in glyph_set
    ],
    dtype=np.float32,
)
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(glyph_features)
cluster_ids = kmeans.labels_
glyph2cluster = {
    g: int(c) + 1 for g, c in zip(glyph_set, cluster_ids)
}  # +1 leave 0 for PAD
vocab_cluster_size = n_clusters + 1  # plus PAD


# ------------------------------------------------------------------- dataset class
class ClusterSeqDS(Dataset):
    def __init__(self, sequences, labels, max_len=None):
        tokenised = [
            [glyph2cluster.get(tok, 0) for tok in seq.strip().split()]
            for seq in sequences
        ]
        self.max_len = max_len or max(len(t) for t in tokenised)
        self.seqs = [t[: self.max_len] for t in tokenised]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr = torch.tensor(self.seqs[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"x": arr, "y": y}


def collate(batch):
    maxlen = max(len(sample["x"]) for sample in batch)
    xs = torch.stack(
        [
            torch.cat(
                [sample["x"], torch.zeros(maxlen - len(sample["x"]), dtype=torch.long)]
            )
            for sample in batch
        ]
    )
    ys = torch.stack([sample["y"] for sample in batch])
    return {"x": xs, "y": ys}


train_ds = ClusterSeqDS(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = ClusterSeqDS(
    spr["dev"]["sequence"], spr["dev"]["label"], max_len=train_ds.max_len
)
test_ds = ClusterSeqDS(
    spr["test"]["sequence"], spr["test"]["label"], max_len=train_ds.max_len
)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate)


# ------------------------------------------------------------------- model
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(h)


model = BiGRUClassifier(vocab_cluster_size, 32, 64, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# must create optimizer AFTER moving model to device (done)

tag = "cluster_biGRU_k8"
experiment_data[tag] = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------------------------------------------------------------- training
epochs = 20
best_cva = -1
best_state = None


def evaluate(loader, split_name):
    model.eval()
    total_loss, seqs_out, ys_out, preds_out = 0, [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            total_loss += loss.item() * batch["y"].size(0)
            preds = logits.argmax(-1).cpu().numpy()
            ys = batch["y"].cpu().numpy()
            # recover original sequences slice
            if split_name == "dev":
                seq_slice = spr["dev"]["sequence"][
                    i * loader.batch_size : i * loader.batch_size + len(ys)
                ]
            elif split_name == "test":
                seq_slice = spr["test"]["sequence"][
                    i * loader.batch_size : i * loader.batch_size + len(ys)
                ]
            else:
                seq_slice = spr["train"]["sequence"][
                    i * loader.batch_size : i * loader.batch_size + len(ys)
                ]
            seqs_out.extend(seq_slice)
            ys_out.extend(ys.tolist())
            preds_out.extend(preds.tolist())
    avg_loss = total_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(seqs_out, ys_out, preds_out)
    swa = shape_weighted_accuracy(seqs_out, ys_out, preds_out)
    cva = composite_variety_accuracy(seqs_out, ys_out, preds_out)
    return avg_loss, cwa, swa, cva, preds_out, ys_out


for epoch in range(1, epochs + 1):
    model.train()
    running = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        running += loss.item() * batch["y"].size(0)
    train_loss = running / len(train_loader.dataset)
    experiment_data[tag]["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # dev evaluation
    val_loss, cwa, swa, cva, preds_dev, ys_dev = evaluate(dev_loader, "dev")
    experiment_data[tag]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data[tag]["SPR_BENCH"]["metrics"]["val"].append(
        {"cwa": cwa, "swa": swa, "cva": cva}
    )
    experiment_data[tag]["SPR_BENCH"]["timestamps"].append(time.time())
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA={cwa:.4f} | SWA={swa:.4f} | CVA={cva:.4f}"
    )

    if cva > best_cva:
        best_cva = cva
        best_state = model.state_dict()

# ------------------------------------------------------------------- test evaluation using best model
if best_state:
    model.load_state_dict(best_state)
test_loss, cwa, swa, cva, preds_test, ys_test = evaluate(test_loader, "test")
print(f"\nTEST: loss={test_loss:.4f} | CWA={cwa:.4f} | SWA={swa:.4f} | CVA={cva:.4f}")

ed = experiment_data[tag]["SPR_BENCH"]
ed["predictions"] = preds_test
ed["ground_truth"] = ys_test
ed["metrics"]["test"] = {"cwa": cwa, "swa": swa, "cva": cva}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Experiment data saved to {os.path.join(working_dir,"experiment_data.npy")}')
