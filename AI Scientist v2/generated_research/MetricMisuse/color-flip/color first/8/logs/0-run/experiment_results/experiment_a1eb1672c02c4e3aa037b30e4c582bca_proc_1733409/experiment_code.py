import os, random, pathlib, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- util metrics ----------
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split()})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(1, sum(w))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(1, sum(w))


def harmonic_mean(a, b, eps=1e-8):
    return 2 * a * b / (a + b + eps)


# ---------- load dataset (real or synthetic) ----------
def load_real(root):
    from datasets import load_dataset

    def _load(name):
        return list(
            load_dataset(
                "csv",
                data_files=str(root / name),
                split="train",
                cache_dir=".cache_dsets",
            )
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


def synth_set(n):
    shapes, colors = "ABCD", "1234"

    def make_seq():
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(8))

    def lbl(seq):
        return max(
            set(t[0] for t in seq.split()), key=[t[0] for t in seq.split()].count
        )

    return [{"id": i, "sequence": (s := make_seq()), "label": lbl(s)} for i in range(n)]


DATA_DIR = pathlib.Path("SPR_BENCH")
try:
    data = load_real(DATA_DIR)
except Exception as e:
    print("Dataset missing, using synthetic data", e)
    data = {"train": synth_set(4000), "dev": synth_set(1000), "test": synth_set(1000)}

# ---------- clustering ----------
all_tokens = [
    tok for row in itertools.chain(*data.values()) for tok in row["sequence"].split()
]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2i = {s: i for i, s in enumerate(shapes)}
color2i = {c: i for i, c in enumerate(colors)}


def tok_vec(t):
    return np.array([shape2i[t[0]], color2i[t[1]]], dtype=np.float32)


token_vecs = np.stack([tok_vec(t) for t in all_tokens])

cand_k = [6, 8, 10, 12, 14]
sil = []
sample = np.random.choice(
    len(token_vecs), size=min(3000, len(token_vecs)), replace=False
)
for k in cand_k:
    km = KMeans(k, n_init=10, random_state=0).fit(token_vecs[sample])
    sil.append(silhouette_score(token_vecs[sample], km.labels_))
best_k = cand_k[int(np.argmax(sil))]
print("Best k:", best_k, "silhouette", max(sil))
kmeans = KMeans(best_k, n_init=20, random_state=1).fit(token_vecs)

train_clusters_present = set(
    kmeans.predict(
        np.stack([tok_vec(t) for row in data["train"] for t in row["sequence"].split()])
    )
)


# ---------- encode sequences ----------
def seq_to_clusters(seq):
    ids = kmeans.predict(np.stack([tok_vec(t) for t in seq.split()]))
    return ids.tolist()


# ---------- dataset ----------
class SPRSeqSet(Dataset):
    def __init__(self, rows, label_encoder):
        self.seqs_raw = [r["sequence"] for r in rows]
        self.clust_seq = [seq_to_clusters(s) for s in self.seqs_raw]
        self.labels = label_encoder.transform([r["label"] for r in rows]).astype(
            np.int64
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seq": self.clust_seq[idx],
            "label": torch.tensor(self.labels[idx]),
            "raw_seq": self.seqs_raw[idx],
        }


le = LabelEncoder()
le.fit([r["label"] for r in data["train"]])
train_ds = SPRSeqSet(data["train"], le)
dev_ds = SPRSeqSet(data["dev"], le)
test_ds = SPRSeqSet(data["test"], le)


def collate(batch):
    seqs = [b["seq"] for b in batch]
    lens = torch.tensor([len(s) for s in seqs])
    max_len = lens.max().item()
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s)
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "x": padded.to(device),
        "len": lens.to(device),
        "y": labels.to(device),
        "raw_seq": raw,
    }


train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_dl = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------- model ----------
class GRUClassifier(nn.Module):
    def __init__(self, num_tokens, num_classes, emb=32, hid=64):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, emb)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x, length):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, length.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[0], h[1]], dim=1)  # bi
        return self.fc(h)


model = GRUClassifier(best_k, len(le.classes_)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# ---------- OCGA ----------
def ocga(raw_seqs, y_t, y_p):
    cnt = tot = 0
    for s, t, p in zip(raw_seqs, y_t, y_p):
        cl = set(kmeans.predict(np.stack([tok_vec(tok) for tok in s.split()])))
        if not cl.issubset(train_clusters_present):
            tot += 1
            cnt += t == p
    return cnt / max(1, tot)


# ---------- experiment data ----------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------- train loop ----------
patience, wait, best_hm = 8, 0, -1
best_state, best_epoch = None, 0
for epoch in range(1, 61):
    # ---- train
    model.train()
    total_loss = 0
    for batch in train_dl:
        opt.zero_grad()
        out = model(batch["x"], batch["len"])
        loss = criterion(out, batch["y"])
        loss.backward()
        opt.step()
        total_loss += loss.item() * batch["y"].size(0)
    tr_loss = total_loss / len(train_ds)
    experiment_data["SPR"]["losses"]["train"].append((epoch, tr_loss))

    # ---- validation
    model.eval()
    v_loss = 0
    preds = []
    gts = []
    raws = []
    with torch.no_grad():
        for batch in dev_dl:
            out = model(batch["x"], batch["len"])
            v_loss += criterion(out, batch["y"]).item() * batch["y"].size(0)
            p = out.argmax(1).cpu().numpy()
            preds.extend(p)
            g = batch["y"].cpu().numpy()
            gts.extend(g)
            raws.extend(batch["raw_seq"])
    v_loss /= len(dev_ds)
    cwa = color_weighted_accuracy(raws, gts, preds)
    swa = shape_weighted_accuracy(raws, gts, preds)
    hm = harmonic_mean(cwa, swa)
    ocg = ocga(raws, gts, preds)
    experiment_data["SPR"]["losses"]["val"].append((epoch, v_loss))
    experiment_data["SPR"]["metrics"]["val"].append((epoch, cwa, swa, hm, ocg))
    print(
        f"Epoch {epoch}: validation_loss = {v_loss:.4f}  CWA={cwa:.3f} SWA={swa:.3f} HM={hm:.3f} OCGA={ocg:.3f}"
    )

    # ---- early stop
    if hm > best_hm + 1e-4:
        best_hm, best_state, best_epoch, wait = (
            hm,
            {k: v.cpu() for k, v in model.state_dict().items()},
            epoch,
            0,
        )
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping at epoch", epoch, "best epoch", best_epoch)
        break

# ---------- test ----------
model.load_state_dict(best_state)
model.to(device)
model.eval()
preds = []
gts = []
raws = []
with torch.no_grad():
    for batch in test_dl:
        out = model(batch["x"], batch["len"])
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["y"].cpu().numpy())
        raws.extend(batch["raw_seq"])
cwa = color_weighted_accuracy(raws, gts, preds)
swa = shape_weighted_accuracy(raws, gts, preds)
ocg = ocga(raws, gts, preds)
print("TEST  CWA={:.3f}  SWA={:.3f}  OCGA={:.3f}".format(cwa, swa, ocg))
experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
