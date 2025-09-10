import os, random, pathlib, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# ---------------- mandatory working dir + device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- metric helpers --------------------------------
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


# ---------------- data loading ----------------------------------
def load_real_spr(root):
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


def create_synth(n):
    shapes, colors = "ABCD", "1234"

    def seq():
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(8))

    def lab(seq):
        return max(
            set([t[0] for t in seq.split()]), key=[t[0] for t in seq.split()].count
        )

    return [{"id": i, "sequence": (s := seq()), "label": lab(s)} for i in range(n)]


DATA_DIR = pathlib.Path("SPR_BENCH")
try:
    data = load_real_spr(DATA_DIR)
except Exception as e:
    print("Could not load real data, using synthetic:", e)
    data = {
        "train": create_synth(4000),
        "dev": create_synth(1000),
        "test": create_synth(1000),
    }

# ---------------- clustering over glyph vectors -----------------
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

ks = [6, 8, 10, 12, 14]
sil = []
sample = np.random.choice(
    len(token_vecs), size=min(3000, len(token_vecs)), replace=False
)
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(token_vecs[sample])
    sil.append(silhouette_score(token_vecs[sample], km.labels_))
best_k = ks[int(np.argmax(sil))]
print("Chosen clusters:", best_k)
kmeans = KMeans(best_k, n_init=20, random_state=1).fit(token_vecs)

train_clusters_present = set(
    kmeans.predict(
        np.stack(
            [tok_vec(tok) for row in data["train"] for tok in row["sequence"].split()]
        )
    )
)

# ---------------- dataset ---------------------------------------
PAD_ID = 0


def encode_seq(seq):
    clust = kmeans.predict(np.stack([tok_vec(t) for t in seq.split()]))
    return (clust + 1).tolist()  # reserve 0 for PAD


class SPRSeqSet(Dataset):
    def __init__(self, rows, le):
        self.seqs_text = [r["sequence"] for r in rows]
        self.seqs_ids = [encode_seq(s) for s in self.seqs_text]
        self.labels = le.transform([r["label"] for r in rows])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "ids": torch.tensor(self.seqs_ids[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_text": self.seqs_text[idx],
        }


def collate(batch):
    max_len = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = b["ids"]
    labels = torch.stack([b["label"] for b in batch])
    texts = [b["seq_text"] for b in batch]
    return {"ids": ids.to(device), "label": labels.to(device), "seq_text": texts}


# ---------------- label encoding --------------------------------
le = LabelEncoder()
le.fit([r["label"] for r in data["train"]])
train_ds, dev_ds, test_ds = (
    SPRSeqSet(data["train"], le),
    SPRSeqSet(data["dev"], le),
    SPRSeqSet(data["test"], le),
)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_dl = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------------- model -----------------------------------------
class SeqClassifier(nn.Module):
    def __init__(self, vocab, emb=32, heads=4, layers=2, num_classes=10):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=PAD_ID)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.classifier = nn.Linear(emb, num_classes)

    def forward(self, ids):
        mask = ids == PAD_ID
        x = self.embed(ids)
        x = self.encoder(x, src_key_padding_mask=mask)
        # mean pooling excluding PAD
        lengths = (~mask).sum(1, keepdim=True).clamp(min=1)
        pooled = (x.masked_fill(mask.unsqueeze(-1), 0).sum(1)) / lengths
        return self.classifier(pooled)


model = SeqClassifier(vocab=best_k + 1, num_classes=len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------------- experiment data dict --------------------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------------- OCGA ------------------------------------------
def ocga(seqs, y_t, y_p):
    ok, tot = 0, 0
    for s, t, p in zip(seqs, y_t, y_p):
        cl = set(kmeans.predict(np.stack([tok_vec(tok) for tok in s.split()])))
        if not cl.issubset(train_clusters_present):
            tot += 1
            ok += int(t == p)
    return ok / max(1, tot)


# ---------------- training loop ---------------------------------
best_hm, best_state, best_epoch, wait, patience = -1, None, 0, 0, 6
for epoch in range(1, 31):
    model.train()
    running = 0
    for batch in train_dl:
        optimizer.zero_grad()
        out = model(batch["ids"])
        loss = criterion(out, batch["label"])
        loss.backward()
        optimizer.step()
        running += loss.item() * batch["label"].size(0)
    train_loss = running / len(train_ds)
    experiment_data["SPR"]["losses"]["train"].append((epoch, train_loss))

    # ---------- validation ----------
    model.eval()
    running = 0
    preds = []
    gts = []
    texts = []
    with torch.no_grad():
        for batch in dev_dl:
            out = model(batch["ids"])
            running += criterion(out, batch["label"]).item() * batch["label"].size(0)
            p = out.argmax(1).cpu().numpy()
            g = batch["label"].cpu().numpy()
            preds.extend(p)
            gts.extend(g)
            texts.extend(batch["seq_text"])
    val_loss = running / len(dev_ds)
    cwa = color_weighted_accuracy(texts, gts, preds)
    swa = shape_weighted_accuracy(texts, gts, preds)
    hm = harmonic_mean(cwa, swa)
    oca = ocga(texts, gts, preds)
    experiment_data["SPR"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR"]["metrics"]["val"].append((epoch, cwa, swa, hm, oca))
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HM={hm:.3f} OCGA={oca:.3f}"
    )
    # early stopping
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
        print(f"Early stop at epoch {epoch} (best {best_epoch})")
        break

# ---------------- test evaluation -------------------------------
model.load_state_dict(best_state)
model.to(device)
model.eval()
preds, gts, texts = [], [], []
with torch.no_grad():
    for batch in test_dl:
        out = model(batch["ids"])
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["label"].cpu().numpy())
        texts.extend(batch["seq_text"])
experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts
print(
    "TEST  CWA={:.3f}  SWA={:.3f}  OCGA={:.3f}".format(
        color_weighted_accuracy(texts, gts, preds),
        shape_weighted_accuracy(texts, gts, preds),
        ocga(texts, gts, preds),
    )
)

# ---------------- save -------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved data to", os.path.join(working_dir, "experiment_data.npy"))
