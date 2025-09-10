# Orderless Sequence (Token Order Shuffled) – single-file script
import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, DatasetDict

# ------------------- experiment bookkeeping ------------------------
experiment_data = {
    "Orderless": {
        "SPR_BENCH": {
            "metrics": {
                "train_loss": [],
                "val_loss": [],
                "val_CWA": [],
                "val_SWA": [],
                "val_CWA2": [],
            },
            "predictions": [],
            "ground_truth": [],
        }
    }
}
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# -------------------------- helpers --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):  # small helper
        return load_dataset(
            "csv", data_files=str(root / fname), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_color_variety(seq):  # for metrics
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def complexity_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


# --------------------------- data ----------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dset = load_spr_bench(DATA_PATH)
train_seqs, dev_seqs = dset["train"]["sequence"], dset["dev"]["sequence"]
y_train = np.asarray(dset["train"]["label"], dtype=np.float32)
y_dev = np.asarray(dset["dev"]["label"], dtype=np.float32)

# --------- shuffle each sequence once to destroy order -------------
rng = np.random.RandomState(0)


def shuffle_tokens(seq: str) -> str:
    toks = seq.strip().split()
    rng.shuffle(toks)
    return " ".join(toks)


train_seqs_shuf = [shuffle_tokens(s) for s in train_seqs]
dev_seqs_shuf = [shuffle_tokens(s) for s in dev_seqs]  # shuffle val too


# ---------------- glyph clustering (k=32) --------------------------
def token_list(seqs):
    out = []
    for s in seqs:
        out.extend(s.split())
    return out


all_tokens = token_list(train_seqs_shuf)
shape_le = LabelEncoder().fit([t[0] for t in all_tokens])
color_le = LabelEncoder().fit([t[1] for t in all_tokens])

token_vecs = np.stack(
    [
        shape_le.transform([t[0] for t in all_tokens]),
        color_le.transform([t[1] for t in all_tokens]),
    ],
    axis=1,
)
K = 32
kmeans = KMeans(n_clusters=K, random_state=0, n_init=10).fit(token_vecs)


def seq_to_ids(seq: str):
    ids = []
    for tok in seq.split():
        sid = shape_le.transform([tok[0]])[0]
        cid = color_le.transform([tok[1]])[0]
        ids.append(kmeans.predict([[sid, cid]])[0] + 1)  # +1 for PAD
    return ids


train_ids = [seq_to_ids(s) for s in train_seqs_shuf]
dev_ids = [seq_to_ids(s) for s in dev_seqs_shuf]
vocab_size = K + 1  # PAD=0


# ----------------------- torch dataset -----------------------------
class SPRDataset(torch.utils.data.Dataset):
    def __init__(self, id_lists, labels):
        self.id_lists, self.labels = id_lists, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.id_lists[idx], self.labels[idx]


def collate(batch):
    seqs, labels = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    maxlen = lens.max().item()
    padded = torch.zeros(len(seqs), maxlen, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return {"input_ids": padded, "lengths": lens, "labels": torch.tensor(labels)}


train_loader = DataLoader(
    SPRDataset(train_ids, y_train), batch_size=256, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(dev_ids, y_dev), batch_size=512, shuffle=False, collate_fn=collate
)


# --------------------------- model ---------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, ids, lens):
        x = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h_cat = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(h_cat).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = GRUClassifier(vocab_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------- training --------------------------------
EPOCHS = 3
for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["lengths"])
        loss = criterion(logits.float(), batch["labels"].float().to(device))
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * batch["labels"].size(0)
    tr_loss /= len(train_loader.dataset)
    experiment_data["Orderless"]["SPR_BENCH"]["metrics"]["train_loss"].append(tr_loss)

    # ---- val ----
    model.eval()
    val_loss = 0.0
    preds = []
    gts = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"])
            loss = criterion(logits.float(), batch["labels"].float().to(device))
            val_loss += loss.item() * batch["labels"].size(0)
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
            gts.extend(batch["labels"].cpu().numpy().astype(int))
    val_loss /= len(dev_loader.dataset)
    CWA = color_weighted_accuracy(dev_seqs_shuf, gts, preds)
    SWA = shape_weighted_accuracy(dev_seqs_shuf, gts, preds)
    CWA2 = complexity_weighted_accuracy(dev_seqs_shuf, gts, preds)

    md = experiment_data["Orderless"]["SPR_BENCH"]["metrics"]
    md["val_loss"].append(val_loss)
    md["val_CWA"].append(CWA)
    md["val_SWA"].append(SWA)
    md["val_CWA2"].append(CWA2)
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} CWA={CWA:.4f} "
        f"SWA={SWA:.4f} CWA2={CWA2:.4f}"
    )

# store final preds/labels
experiment_data["Orderless"]["SPR_BENCH"]["predictions"] = preds
experiment_data["Orderless"]["SPR_BENCH"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment finished and saved to working/experiment_data.npy")
