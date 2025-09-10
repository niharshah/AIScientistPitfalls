# Uni-GRU (No Bidirectional Context) ablation for SPR_BENCH
import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, DatasetDict

# ----------------- experiment bookkeeping --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
ABLT = "UniGRU_NoBi"
DATASET = "SPR_BENCH"
experiment_data = {
    ABLT: {
        DATASET: {
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------- helper functions ------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def complexity_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


# ----------------------- dataset path ------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dset = load_spr_bench(DATA_PATH)

train_seqs = dset["train"]["sequence"]
dev_seqs = dset["dev"]["sequence"]
y_train = np.array(dset["train"]["label"], dtype=np.float32)
y_dev = np.array(dset["dev"]["label"], dtype=np.float32)


# -------------------- glyph clustering k=32 ------------------------
def token_list(seqs):
    out = []
    for s in seqs:
        out.extend(s.strip().split())
    return out


all_tokens = token_list(train_seqs)
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


def seq_to_ids(seq):
    ids = []
    for tok in seq.strip().split():
        sid = shape_le.transform([tok[0]])[0]
        cid = color_le.transform([tok[1]])[0]
        cl = kmeans.predict([[sid, cid]])[0] + 1  # +1 for PAD=0
        ids.append(cl)
    return ids


train_ids = [seq_to_ids(s) for s in train_seqs]
dev_ids = [seq_to_ids(s) for s in dev_seqs]
vocab_size = K + 1  # PAD = 0


# ---------------------- torch dataset ------------------------------
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
    return {
        "input_ids": padded,
        "lengths": lens,
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


train_loader = DataLoader(
    SPRDataset(train_ids, y_train), batch_size=256, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(dev_ids, y_dev), batch_size=512, shuffle=False, collate_fn=collate
)


# -------------------------- model ----------------------------------
class UniGRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hid, 1)

    def forward(self, ids, lengths):
        x = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h shape: (1, batch, hid)
        h = h.squeeze(0)
        return self.fc(h).squeeze(-1)


model = UniGRUClassifier(vocab_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------- training loop ---------------------------
EPOCHS = 3
for epoch in range(1, EPOCHS + 1):
    # ----- train -----
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["lengths"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch["labels"].size(0)
    train_loss /= len(train_loader.dataset)
    experiment_data[ABLT][DATASET]["metrics"]["train_loss"].append(train_loss)

    # ----- validation -----
    model.eval()
    val_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
            gts.extend(batch["labels"].cpu().numpy().astype(int))
    val_loss /= len(dev_loader.dataset)
    CWA = color_weighted_accuracy(dev_seqs, gts, preds)
    SWA = shape_weighted_accuracy(dev_seqs, gts, preds)
    CWA2 = complexity_weighted_accuracy(dev_seqs, gts, preds)

    md = experiment_data[ABLT][DATASET]["metrics"]
    md["val_loss"].append(val_loss)
    md["val_CWA"].append(CWA)
    md["val_SWA"].append(SWA)
    md["val_CWA2"].append(CWA2)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
        f"CWA={CWA:.4f} SWA={SWA:.4f} CWA2={CWA2:.4f}"
    )

experiment_data[ABLT][DATASET]["predictions"] = preds
experiment_data[ABLT][DATASET]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment finished and saved to working/experiment_data.npy")
