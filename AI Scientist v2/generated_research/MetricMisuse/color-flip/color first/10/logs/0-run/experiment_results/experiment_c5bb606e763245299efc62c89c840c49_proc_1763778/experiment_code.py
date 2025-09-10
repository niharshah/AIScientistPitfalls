import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, DatasetDict

# ----------------- experiment bookkeeping --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "Factorized_SC": {
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
train_seqs, dev_seqs = dset["train"]["sequence"], dset["dev"]["sequence"]
y_train = np.array(dset["train"]["label"], dtype=np.float32)
y_dev = np.array(dset["dev"]["label"], dtype=np.float32)


# -------------------- factorized ID preparation --------------------
def token_list(seqs):
    out = []
    for s in seqs:
        out.extend(s.strip().split())
    return out


all_tokens = token_list(train_seqs)
shape_le = LabelEncoder().fit([t[0] for t in all_tokens])
color_le = LabelEncoder().fit([t[1] for t in all_tokens])
PAD = 0  # reserve 0 for padding


def seq_to_shape_color_ids(seq):
    s_ids, c_ids = [], []
    for tok in seq.strip().split():
        s_ids.append(shape_le.transform([tok[0]])[0] + 1)  # shift by 1
        c_ids.append(color_le.transform([tok[1]])[0] + 1)
    return s_ids, c_ids


train_shape_ids, train_color_ids = [], []
for s in train_seqs:
    s_ids, c_ids = seq_to_shape_color_ids(s)
    train_shape_ids.append(s_ids)
    train_color_ids.append(c_ids)

dev_shape_ids, dev_color_ids = [], []
for s in dev_seqs:
    s_ids, c_ids = seq_to_shape_color_ids(s)
    dev_shape_ids.append(s_ids)
    dev_color_ids.append(c_ids)

shape_vocab = len(shape_le.classes_) + 1  # +1 for PAD=0
color_vocab = len(color_le.classes_) + 1


# ---------------------- torch dataset ------------------------------
class SPRDataset(torch.utils.data.Dataset):
    def __init__(self, shape_lists, color_lists, labels):
        self.shape_lists, self.color_lists, self.labels = (
            shape_lists,
            color_lists,
            labels,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.shape_lists[idx], self.color_lists[idx], self.labels[idx]


def collate(batch):
    s_lists, c_lists, labels = zip(*batch)
    lens = torch.tensor([len(x) for x in s_lists], dtype=torch.long)
    maxlen = lens.max().item()
    shape_pad = torch.zeros(len(batch), maxlen, dtype=torch.long)
    color_pad = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, (s_ids, c_ids) in enumerate(zip(s_lists, c_lists)):
        L = len(s_ids)
        shape_pad[i, :L] = torch.tensor(s_ids, dtype=torch.long)
        color_pad[i, :L] = torch.tensor(c_ids, dtype=torch.long)
    return {
        "shape_ids": shape_pad,
        "color_ids": color_pad,
        "lengths": lens,
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


train_loader = DataLoader(
    SPRDataset(train_shape_ids, train_color_ids, y_train),
    batch_size=256,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRDataset(dev_shape_ids, dev_color_ids, y_dev),
    batch_size=512,
    shuffle=False,
    collate_fn=collate,
)


# -------------------------- model ----------------------------------
class GRUClassifierFactorized(nn.Module):
    def __init__(self, shp_vocab, col_vocab, shp_dim=8, col_dim=8, hid=128):
        super().__init__()
        self.shape_emb = nn.Embedding(shp_vocab, shp_dim, padding_idx=0)
        self.color_emb = nn.Embedding(col_vocab, col_dim, padding_idx=0)
        emb_dim = shp_dim + col_dim
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, s_ids, c_ids, lengths):
        s_e = self.shape_emb(s_ids)
        c_e = self.color_emb(c_ids)
        x = torch.cat([s_e, c_e], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h_cat = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(h_cat).squeeze(-1)


model = GRUClassifierFactorized(shape_vocab, color_vocab).to(device)
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
        logits = model(batch["shape_ids"], batch["color_ids"], batch["lengths"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch["labels"].size(0)
    train_loss /= len(train_loader.dataset)
    experiment_data["Factorized_SC"]["SPR_BENCH"]["metrics"]["train_loss"].append(
        train_loss
    )

    # ----- validation -----
    model.eval()
    val_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["shape_ids"], batch["color_ids"], batch["lengths"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
            gts.extend(batch["labels"].cpu().numpy().astype(int))
    val_loss /= len(dev_loader.dataset)
    CWA = color_weighted_accuracy(dev_seqs, gts, preds)
    SWA = shape_weighted_accuracy(dev_seqs, gts, preds)
    CWA2 = complexity_weighted_accuracy(dev_seqs, gts, preds)
    md = experiment_data["Factorized_SC"]["SPR_BENCH"]["metrics"]
    md["val_loss"].append(val_loss)
    md["val_CWA"].append(CWA)
    md["val_SWA"].append(SWA)
    md["val_CWA2"].append(CWA2)

    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | CWA={CWA:.4f} SWA={SWA:.4f} CWA2={CWA2:.4f}"
    )

experiment_data["Factorized_SC"]["SPR_BENCH"]["predictions"] = preds
experiment_data["Factorized_SC"]["SPR_BENCH"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment finished and saved to working/experiment_data.npy")
