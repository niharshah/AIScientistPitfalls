# ------------------- No-Clustering Raw Glyph Vocabulary -------------
import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ----------------- boiler-plate workspace ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "no_cluster_raw_vocab": {
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


def count_color_variety(seq):  # second char
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq):  # first char
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


# -------------------- Raw glyph vocabulary -------------------------
def token_list(seqs):
    out = []
    for s in seqs:
        out.extend(s.strip().split())
    return out


train_tokens = set(token_list(train_seqs))
PAD, UNK = 0, 1
token2id = {tok: i + 2 for i, tok in enumerate(sorted(train_tokens))}
vocab_size = len(token2id) + 2  # PAD & UNK


def seq_to_ids(seq):
    ids = []
    for tok in seq.strip().split():
        ids.append(token2id.get(tok, UNK))
    return ids


train_ids = [seq_to_ids(s) for s in train_seqs]
dev_ids = [seq_to_ids(s) for s in dev_seqs]


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
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, ids, lengths):
        x = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h_cat = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(h_cat).squeeze(-1)


model = GRUClassifier(vocab_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------- training loop ---------------------------
EPOCHS = 3
for epoch in range(1, EPOCHS + 1):
    # ---------- train -------------
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
    exp = experiment_data["no_cluster_raw_vocab"]["SPR_BENCH"]
    exp["metrics"]["train_loss"].append(train_loss)

    # ---------- validation --------
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

    exp["metrics"]["val_loss"].append(val_loss)
    exp["metrics"]["val_CWA"].append(CWA)
    exp["metrics"]["val_SWA"].append(SWA)
    exp["metrics"]["val_CWA2"].append(CWA2)

    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | CWA={CWA:.4f} SWA={SWA:.4f} CWA2={CWA2:.4f}"
    )

exp["predictions"] = preds
exp["ground_truth"] = gts

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment finished and saved to working/experiment_data.npy")
