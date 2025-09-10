# NoContrastivePretrain – self-contained script
import os, pathlib, time, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# --------- experiment dict ----------
experiment_data = {
    "NoContrastivePretrain": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load SPR -------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)


# ---------- helper metrics ----------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def comp_weight(seq):
    return count_shape_variety(seq) * count_color_variety(seq)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [comp_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ---------- vocab & labels ----------
vocab = {"<pad>": 0, "<unk>": 1}
for s in spr["train"]["sequence"]:
    for tok in s.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
pad_id = vocab["<pad>"]
unk_id = vocab["<unk>"]
labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}
num_classes = len(labels)
print(f"Vocabulary size = {len(vocab)}, num classes = {num_classes}")


# ---------- dataset --------------
class SPRDatasetCLS(Dataset):
    def __init__(self, seqs, labels_):
        self.seqs = seqs
        self.labels = [lab2id[l] for l in labels_]

    def encode(self, toks):
        return [vocab.get(t, unk_id) for t in toks]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        t = self.encode(self.seqs[idx].split())
        return t, self.labels[idx], self.seqs[idx]


def collate_cls(batch):
    seqs, ys, raw = zip(*batch)
    lens = [len(s) for s in seqs]
    mx = max(lens)
    arr = np.full((len(seqs), mx), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return (torch.tensor(arr), torch.tensor(lens), torch.tensor(ys)), list(raw)


# ---------- model ------------------
class Encoder(nn.Module):
    # simple encoder without projection head
    def __init__(self, vocab_sz, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)

    def forward(self, x, lens):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # [2, B, hid]
        h = torch.cat([h[0], h[1]], dim=1)  # B, hid*2
        return h


class Classifier(nn.Module):
    def __init__(self, encoder, nclass):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(256, nclass)  # hid*2 -> nclass

    def forward(self, x, lens):
        z = self.enc(x, lens)
        return self.head(z)


# ---------- training data loaders -------------
train_loader = DataLoader(
    SPRDatasetCLS(spr["train"]["sequence"], spr["train"]["label"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_cls,
)
dev_loader = DataLoader(
    SPRDatasetCLS(spr["dev"]["sequence"], spr["dev"]["label"]),
    batch_size=512,
    shuffle=False,
    collate_fn=collate_cls,
)

# ---------- train from scratch ----------------
model = Classifier(Encoder(len(vocab)).to(device), num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val, wait, patience = 1e9, 0, 3

for epoch in range(1, 21):
    # ----- train -----
    model.train()
    tr_loss = 0
    for (x, lens, y), _ in train_loader:
        x, lens, y = x.to(device), lens.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, lens)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * x.size(0)
    tr_loss /= len(train_loader.dataset)

    # ----- validation -----
    model.eval()
    val_loss = 0
    preds, gt, rawseq = [], [], []
    with torch.no_grad():
        for (x, lens, y), raw in dev_loader:
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            out = model(x, lens)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            gt.extend(y.cpu().tolist())
            rawseq.extend(raw)
    val_loss /= len(dev_loader.dataset)
    cwa = comp_weighted_accuracy(
        rawseq, [id2lab[i] for i in gt], [id2lab[i] for i in preds]
    )

    # log
    ed = experiment_data["NoContrastivePretrain"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["val"].append(cwa)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  CompWA={cwa:.4f}"
    )

    # early stopping
    if val_loss < best_val - 1e-4:
        best_val, wait = val_loss, 0
        ed["predictions"], ed["ground_truth"] = preds, gt
        best_state = model.state_dict()
    else:
        wait += 1
    if wait >= patience:
        break

# ---------- save & finish ------------
model.load_state_dict(best_state)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished. Results saved to ./working/experiment_data.npy")
