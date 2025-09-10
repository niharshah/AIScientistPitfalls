# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, time, random, pathlib, math, copy, numpy as np
from collections import defaultdict
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# ===== 1.  HELPERS (metrics + IO) =================================
def count_color_variety(seq):  # token = ShapeChar + ColorChar
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_acc(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_acc(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def complexity_weighted_acc(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ------------------------------------------------------------------
# ===== 2.  DATA  ===================================================
def load_spr(root):
    def _l(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


def make_synth(n, shapes=6, colors=5, max_len=8, num_labels=4, seed=123):
    rng = random.Random(seed)
    data = {"id": [], "sequence": [], "label": []}
    for i in range(n):
        L = rng.randint(3, max_len)
        seq = " ".join(
            chr(ord("A") + rng.randint(0, shapes - 1)) + str(rng.randint(0, colors - 1))
            for _ in range(L)
        )
        data["id"].append(str(i))
        data["sequence"].append(seq)
        data["label"].append(rng.randint(0, num_labels - 1))
    return data


def get_dataset():
    root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    try:
        if not root.exists():
            raise FileNotFoundError
        print("Using real SPR_BENCH")
        return load_spr(root)
    except Exception:
        print("SPR_BENCH not found – generating synthetic.")
        train = load_dataset(
            "json", data_files={"train": [make_synth(4000)]}, split="train"
        )
        dev = load_dataset(
            "json", data_files={"train": [make_synth(800, seed=999)]}, split="train"
        )
        test = load_dataset(
            "json", data_files={"train": [make_synth(1000, seed=555)]}, split="train"
        )
        return DatasetDict(train=train, dev=dev, test=test)


dset = get_dataset()
num_classes = len(set(dset["train"]["label"]))
print(f"Num classes={num_classes}")

# ------------------------------------------------------------------
# ===== 3.  TOKEN VOCAB & GLYPH CLUSTERING ==========================
all_tokens = set(tok for seq in dset["train"]["sequence"] for tok in seq.split())
vocab = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}  # 0=pad
vocab_size = len(vocab) + 1
print(f"Vocab size={vocab_size}")

# --- build one-hot glyph matrix for clustering
onehots = np.eye(vocab_size - 1, dtype=np.float32)  # index order matches sorted tokens
# simple autoencoder to get dense rep
ae_dim = 4
ae = nn.Sequential(
    nn.Linear(vocab_size - 1, ae_dim), nn.Tanh(), nn.Linear(ae_dim, vocab_size - 1)
)
optim_ae = torch.optim.Adam(ae.parameters(), lr=1e-2)
criterion = nn.MSELoss()
onehots_t = torch.tensor(onehots)
for epoch in range(200):
    optim_ae.zero_grad()
    out = ae(onehots_t)
    loss = criterion(out, onehots_t)
    loss.backward()
    optim_ae.step()
    if epoch % 50 == 0:
        print(f"AE epoch {epoch} loss {loss.item():.4f}")
with torch.no_grad():
    latents = ae[:2](onehots_t).numpy()  # 4-dim latent

# ---- KMeans
K = 8
km = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(latents)
cluster_ids = km.labels_  # len = vocab_size-1
cluster_map = {tok: cluster_ids[i] for i, tok in enumerate(sorted(all_tokens))}
print("Cluster counts:", np.bincount(cluster_ids))


# ------------------------------------------------------------------
# ===== 4.  TORCH DATA WRAPPER ======================================
def encode_seq(seq):
    ids = [vocab.get(tok, 0) for tok in seq.split()]
    clust = [cluster_map.get(tok, 0) for tok in seq.split()]
    return ids, clust


class SPRTorch(Dataset):
    def __init__(self, split):
        enc = [encode_seq(s) for s in split["sequence"]]
        self.ids = [e[0] for e in enc]
        self.clust = [e[1] for e in enc]
        self.labels = split["label"]
        self.raw = split["sequence"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "tok": torch.tensor(self.ids[idx], dtype=torch.long),
            "clu": torch.tensor(self.clust[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    maxlen = max(len(b["tok"]) for b in batch)
    pad = torch.zeros((len(batch), maxlen), dtype=torch.long)
    padc = torch.zeros((len(batch), maxlen), dtype=torch.long)
    lens = []
    labs = []
    for i, b in enumerate(batch):
        L = len(b["tok"])
        pad[i, :L] = b["tok"]
        padc[i, :L] = b["clu"]
        lens.append(L)
        labs.append(b["label"])
    return {
        "tok": pad,
        "clu": padc,
        "len": torch.tensor(lens),
        "label": torch.stack(labs),
    }


train_ds = SPRTorch(dset["train"])
dev_ds = SPRTorch(dset["dev"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate)


# ------------------------------------------------------------------
# ===== 5.  MODEL ====================================================
class ClusterAwareClassifier(nn.Module):
    def __init__(self, vocab_sz, cluster_k, emb=32, hid=64, classes=3):
        super().__init__()
        self.emb_tok = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.emb_clu = nn.Embedding(cluster_k, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hid * 2, 128), nn.ReLU(), nn.Linear(128, classes)
        )

    def forward(self, tok, clu, len_):
        x = self.emb_tok(tok) + self.emb_clu(clu)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, len_.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h)


model = ClusterAwareClassifier(vocab_size, K, classes=num_classes).to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------------
# ===== 6.  EXPERIMENT DATA STORE ===================================
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ------------------------------------------------------------------
# ===== 7.  TRAIN / EVAL ============================================
def eval_loader(loader, raw_seqs):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            b = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
            logits = model(b["tok"], b["clu"], b["len"])
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(b["label"].cpu().tolist())
    cwa = color_weighted_acc(raw_seqs, labels, preds)
    swa = shape_weighted_acc(raw_seqs, labels, preds)
    cpx = complexity_weighted_acc(raw_seqs, labels, preds)
    return preds, labels, (cwa, swa, cpx)


MAX_EPOCHS = 25
patience = 5
best_val = -1
stall = 0
for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    model.train()
    total = 0
    n = 0
    for b in train_loader:
        b = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()
        }
        optimizer.zero_grad()
        out = model(b["tok"], b["clu"], b["len"])
        loss = criterion_cls(out, b["label"])
        loss.backward()
        optimizer.step()
        total += loss.item() * b["label"].size(0)
        n += b["label"].size(0)
    train_loss = total / n
    train_preds, train_labels, (tr_cwa, tr_swa, tr_cpx) = eval_loader(
        train_loader, train_ds.raw
    )
    val_preds, val_labels, (v_cwa, v_swa, v_cpx) = eval_loader(dev_loader, dev_ds.raw)

    ed = experiment_data["SPR_BENCH"]
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(train_loss)
    ed["metrics"]["train"].append({"cwa": tr_cwa, "swa": tr_swa, "cpx": tr_cpx})
    ed["metrics"]["val"].append({"cwa": v_cwa, "swa": v_swa, "cpx": v_cpx})

    print(
        f"Epoch {epoch:02d} loss={train_loss:.4f}  ValCpx={v_cpx:.4f}  (CWA {v_cwa:.3f} SWA {v_swa:.3f})  time {time.time()-t0:.1f}s"
    )

    if v_cpx > best_val + 1e-6:
        best_val = v_cpx
        stall = 0
        ed["predictions"] = val_preds
        ed["ground_truth"] = val_labels
    else:
        stall += 1
        if stall >= patience:
            print("Early stop.")
            break

# ------------------------------------------------------------------
# ===== 8.  SAVE / PLOT =============================================
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
val_cpx = [m["cpx"] for m in experiment_data["SPR_BENCH"]["metrics"]["val"]]
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["epochs"], val_cpx, marker="o")
plt.title("Val Complexity-Weighted Accuracy")
plt.xlabel("Epoch")
plt.ylabel("CpxWA")
plt.savefig(os.path.join(working_dir, "val_cpxwa.png"))
print("Finished; artefacts saved in ./working")
