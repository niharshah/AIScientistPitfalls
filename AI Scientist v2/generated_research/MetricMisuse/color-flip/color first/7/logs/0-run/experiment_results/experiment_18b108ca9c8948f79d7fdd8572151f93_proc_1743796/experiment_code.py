import os, time, random, pathlib, math, copy, numpy as np
from collections import defaultdict
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------
# 1.  Metric helpers
def count_color_variety(seq):
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


# -------------------------------------------------------------
# 2.  Load dataset (fallback to synthetic if missing)
def load_spr(root: pathlib.Path):
    def _l(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


def make_synth(n, shapes=6, colors=5, max_len=8, num_labels=4, seed=0):
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
    return load_dataset("json", data_files={"train": [data]}, split="train")


root_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dset = load_spr(root_path)
    print("Loaded real SPR_BENCH")
except Exception:
    print("Using synthetic dataset")
    dset = DatasetDict(
        train=make_synth(6000),
        dev=make_synth(1200, seed=1),
        test=make_synth(1500, seed=2),
    )

num_classes = len(set(dset["train"]["label"]))
print("Num classes:", num_classes)

# -------------------------------------------------------------
# 3.  Build vocab + cluster glyphs
all_tokens = sorted({tok for seq in dset["train"]["sequence"] for tok in seq.split()})
vocab = {tok: i + 2 for i, tok in enumerate(all_tokens)}  # 0 PAD, 1 CLS
PAD_ID, CLS_ID = 0, 1
vocab_size = len(vocab) + 2
print("Vocab size", vocab_size)

# Auto-encoder for glyph latent space
onehots = np.eye(len(all_tokens), dtype=np.float32)
ae = nn.Sequential(
    nn.Linear(len(all_tokens), 8), nn.Tanh(), nn.Linear(8, len(all_tokens))
)
ae.to(device)
opt_ae = torch.optim.Adam(ae.parameters(), lr=1e-2)
crit = nn.MSELoss()
x = torch.tensor(onehots, device=device)
for ep in range(150):
    opt_ae.zero_grad()
    out = ae(x)
    loss = crit(out, x)
    loss.backward()
    opt_ae.step()
    if ep % 50 == 0:
        print(f"AE {ep} loss {loss.item():.4f}")
with torch.no_grad():
    lat = ae[:2](x).cpu().numpy()

K = 10
km = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(lat)
cluster_ids = km.labels_
cluster_map = {tok: cluster_ids[i] for i, tok in enumerate(all_tokens)}
print("Cluster distribution:", np.bincount(cluster_ids))


# -------------------------------------------------------------
# 4.  Torch dataset
def encode(seq):
    toks = seq.split()
    ids = [CLS_ID] + [vocab[t] for t in toks]
    clus = [K] + [cluster_map[t] for t in toks]  # treat CLS as its own cluster id = K
    return ids, clus, len(ids)


class SPRTorch(Dataset):
    def __init__(self, split):
        enc = [encode(s) for s in split["sequence"]]
        self.ids = [e[0] for e in enc]
        self.clu = [e[1] for e in enc]
        self.len = [e[2] for e in enc]
        self.labels = split["label"]
        self.raw = split["sequence"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "tok": torch.tensor(self.ids[idx], dtype=torch.long),
            "clu": torch.tensor(self.clu[idx], dtype=torch.long),
            "len": self.len[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    maxlen = max(b["tok"].size(0) for b in batch)
    tok = torch.full((len(batch), maxlen), PAD_ID, dtype=torch.long)
    clu = torch.full((len(batch), maxlen), K, dtype=torch.long)
    mask = torch.zeros(len(batch), maxlen, dtype=torch.bool)
    labels = []
    raw = []
    for i, b in enumerate(batch):
        l = b["tok"].size(0)
        tok[i, :l] = b["tok"]
        clu[i, :l] = b["clu"]
        mask[i, :l] = 1
        labels.append(b["label"])
        raw.append(b.get("raw", ""))
    return {
        "tok": tok,
        "clu": clu,
        "mask": mask,
        "label": torch.stack(labels),
        "raw": raw,
    }


batch_size = 64
train_loader = DataLoader(
    SPRTorch(dset["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(dset["dev"]), batch_size=batch_size * 2, shuffle=False, collate_fn=collate
)


# -------------------------------------------------------------
# 5.  Transformer model
class ClusterTransformer(nn.Module):
    def __init__(
        self,
        vocab_sz,
        cluster_k,
        emb_dim=64,
        num_layers=2,
        nhead=4,
        num_cls=num_classes,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.clu_emb = nn.Embedding(cluster_k + 1, emb_dim)  # +1 for CLS cluster
        self.pos_emb = nn.Parameter(torch.zeros(512, emb_dim))  # enough for max len
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_cls),
        )

    def forward(self, tok, clu, mask):
        seq_len = tok.size(1)
        pos = self.pos_emb[:seq_len]
        x = self.tok_emb(tok) + self.clu_emb(clu) + pos
        x = self.encoder(x, src_key_padding_mask=~mask)
        cls_vec = x[:, 0]  # CLS position
        return self.fc(cls_vec)


model = ClusterTransformer(vocab_size, K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------------------------------------------------
def evaluate(loader, raw_seqs):
    model.eval()
    total_loss = 0
    n = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["tok"], batch["clu"], batch["mask"])
            loss = criterion(logits, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
            p = logits.argmax(1).cpu().tolist()
            preds.extend(p)
            labels.extend(batch["label"].cpu().tolist())
    cwa = color_weighted_acc(raw_seqs, labels, preds)
    swa = shape_weighted_acc(raw_seqs, labels, preds)
    cpx = complexity_weighted_acc(raw_seqs, labels, preds)
    return total_loss / n, preds, labels, (cwa, swa, cpx)


# -------------------------------------------------------------
max_epochs = 20
best_cpx = -1
patience = 5
wait = 0
for epoch in range(1, max_epochs + 1):
    t0 = time.time()
    model.train()
    tot = 0
    n = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["tok"], batch["clu"], batch["mask"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        tot += loss.item() * batch["label"].size(0)
        n += batch["label"].size(0)
    train_loss = tot / n
    tr_loss, tr_preds, tr_labels, (tr_cwa, tr_swa, tr_cpx) = evaluate(
        train_loader, train_loader.dataset.raw
    )
    val_loss, val_preds, val_labels, (v_cwa, v_swa, v_cpx) = evaluate(
        dev_loader, dev_loader.dataset.raw
    )

    ed = experiment_data["SPR_BENCH"]
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append({"cwa": tr_cwa, "swa": tr_swa, "cpx": tr_cpx})
    ed["metrics"]["val"].append({"cwa": v_cwa, "swa": v_swa, "cpx": v_cpx})

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"CWA={v_cwa:.3f} SWA={v_swa:.3f} CompWA={v_cpx:.3f} time={time.time()-t0:.1f}s"
    )

    if v_cpx > best_cpx + 1e-6:
        best_cpx = v_cpx
        wait = 0
        ed["predictions"] = val_preds
        ed["ground_truth"] = val_labels
        best_state = copy.deepcopy(model.state_dict())
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

# -------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
vals = [m["cpx"] for m in experiment_data["SPR_BENCH"]["metrics"]["val"]]
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["epochs"], vals, "o-")
plt.title("Validation Complexity-Weighted Accuracy")
plt.xlabel("Epoch")
plt.ylabel("CompWA")
plt.savefig(os.path.join(working_dir, "val_compwa.png"))
print("Artifacts saved in ./working")
