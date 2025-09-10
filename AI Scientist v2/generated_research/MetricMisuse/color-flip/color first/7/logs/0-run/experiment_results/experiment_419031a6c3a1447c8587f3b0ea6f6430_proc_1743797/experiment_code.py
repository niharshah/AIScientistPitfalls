import os, time, random, pathlib, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# ========== metric helpers ========================================
def _shape_id(tok):  # first char
    return ord(tok[0].upper()) - ord("A")


def _color_id(tok):  # rest is colour numeric if any
    try:
        return int(tok[1:])
    except ValueError:
        return 0


def count_color_variety(sequence):
    return len(set(_color_id(t) for t in sequence.split() if t))


def count_shape_variety(sequence):
    return len(set(_shape_id(t) for t in sequence.split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


# ------------------------------------------------------------------
# ========== data ===================================================
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def synth_dataset(n_rows, seed=42, shapes=6, colors=6, max_len=8, n_labels=4):
    rnd = random.Random(seed)
    rows = []
    for idx in range(n_rows):
        L = rnd.randint(3, max_len)
        seq = " ".join(
            chr(ord("A") + rnd.randint(0, shapes - 1)) + str(rnd.randint(0, colors - 1))
            for _ in range(L)
        )
        rows.append(
            {"id": str(idx), "sequence": seq, "label": rnd.randint(0, n_labels - 1)}
        )
    return rows


def get_dataset():
    root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if root.exists() and (root / "train.csv").exists():
        print("Loaded real SPR_BENCH dataset.")
        return load_spr_bench(root)
    # fallback
    print("Real dataset not found – using synthetic data.")
    return DatasetDict(
        train=load_dataset(
            "json", data_files={"train": [synth_dataset(4000)]}, split="train"
        ),
        dev=load_dataset(
            "json", data_files={"train": [synth_dataset(800, seed=1)]}, split="train"
        ),
        test=load_dataset(
            "json", data_files={"train": [synth_dataset(1000, seed=2)]}, split="train"
        ),
    )


dset = get_dataset()
num_classes = len(set(dset["train"]["label"]))
print(f"Number of classes: {num_classes}")

# ------------------------------------------------------------------
# ========== vocabulary & clustering ===============================
tokens = set(tok for seq in dset["train"]["sequence"] for tok in seq.split())
tok2idx = {tok: i + 1 for i, tok in enumerate(sorted(tokens))}  # 0 is PAD
vocab_size = len(tok2idx) + 1

# 2-dim structural feature per glyph
features = np.array(
    [[_shape_id(tok), _color_id(tok)] for tok in sorted(tokens)], dtype=np.float32
)
K = 12
kmeans = KMeans(n_clusters=K, n_init=20, random_state=0).fit(features)
cluster_ids = kmeans.labels_
cluster_map = {tok: cluster_ids[i] for i, tok in enumerate(sorted(tokens))}

print("Cluster distribution:", np.bincount(cluster_ids))


# ------------------------------------------------------------------
# ========== torch dataset =========================================
def encode_sequence(seq):
    toks = seq.split()
    ids = [tok2idx.get(t, 0) for t in toks]
    clus = [cluster_map.get(t, 0) for t in toks]
    return ids, clus


class SPRTorch(Dataset):
    def __init__(self, split):
        self.raw = split["sequence"]
        enc = [encode_sequence(s) for s in self.raw]
        self.ids = [e[0] for e in enc]
        self.clu = [e[1] for e in enc]
        self.labels = split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "tok": torch.tensor(self.ids[idx], dtype=torch.long),
            "clu": torch.tensor(self.clu[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    max_len = max(len(b["tok"]) for b in batch)
    tok = torch.zeros((len(batch), max_len), dtype=torch.long)
    clu = torch.zeros((len(batch), max_len), dtype=torch.long)
    length = []
    labels = []
    for i, b in enumerate(batch):
        L = len(b["tok"])
        tok[i, :L] = b["tok"]
        clu[i, :L] = b["clu"]
        length.append(L)
        labels.append(b["label"])
    return {
        "tok": tok,
        "clu": clu,
        "len": torch.tensor(length),
        "label": torch.stack(labels),
    }


train_ds = SPRTorch(dset["train"])
dev_ds = SPRTorch(dset["dev"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate)


# ------------------------------------------------------------------
# ========== model ==================================================
class ClusterAwareTransformer(nn.Module):
    def __init__(
        self, vocab_sz, cluster_k, emb_dim=64, n_heads=4, n_layers=2, n_classes=3
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.clu_emb = nn.Embedding(cluster_k, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(128, emb_dim)  # assume seq len <128
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, n_classes)
        )

    def forward(self, tok, clu, length):
        B, L = tok.shape
        pos = torch.arange(L, device=tok.device).unsqueeze(0).repeat(B, 1)
        x = self.tok_emb(tok) + self.clu_emb(clu) + self.pos_emb(pos)
        mask = tok == 0
        z = self.encoder(x, src_key_padding_mask=mask)
        # mean pool ignoring padding
        mask_f = (~mask).unsqueeze(-1)
        pooled = (z * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        return self.fc(pooled)


model = ClusterAwareTransformer(vocab_size, K, n_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

# ------------------------------------------------------------------
# ========== experiment logger =====================================
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
# ========== training ==============================================
def evaluate(loader, raw_sequences):
    model.eval()
    preds, lbls = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["tok"], batch["clu"], batch["len"])
            preds.extend(logits.argmax(1).cpu().tolist())
            lbls.extend(batch["label"].cpu().tolist())
    cwa = color_weighted_accuracy(raw_sequences, lbls, preds)
    swa = shape_weighted_accuracy(raw_sequences, lbls, preds)
    cpx = complexity_weighted_accuracy(raw_sequences, lbls, preds)
    return preds, lbls, (cwa, swa, cpx)


best_val_cpx = -1
patience, wait = 5, 0
EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss, seen = 0.0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["tok"], batch["clu"], batch["len"])
        loss = criterion(out, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["label"].size(0)
        seen += batch["label"].size(0)
    train_loss = total_loss / seen

    tr_preds, tr_lbls, (tr_cwa, tr_swa, tr_cpx) = evaluate(train_loader, train_ds.raw)
    val_preds, val_lbls, (v_cwa, v_swa, v_cpx) = evaluate(dev_loader, dev_ds.raw)

    ed = experiment_data["SPR_BENCH"]
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(train_loss)
    ed["metrics"]["train"].append({"cwa": tr_cwa, "swa": tr_swa, "cpx": tr_cpx})
    ed["metrics"]["val"].append({"cwa": v_cwa, "swa": v_swa, "cpx": v_cpx})

    print(
        f"Epoch {epoch:02d}: validation_loss = {v_cpx:.4f}  | Val CWA {v_cwa:.3f} SWA {v_swa:.3f} Cpx {v_cpx:.3f}"
    )

    if v_cpx > best_val_cpx + 1e-6:
        best_val_cpx = v_cpx
        wait = 0
        ed["predictions"] = val_preds
        ed["ground_truth"] = val_lbls
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# ------------------------------------------------------------------
# ========== save artefacts ========================================
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
val_cpx_curve = [m["cpx"] for m in experiment_data["SPR_BENCH"]["metrics"]["val"]]
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["epochs"], val_cpx_curve, marker="o")
plt.title("Validation Complexity-Weighted Accuracy")
plt.xlabel("Epoch")
plt.ylabel("CompWA")
plt.savefig(os.path.join(working_dir, "val_compwa.png"))
print("Finished. Data saved to ./working/")
