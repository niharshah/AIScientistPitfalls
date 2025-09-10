import os, pathlib, random, numpy as np, torch
from collections import Counter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------
# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device handling ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# experiment_data skeleton
experiment_data = {
    "spr": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------------------------
# helpers copied from SPR.py (no HF dependency required here)
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split()))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split()))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(w_i for w_i, t, p in zip(w, y_true, y_pred) if t == p) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(w_i for w_i, t, p in zip(w, y_true, y_pred) if t == p) / (sum(w) or 1)


def harmonic_mean(a, b, eps=1e-8):
    return 2 * a * b / (a + b + eps)


# ------------------------------------------------------------------
# 1.  Dataset loading (real → synthetic fallback)
def load_real_spr(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict  # lazy import

    ddict = DatasetDict()
    for split in ("train.csv", "dev.csv", "test.csv"):
        ddict[split.split(".")[0]] = load_dataset(
            "csv", data_files=str(root / split), split="train", cache_dir=".cache_dsets"
        )
    # convert to list-of-dicts for simplicity
    return {k: list(v) for k, v in ddict.items()}


def create_synthetic_spr(n_train=5000, n_dev=1000, n_test=1000, seq_len=10):
    shapes, colors = list("ABCD"), list("123456")

    def make_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(seq_len)
        )

    def label_rule(seq):
        return Counter(tok[0] for tok in seq.split()).most_common(1)[0][0]

    def build(n):
        return [
            {"id": i, "sequence": (s := make_seq()), "label": label_rule(s)}
            for i in range(n)
        ]

    return {"train": build(n_train), "dev": build(n_dev), "test": build(n_test)}


try:
    DATA_PATH = pathlib.Path("SPR_BENCH")
    spr_data = load_real_spr(DATA_PATH)
except Exception:
    print("Real SPR_BENCH not found – falling back to synthetic data.")
    spr_data = create_synthetic_spr()

print({k: len(v) for k, v in spr_data.items()})


# ------------------------------------------------------------------
# 2.  Unsupervised glyph clustering (fit on TRAIN tokens only)
def token_to_vec(tok):
    # simple hand-crafted 2-D vec: [shape_ord, color_ord]
    return [ord(tok[0]) - ord("A"), ord(tok[1]) - ord("0")]


train_tokens = {tok for row in spr_data["train"] for tok in row["sequence"].split()}
X_train_tokens = np.array([token_to_vec(t) for t in train_tokens])
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X_train_tokens)
train_cluster_set = set(kmeans.labels_)  # which clusters appeared in training glyphs

pad_idx = n_clusters  # one extra index used for padding


def seq_to_cluster_ids(seq, max_len):
    ids = kmeans.predict(np.array([token_to_vec(tok) for tok in seq.split()]))
    ids = ids[:max_len]  # truncate if longer
    if len(ids) < max_len:
        ids = np.pad(ids, (0, max_len - len(ids)), constant_values=pad_idx)
    return ids.astype(np.int64)


# ------------------------------------------------------------------
# 3. Torch Dataset
MAX_LEN = max(len(row["sequence"].split()) for row in spr_data["train"])
le = LabelEncoder().fit([r["label"] for r in spr_data["train"]])
n_classes = len(le.classes_)


class SPRDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = seq_to_cluster_ids(row["sequence"], MAX_LEN)
        y = le.transform([row["label"]])[0]
        return {"x": torch.tensor(x), "y": torch.tensor(y), "seq": row["sequence"]}


batch_size = 128
train_ds, dev_ds, test_ds = map(
    SPRDataset, (spr_data["train"], spr_data["dev"], spr_data["test"])
)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


# ------------------------------------------------------------------
# 4. Model definition
class MeanEmbedClassifier(nn.Module):
    def __init__(self, n_tokens, emb_dim, n_out, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(n_tokens, emb_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(emb_dim, n_out)

    def forward(self, x):  # x: [B, L]
        emb = self.embed(x)  # [B, L, D]
        mask = (x != pad_idx).unsqueeze(-1)  # [B, L, 1]
        summed = (emb * mask).sum(1)
        length = mask.sum(1).clamp(min=1e-6)
        mean = summed / length
        return self.fc(mean)


model = MeanEmbedClassifier(n_clusters + 1, 64, n_classes, pad_idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ------------------------------------------------------------------
# 5. Training loop with metric tracking
best_cshm, best_state, wait, patience = -1.0, None, 0, 5
num_epochs = 50

for epoch in range(1, num_epochs + 1):
    # ---- train ----
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["y"].size(0)
    tr_loss = running_loss / len(train_ds)
    experiment_data["spr"]["losses"]["train"].append((epoch, tr_loss))

    # ---- validation ----
    model.eval()
    val_loss, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            val_loss += loss.item() * batch["y"].size(0)
            p = logits.argmax(1).cpu().numpy()
            preds.extend(p)
            gts.extend(batch["y"].cpu().numpy())
            seqs.extend(batch["seq"])
    val_loss /= len(dev_ds)
    experiment_data["spr"]["losses"]["val"].append((epoch, val_loss))

    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cshm = harmonic_mean(cwa, swa)

    # OCGA on dev
    def seq_has_ooc(s):
        seq_clusters = set(
            kmeans.predict(np.array([token_to_vec(t) for t in s.split()]))
        )
        return len(seq_clusters - train_cluster_set) > 0

    ooc_mask = [seq_has_ooc(s) for s in seqs]
    if any(ooc_mask):
        ocga = np.mean([p == t for p, t, m in zip(preds, gts, ooc_mask) if m])
    else:
        ocga = np.nan  # not defined
    experiment_data["spr"]["metrics"]["val"].append((epoch, cwa, swa, cshm, ocga))

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} CSHM={cshm:.3f} OCGA={ocga:.3f}"
    )

    # early stopping on cshm
    if cshm > best_cshm + 1e-4:
        best_cshm, best_state, wait = (
            cshm,
            {k: v.cpu() for k, v in model.state_dict().items()},
            0,
        )
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping triggered")
        break

# ------------------------------------------------------------------
# 6. Test evaluation with best model
model.load_state_dict(best_state)
model.to(device)
model.eval()
test_preds, test_gts, test_seqs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["x"])
        p = logits.argmax(1).cpu().numpy()
        test_preds.extend(p)
        test_gts.extend(batch["y"].cpu().numpy())
        test_seqs.extend(batch["seq"])

experiment_data["spr"]["predictions"] = test_preds
experiment_data["spr"]["ground_truth"] = test_gts

# compute final metrics on test
test_cwa = color_weighted_accuracy(test_seqs, test_gts, test_preds)
test_swa = shape_weighted_accuracy(test_seqs, test_gts, test_preds)
test_cshm = harmonic_mean(test_cwa, test_swa)
test_ooc_mask = [seq_has_ooc(s) for s in test_seqs]
if any(test_ooc_mask):
    test_ocga = np.mean(
        [p == t for p, t, m in zip(test_preds, test_gts, test_ooc_mask) if m]
    )
else:
    test_ocga = np.nan
print(
    f"TEST -> CWA={test_cwa:.3f} SWA={test_swa:.3f} CSHM={test_cshm:.3f} OCGA={test_ocga:.3f}"
)

# ------------------------------------------------------------------
# 7. save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
