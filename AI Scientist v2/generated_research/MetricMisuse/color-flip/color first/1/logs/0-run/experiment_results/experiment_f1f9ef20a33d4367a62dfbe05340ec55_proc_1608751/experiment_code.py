import os, time, random, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------------------------#
# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------------------------#
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------#
# experiment dict skeleton
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# -----------------------------------------------------------------------------#
# helpers / metrics
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({k: _load(f"{k}.csv") for k in ["train", "dev", "test"]})


def tokenize(seq: str):
    return seq.strip().split()


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in tokenize(seq) if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in tokenize(seq) if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def composite_variety_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def harmonic_mean_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# -----------------------------------------------------------------------------#
# synthetic fallback (for CI or missing data)
def create_synth():
    def rand_seq():
        return " ".join(
            random.choice("ABCD") + random.choice("0123")
            for _ in range(random.randint(4, 10))
        )

    def lbl(seq):
        return (count_color_variety(seq) + count_shape_variety(seq)) % 4

    def make_split(n):
        seqs = [rand_seq() for _ in range(n)]
        return {"sequence": seqs, "label": [lbl(s) for s in seqs]}

    dd = DatasetDict()
    for k, n in zip(["train", "dev", "test"], [2000, 400, 400]):
        dd[k] = load_dataset("json", split=[], data=make_split(n))
    return dd


# -----------------------------------------------------------------------------#
# data loading
try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded official SPR_BENCH dataset.")
except Exception as e:
    print(f"Falling back to synthetic dataset. Reason: {e}")
    spr = create_synth()

num_classes = len(set(spr["train"]["label"]))

# -----------------------------------------------------------------------------#
# glyph clustering with explicit UNK id
try:
    from sklearn.cluster import KMeans

    sk_ok = True
except ImportError:
    sk_ok = False
    print("sklearn not available, random clusters will be used.")


def glyph_vec(tok):
    # simple 2-d vector: normalised ASCII codes for first two chars
    s = ord(tok[0]) if tok else 65
    c = ord(tok[1]) if len(tok) > 1 else 48
    return [(s - 32) / 95.0, (c - 32) / 95.0]  # scale to [0,1]


all_tokens = [tk for seq in spr["train"]["sequence"] for tk in tokenize(seq)]
unique_tokens = sorted(set(all_tokens))
N_CLUSTERS = 16
token_to_cluster = {}

if sk_ok:
    km = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10).fit(
        np.asarray([glyph_vec(t) for t in unique_tokens])
    )
    for t, cid in zip(unique_tokens, km.labels_):
        token_to_cluster[t] = int(cid)
else:
    for t in unique_tokens:
        token_to_cluster[t] = random.randint(0, N_CLUSTERS - 1)

PAD_ID = N_CLUSTERS  # pad
UNK_ID = N_CLUSTERS + 1  # unknown glyph
VOCAB_SIZE = N_CLUSTERS + 2  # clusters + pad + unk


def seq_to_cluster_ids(seq: str):
    return [token_to_cluster.get(tok, UNK_ID) for tok in tokenize(seq)]


# -----------------------------------------------------------------------------#
# PyTorch dataset
class SPRClusterDataset(Dataset):
    def __init__(self, sequences, labels):
        self.seqs_txt = sequences
        self.labels = labels
        self.ids = [seq_to_cluster_ids(s) for s in sequences]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "ids": self.ids[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": self.seqs_txt[idx],
        }


def collate_fn(batch):
    max_len = max(len(b["ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, : len(b["ids"])] = torch.tensor(b["ids"], dtype=torch.long)
    attention_mask = input_ids != PAD_ID
    labels = torch.stack([b["label"] for b in batch])
    texts = [b["text"] for b in batch]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "texts": texts,
    }


train_ds = SPRClusterDataset(spr["train"]["sequence"], spr["train"]["label"])
dev_ds = SPRClusterDataset(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRClusterDataset(spr["test"]["sequence"], spr["test"]["label"])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)


# -----------------------------------------------------------------------------#
# model
class ClusterTransformer(nn.Module):
    def __init__(self, vocab, pad_idx, emb_dim=64, nhead=4, nlayers=2, n_classes=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_idx)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.fc = nn.Linear(emb_dim, n_classes)

    def forward(self, x, mask):
        emb = self.emb(x)
        src_key_padding_mask = ~mask.bool()
        enc = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        # mean pool (avoid /0 with tiny epsilon)
        denom = mask.sum(1, keepdim=True).clamp(min=1).type_as(enc)
        seq_repr = (enc * mask.unsqueeze(-1)).sum(1) / denom
        return self.fc(seq_repr)


model = ClusterTransformer(VOCAB_SIZE, PAD_ID, n_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# -----------------------------------------------------------------------------#
# training loop
EPOCHS = 15
best_hmwa, best_state = 0.0, None

for epoch in range(1, EPOCHS + 1):
    # ------------- train -------------
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
    train_loss = running_loss / len(train_ds)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ------------- validation -------------
    model.eval()
    val_loss, preds, golds, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            p = logits.argmax(dim=-1).cpu().tolist()
            g = batch["labels"].cpu().tolist()
            preds.extend(p)
            golds.extend(g)
            seqs.extend(batch["texts"])
    val_loss /= len(dev_ds)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    cwa = color_weighted_accuracy(seqs, golds, preds)
    swa = shape_weighted_accuracy(seqs, golds, preds)
    cva = composite_variety_accuracy(seqs, golds, preds)
    hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"cwa": cwa, "swa": swa, "cva": cva, "hmwa": hmwa}
    )
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | CWA={cwa:.4f} SWA={swa:.4f} CVA={cva:.4f} HMWA={hmwa:.4f}"
    )

    if hmwa > best_hmwa:
        best_hmwa = hmwa
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

# -----------------------------------------------------------------------------#
# test evaluation
if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
preds, golds, seqs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        preds.extend(logits.argmax(-1).cpu().tolist())
        golds.extend(batch["labels"].cpu().tolist())
        seqs.extend(batch["texts"])

cwa_t = color_weighted_accuracy(seqs, golds, preds)
swa_t = shape_weighted_accuracy(seqs, golds, preds)
cva_t = composite_variety_accuracy(seqs, golds, preds)
hmwa_t = harmonic_mean_weighted_accuracy(cwa_t, swa_t)

print(f"\nTEST: CWA={cwa_t:.4f} SWA={swa_t:.4f} CVA={cva_t:.4f} HMWA={hmwa_t:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = golds
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "cwa": cwa_t,
    "swa": swa_t,
    "cva": cva_t,
    "hmwa": hmwa_t,
}

# -----------------------------------------------------------------------------#
# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
