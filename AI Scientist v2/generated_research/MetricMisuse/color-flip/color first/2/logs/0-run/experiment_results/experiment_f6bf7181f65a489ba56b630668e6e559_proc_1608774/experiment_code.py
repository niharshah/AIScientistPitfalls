import os, time, random, json, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, Dataset as HFDataset
from sklearn.cluster import KMeans

# ------------------------------------------------------------------ #
# basic folders & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------ #
# helpers for metrics
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ------------------------------------------------------------------ #
# data loading
def load_spr_bench(possible_root: pathlib.Path) -> DatasetDict | None:
    if not possible_root.exists():
        return None

    def _ld(name):
        return load_dataset(
            "csv",
            data_files=str(possible_root / name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({k: _ld(f"{k}.csv") for k in ["train", "dev", "test"]})


# ---- synthetic fallback with BUGFIX ---- #
def build_synthetic() -> DatasetDict:
    """
    Build a small synthetic SPR-like dataset entirely in memory.
    (Bugfix: use Dataset.from_list instead of load_dataset with bogus file names.)
    """
    shapes, colors = list("ABCD"), list("1234")

    def gen_split(n, seed):
        random.seed(seed)
        rows = []
        for i in range(n):
            length = random.randint(3, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            lbl = (
                count_color_variety(seq) + count_shape_variety(seq)
            ) % 4  # deterministic label
            rows.append({"id": i, "sequence": seq, "label": lbl})
        return HFDataset.from_list(rows)

    return DatasetDict(
        {
            "train": gen_split(600, 0),
            "dev": gen_split(120, 1),
            "test": gen_split(120, 2),
        }
    )


DATA_PATH = pathlib.Path("SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
if spr is None:
    print("SPR_BENCH not found – using synthetic data.")
    spr = build_synthetic()

num_classes = len(set(spr["train"]["label"]))
# ------------------------------------------------------------------ #
# build vocabulary, clusters
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
tokens_sorted = sorted(set(all_tokens))
token2id = {t: i + 1 for i, t in enumerate(tokens_sorted)}  # 0 reserved PAD

shapes = sorted({t[0] for t in tokens_sorted})
colors = sorted({t[1] for t in tokens_sorted})
shape2id = {s: i for i, s in enumerate(shapes)}
color2id = {c: i for i, c in enumerate(colors)}

token_vec = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in tokens_sorted], dtype=float
)
n_clusters = min(max(4, len(token_vec) // 3), 32)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(token_vec)
token2cluster = {
    tok: int(cl) + 1 for tok, cl in zip(tokens_sorted, kmeans.labels_)
}  # 0 pad


# ------------------------------------------------------------------ #
# torch dataset
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seq = spr[split]["sequence"]
        self.lab = spr[split]["label"]

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        toks = self.seq[idx].split()
        return {
            "tokens": [token2id[t] for t in toks],
            "clusters": [token2cluster[t] for t in toks],
            "label": self.lab[idx],
            "seq_str": self.seq[idx],
        }


def collate(batch):
    maxlen = max(len(b["tokens"]) for b in batch)

    def pad(key):
        return torch.tensor(
            [b[key] + [0] * (maxlen - len(b[key])) for b in batch], dtype=torch.long
        )

    tokens, clusters = pad("tokens"), pad("clusters")
    mask = tokens != 0
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    seqs = [b["seq_str"] for b in batch]
    return {
        "tokens": tokens,
        "clusters": clusters,
        "mask": mask,
        "labels": labels,
        "seqs": seqs,
    }


train_loader = DataLoader(
    SPRTorchDataset("train"), batch_size=128, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRTorchDataset("dev"), batch_size=128, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorchDataset("test"), batch_size=128, shuffle=False, collate_fn=collate
)


# ------------------------------------------------------------------ #
# model
class TransformerGlyphModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_clusters,
        num_classes,
        emb_dim=32,
        n_heads=4,
        n_layers=2,
        max_len=60,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.cluster_emb = nn.Embedding(n_clusters + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, emb_dim)  # +1 for [CLS]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.cls_param = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, tokens, clusters, mask):
        B, L = tokens.shape
        pos_ids = torch.arange(1, L + 1, device=tokens.device).unsqueeze(0).repeat(B, 1)
        x = self.token_emb(tokens) + self.cluster_emb(clusters) + self.pos_emb(pos_ids)
        cls = self.cls_param.repeat(B, 1, 1)
        x = torch.cat([cls, x], dim=1)
        src_mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1
        )
        enc = self.encoder(x, src_key_padding_mask=~src_mask)
        return self.fc(enc[:, 0])


# ------------------------------------------------------------------ #
# train / eval helpers
def run_eval(model, loader, criterion):
    model.eval()
    tot_loss, all_p, all_t, all_s = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["tokens"], batch["clusters"], batch["mask"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_p.extend(preds)
            all_t.extend(batch["labels"].cpu().tolist())
            all_s.extend(batch["seqs"])
    avg_loss = tot_loss / len(loader.dataset)
    metrics = {
        "CWA": color_weighted_accuracy(all_s, all_t, all_p),
        "SWA": shape_weighted_accuracy(all_s, all_t, all_p),
        "GCWA": glyph_complexity_weighted_accuracy(all_s, all_t, all_p),
    }
    return avg_loss, metrics, all_p, all_t


# ------------------------------------------------------------------ #
# experiment log
experiment_data = {
    "SPR_transformer": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------------ #
# training loop
model = TransformerGlyphModel(len(token2id), n_clusters, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

num_epochs = 8
start = time.time()
for epoch in range(1, num_epochs + 1):
    model.train()
    run_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["tokens"], batch["clusters"], batch["mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["labels"].size(0)
    train_loss = run_loss / len(train_loader.dataset)
    val_loss, val_metrics, _, _ = run_eval(model, val_loader, criterion)
    tr_loss_tmp, tr_metrics, _, _ = run_eval(model, train_loader, criterion)

    experiment_data["SPR_transformer"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_transformer"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_transformer"]["metrics"]["train"].append(tr_metrics)
    experiment_data["SPR_transformer"]["metrics"]["val"].append(val_metrics)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
        f"CWA={val_metrics['CWA']:.3f} SWA={val_metrics['SWA']:.3f} GCWA={val_metrics['GCWA']:.3f}"
    )

# ------------------------------------------------------------------ #
# final test
test_loss, test_metrics, preds, tgts = run_eval(model, test_loader, criterion)
experiment_data["SPR_transformer"]["losses"]["test"] = test_loss
experiment_data["SPR_transformer"]["metrics"]["test"] = test_metrics
experiment_data["SPR_transformer"]["predictions"] = preds
experiment_data["SPR_transformer"]["ground_truth"] = tgts

print(
    f"\nTEST: loss={test_loss:.4f} | "
    f"CWA={test_metrics['CWA']:.3f} SWA={test_metrics['SWA']:.3f} GCWA={test_metrics['GCWA']:.3f}"
)

# ------------------------------------------------------------------ #
# save experiment
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
print(f"Total runtime {time.time()-start:.2f}s")
