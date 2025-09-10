import os, pathlib, random, copy, math, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict
from typing import List, Dict

# ---------------------------------------------------- #
# 0. house-keeping                                      #
# ---------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ---------------------------------------------------- #
# 1. locate & load SPR_BENCH                            #
# ---------------------------------------------------- #
def find_spr_bench_root() -> pathlib.Path:
    env = os.getenv("SPR_BENCH_ROOT")
    cands = [pathlib.Path(env)] if env else []
    cwd = pathlib.Path.cwd()
    for p in [cwd] + list(cwd.parents):
        cands.append(p / "SPR_BENCH")
    cands.extend(
        [
            pathlib.Path.home() / "SPR_BENCH",
            pathlib.Path.home() / "AI-Scientist-v2" / "SPR_BENCH",
            pathlib.Path("/workspace/SPR_BENCH"),
            pathlib.Path("/data/SPR_BENCH"),
        ]
    )
    for p in cands:
        if (
            (p / "train.csv").exists()
            and (p / "dev.csv").exists()
            and (p / "test.csv").exists()
        ):
            print(f"Found SPR_BENCH at {p}")
            return p.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


DATA_PATH = find_spr_bench_root()


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({split: _load(split) for split in ["train", "dev", "test"]})


spr = load_spr_bench(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))


# ---------------------------------------------------- #
# 2. helper metrics                                     #
# ---------------------------------------------------- #
def count_color_variety(seq: str) -> int:
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(t[0] for t in seq.split()))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [cw if t == p else 0 for cw, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [sw if t == p else 0 for sw, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def harmonic_csa(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ---------------------------------------------------- #
# 3. glyph clustering                                   #
# ---------------------------------------------------- #
def glyph_vector(g: str):
    return [ord(g[0]) - 65, ord(g[1]) - 48] if len(g) >= 2 else [ord(g[0]) - 65, 0]


all_glyphs = set(tok for seq in spr["train"]["sequence"] for tok in seq.split())
vecs = np.array([glyph_vector(g) for g in all_glyphs])
k_clusters = 16
kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
glyph_to_cluster = {g: c for g, c in zip(all_glyphs, kmeans.fit_predict(vecs))}
print(f"Clustered {len(all_glyphs)} glyphs into {k_clusters} clusters.")

# ---------------------------------------------------- #
# 4. sequence-to-tensor dataset                         #
# ---------------------------------------------------- #
PAD_IDX = k_clusters


def seq_to_clusters(seq: str):
    return [glyph_to_cluster.get(t, 0) for t in seq.split()]


def build_known_pairs(seqs):
    return {(tok, glyph_to_cluster.get(tok, 0)) for seq in seqs for tok in seq.split()}


train_known_pairs = build_known_pairs(spr["train"]["sequence"])


def sequence_novelty_weight(seq: str):
    total = novel = 0
    for tok in seq.split():
        total += 1
        novel += (tok, glyph_to_cluster.get(tok, 0)) not in train_known_pairs
    return 1.0 + (novel / total if total else 0.0)


def snwa(seqs, y_true, y_pred):
    w = [sequence_novelty_weight(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


class SPRSeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = [seq_to_clusters(s) for s in seqs]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"seq": self.seqs[idx], "label": self.labels[idx]}


def collate(batch):
    lens = [len(b["seq"]) for b in batch]
    max_len = max(lens)
    seq_tensor = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    for i, b in enumerate(batch):
        seq_tensor[i, : lens[i]] = torch.tensor(b["seq"], dtype=torch.long)
    return {
        "seq": seq_tensor,
        "len": torch.tensor(lens),
        "label": torch.stack([b["label"] for b in batch]),
    }


batch_size = 128
train_loader = DataLoader(
    SPRSeqDataset(spr["train"]["sequence"], spr["train"]["label"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRSeqDataset(spr["dev"]["sequence"], spr["dev"]["label"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRSeqDataset(spr["test"]["sequence"], spr["test"]["label"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)


# ---------------------------------------------------- #
# 5. model with FROZEN cluster embeddings              #
# ---------------------------------------------------- #
class GRUClassifier(nn.Module):
    def __init__(
        self, n_clusters, pad_idx, emb_dim=64, hid=128, num_classes=10, frozen=True
    ):
        super().__init__()
        self.emb = nn.Embedding(n_clusters + 1, emb_dim, padding_idx=pad_idx)
        if frozen:
            self.emb.weight.requires_grad_(False)  # key line for ablation
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hid * 2, num_classes)

    def forward(self, x, lens):
        x = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.out(h)


model = GRUClassifier(k_clusters, PAD_IDX, num_classes=num_classes, frozen=True).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)


# ---------------------------------------------------- #
# 6. evaluation helper                                 #
# ---------------------------------------------------- #
def evaluate(model, loader, seq_raw):
    model.eval()
    preds = []
    gts = []
    tot_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["seq"].to(device), batch["len"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            tot_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch["label"].tolist())
    avg_loss = tot_loss / len(gts)
    cwa = color_weighted_accuracy(seq_raw, gts, preds)
    swa = shape_weighted_accuracy(seq_raw, gts, preds)
    return {
        "loss": avg_loss,
        "CWA": cwa,
        "SWA": swa,
        "HCSA": harmonic_csa(cwa, swa),
        "SNWA": snwa(seq_raw, gts, preds),
        "preds": preds,
        "gts": gts,
    }


# ---------------------------------------------------- #
# 7. training loop with early stopping                 #
# ---------------------------------------------------- #
max_epochs, patience = 25, 5
best_hcs, best_state = -1.0, None
since_best = 0

experiment_data = {
    "Frozen-Cluster-Embeddings": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": {"dev": [], "test": []},
            "ground_truth": {"dev": [], "test": []},
        }
    }
}

for epoch in range(1, max_epochs + 1):
    model.train()
    tot_loss = 0.0
    n_seen = 0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["seq"].to(device), batch["len"].to(device))
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch["label"].size(0)
        n_seen += batch["label"].size(0)
    tr_loss = tot_loss / n_seen
    experiment_data["Frozen-Cluster-Embeddings"]["SPR_BENCH"]["losses"]["train"].append(
        (epoch, tr_loss)
    )

    val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])
    experiment_data["Frozen-Cluster-Embeddings"]["SPR_BENCH"]["losses"]["val"].append(
        (epoch, val_stats["loss"])
    )
    experiment_data["Frozen-Cluster-Embeddings"]["SPR_BENCH"]["metrics"]["val"].append(
        (
            epoch,
            val_stats["CWA"],
            val_stats["SWA"],
            val_stats["HCSA"],
            val_stats["SNWA"],
        )
    )
    print(
        f"Epoch {epoch}: val_loss={val_stats['loss']:.4f} HCSA={val_stats['HCSA']:.3f} SNWA={val_stats['SNWA']:.3f}"
    )
    if val_stats["HCSA"] > best_hcs + 1e-6:
        best_hcs = val_stats["HCSA"]
        best_state = copy.deepcopy(model.state_dict())
        since_best = 0
    else:
        since_best += 1
    if since_best >= patience:
        print("Early stopping.")
        break

if best_state:
    model.load_state_dict(best_state)

# ---------------------------------------------------- #
# 8. final evaluation                                  #
# ---------------------------------------------------- #
dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
test_final = evaluate(model, test_loader, spr["test"]["sequence"])
exp = experiment_data["Frozen-Cluster-Embeddings"]["SPR_BENCH"]
exp["predictions"]["dev"], exp["ground_truth"]["dev"] = (
    dev_final["preds"],
    dev_final["gts"],
)
exp["predictions"]["test"], exp["ground_truth"]["test"] = (
    test_final["preds"],
    test_final["gts"],
)

print(
    f"Final Dev - CWA:{dev_final['CWA']:.3f} SWA:{dev_final['SWA']:.3f} "
    f"HCSA:{dev_final['HCSA']:.3f} SNWA:{dev_final['SNWA']:.3f}"
)
print(
    f"Final Test - CWA:{test_final['CWA']:.3f} SWA:{test_final['SWA']:.3f} "
    f"HCSA:{test_final['HCSA']:.3f} SNWA:{test_final['SNWA']:.3f}"
)

# ---------------------------------------------------- #
# 9. save experiment data                              #
# ---------------------------------------------------- #
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Saved experiment data to {working_dir}/experiment_data.npy")
