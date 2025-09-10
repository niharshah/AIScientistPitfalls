# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, copy, math, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict
from typing import List, Dict, Tuple

# ---------------------------------------------------- #
# 0. house-keeping                                      #
# ---------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ---------------------------------------------------- #
# 1. locate SPR_BENCH                                   #
# ---------------------------------------------------- #
def find_spr_bench_root() -> pathlib.Path:
    env = os.getenv("SPR_BENCH_ROOT")
    cands = [pathlib.Path(env)] if env else []
    cwd = pathlib.Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        cands.append(parent / "SPR_BENCH")
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
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split()))


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
def glyph_vector(g: str) -> List[int]:
    if len(g) >= 2:
        return [ord(g[0]) - 65, ord(g[1]) - 48]
    return [ord(g[0]) - 65, 0]


all_glyphs = set(tok for seq in spr["train"]["sequence"] for tok in seq.strip().split())
vecs = np.array([glyph_vector(g) for g in all_glyphs])
k_clusters = 16
kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
glyph_to_cluster = {g: c for g, c in zip(all_glyphs, kmeans.fit_predict(vecs))}
print(f"Clustered {len(all_glyphs)} glyphs into {k_clusters} clusters.")

# ---------------------------------------------------- #
# 4. Sequence→tensor dataset                            #
# ---------------------------------------------------- #
PAD_IDX = k_clusters  # padding index after clusters 0..k-1


def seq_to_clusters(seq: str) -> List[int]:
    return [glyph_to_cluster.get(tok, 0) for tok in seq.strip().split()]


def build_known_pairs(seqs) -> set:
    s = set()
    for seq in seqs:
        for tok in seq.strip().split():
            s.add((tok, glyph_to_cluster.get(tok, 0)))
    return s


train_known_pairs = build_known_pairs(spr["train"]["sequence"])


def sequence_novelty_weight(seq: str) -> float:
    total = 0
    novel = 0
    for tok in seq.strip().split():
        pair = (tok, glyph_to_cluster.get(tok, 0))
        total += 1
        if pair not in train_known_pairs:
            novel += 1
    novelty_ratio = novel / total if total else 0.0
    return 1.0 + novelty_ratio


def snwa(seqs, y_true, y_pred):
    w = [sequence_novelty_weight(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


class SPRSeqDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int]):
        self.seqs = [seq_to_clusters(s) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"seq": self.seqs[idx], "label": self.labels[idx]}


def collate(batch):
    lengths = [len(b["seq"]) for b in batch]
    max_len = max(lengths)
    seq_tensor = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    for i, b in enumerate(batch):
        seq_tensor[i, : len(b["seq"])] = torch.tensor(b["seq"], dtype=torch.long)
    labels = torch.stack([b["label"] for b in batch])
    lengths = torch.tensor(lengths, dtype=torch.long)
    return {"seq": seq_tensor, "len": lengths, "label": labels}


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
# 5. model                                             #
# ---------------------------------------------------- #
class GRUClassifier(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        pad_idx: int,
        emb_dim: int = 64,
        hid: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_clusters + 1, emb_dim, padding_idx=pad_idx)
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


model = GRUClassifier(k_clusters, PAD_IDX, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------------------------------- #
# 6. evaluation helper                                 #
# ---------------------------------------------------- #
def evaluate(model, loader, seq_raw) -> Dict[str, float]:
    model.eval()
    totals, preds, gts = 0.0, [], []
    total_loss = 0.0
    with torch.no_grad():
        idx_start = 0
        for batch in loader:
            bseq = batch["seq"].to(device)
            blen = batch["len"].to(device)
            lbl = batch["label"].to(device)
            logits = model(bseq, blen)
            loss = criterion(logits, lbl)
            total_loss += loss.item() * lbl.size(0)
            pred = logits.argmax(1)
            preds.extend(pred.cpu().tolist())
            gts.extend(lbl.cpu().tolist())
            idx_start += lbl.size(0)
    avg_loss = total_loss / len(gts)
    cwa = color_weighted_accuracy(seq_raw, gts, preds)
    swa = shape_weighted_accuracy(seq_raw, gts, preds)
    hcs = harmonic_csa(cwa, swa)
    snw = snwa(seq_raw, gts, preds)
    return {
        "loss": avg_loss,
        "CWA": cwa,
        "SWA": swa,
        "HCSA": hcs,
        "SNWA": snw,
        "preds": preds,
        "gts": gts,
    }


# ---------------------------------------------------- #
# 7. training loop with early stopping                 #
# ---------------------------------------------------- #
max_epochs = 25
patience = 5
best_hcs = -1.0
since_best = 0
best_state = None

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
    }
}

for epoch in range(1, max_epochs + 1):
    # --- train ---
    model.train()
    total_loss = 0.0
    n_seen = 0
    for batch in train_loader:
        bseq = batch["seq"].to(device)
        blen = batch["len"].to(device)
        lbl = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(bseq, blen)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * lbl.size(0)
        n_seen += lbl.size(0)
    train_loss = total_loss / n_seen
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))
    # --- validate ---
    val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_stats["loss"]))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (
            epoch,
            val_stats["CWA"],
            val_stats["SWA"],
            val_stats["HCSA"],
            val_stats["SNWA"],
        )
    )
    print(
        f'Epoch {epoch}: validation_loss = {val_stats["loss"]:.4f} '
        f'HCSA={val_stats["HCSA"]:.3f} SNWA={val_stats["SNWA"]:.3f}'
    )
    # early stopping
    if val_stats["HCSA"] > best_hcs + 1e-6:
        best_hcs = val_stats["HCSA"]
        best_state = copy.deepcopy(model.state_dict())
        since_best = 0
    else:
        since_best += 1
    if since_best >= patience:
        print("Early stopping.")
        break

# restore best
if best_state:
    model.load_state_dict(best_state)

# ---------------------------------------------------- #
# 8. final evaluation                                  #
# ---------------------------------------------------- #
dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
test_final = evaluate(model, test_loader, spr["test"]["sequence"])
experiment_data["SPR_BENCH"]["predictions"]["dev"] = dev_final["preds"]
experiment_data["SPR_BENCH"]["ground_truth"]["dev"] = dev_final["gts"]
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_final["preds"]
experiment_data["SPR_BENCH"]["ground_truth"]["test"] = test_final["gts"]

print(
    f'Final Dev - CWA:{dev_final["CWA"]:.3f} SWA:{dev_final["SWA"]:.3f} '
    f'HCSA:{dev_final["HCSA"]:.3f} SNWA:{dev_final["SNWA"]:.3f}'
)
print(
    f'Final Test - CWA:{test_final["CWA"]:.3f} SWA:{test_final["SWA"]:.3f} '
    f'HCSA:{test_final["HCSA"]:.3f} SNWA:{test_final["SNWA"]:.3f}'
)

# ---------------------------------------------------- #
# 9. save experiment data                              #
# ---------------------------------------------------- #
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Saved experiment data to {working_dir}/experiment_data.npy")
