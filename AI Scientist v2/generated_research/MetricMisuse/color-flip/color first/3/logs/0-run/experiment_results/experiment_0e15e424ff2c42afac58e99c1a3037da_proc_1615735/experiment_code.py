import os, pathlib, random, copy, math, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from typing import List, Dict

# ---------------------------------------------------- #
# 0. house-keeping & reproducibility                    #
# ---------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# experiment data container (required format)
experiment_data = {
    "RandomClusterAssignment": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": {"dev": [], "test": []},
            "ground_truth": {"dev": [], "test": []},
        }
    }
}


# ---------------------------------------------------- #
# 1. locate and load SPR_BENCH                          #
# ---------------------------------------------------- #
def find_spr_bench_root() -> pathlib.Path:
    env = os.getenv("SPR_BENCH_ROOT")
    cands = [pathlib.Path(env)] if env else []
    cwd = pathlib.Path.cwd()
    for p in [cwd] + list(cwd.parents):
        cands.append(p / "SPR_BENCH")
    cands += [
        pathlib.Path.home() / "SPR_BENCH",
        pathlib.Path.home() / "AI-Scientist-v2" / "SPR_BENCH",
        pathlib.Path("/workspace/SPR_BENCH"),
        pathlib.Path("/data/SPR_BENCH"),
    ]
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
    def _load(split):
        return load_dataset(
            "csv",
            data_files=str(root / f"{split}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(s) for s in ["train", "dev", "test"]})


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
# 3. RANDOM glyph→cluster mapping (ablation)            #
# ---------------------------------------------------- #
all_glyphs = set(tok for seq in spr["train"]["sequence"] for tok in seq.strip().split())
k_clusters = 16
glyph_list = list(all_glyphs)
rng = np.random.RandomState(SEED)
rng.shuffle(glyph_list)
glyph_to_cluster = {g: i % k_clusters for i, g in enumerate(glyph_list)}
print(f"(Random) Assigned {len(all_glyphs)} glyphs to {k_clusters} clusters.")

# ---------------------------------------------------- #
# 4. Sequence → tensor dataset                          #
# ---------------------------------------------------- #
PAD_IDX = k_clusters  # padding after 0..k-1


def seq_to_clusters(seq: str) -> List[int]:
    return [glyph_to_cluster.get(tok, 0) for tok in seq.strip().split()]


def build_known_pairs(seqs) -> set:
    return {
        (tok, glyph_to_cluster.get(tok, 0))
        for seq in seqs
        for tok in seq.strip().split()
    }


train_known_pairs = build_known_pairs(spr["train"]["sequence"])


def sequence_novelty_weight(seq: str) -> float:
    total, novel = 0, 0
    for tok in seq.strip().split():
        pair = (tok, glyph_to_cluster.get(tok, 0))
        total += 1
        if pair not in train_known_pairs:
            novel += 1
    return 1.0 + (novel / total if total else 0.0)


def snwa(seqs, y_true, y_pred):
    w = [sequence_novelty_weight(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


class SPRSeqDataset(Dataset):
    def __init__(self, sequences, labels):
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
    return {
        "seq": seq_tensor,
        "len": torch.tensor(lengths, dtype=torch.long),
        "label": labels,
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
# 5. model                                              #
# ---------------------------------------------------- #
class GRUClassifier(nn.Module):
    def __init__(self, n_clusters, pad_idx, emb_dim=64, hid=128, num_classes=10):
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
# 6. evaluation helper                                  #
# ---------------------------------------------------- #
def evaluate(model, loader, seq_raw):
    model.eval()
    preds, gts, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            bseq = batch["seq"].to(device)
            blen = batch["len"].to(device)
            lbl = batch["label"].to(device)
            logits = model(bseq, blen)
            loss = criterion(logits, lbl)
            total_loss += loss.item() * lbl.size(0)
            p = logits.argmax(1)
            preds.extend(p.cpu().tolist())
            gts.extend(lbl.cpu().tolist())
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
# 7. training loop with early stopping                  #
# ---------------------------------------------------- #
max_epochs, patience = 25, 5
best_hcs, since_best, best_state = -1.0, 0, None

for epoch in range(1, max_epochs + 1):
    # training
    model.train()
    total_loss, n_seen = 0.0, 0
    for batch in train_loader:
        bseq, blen, lbl = (
            batch["seq"].to(device),
            batch["len"].to(device),
            batch["label"].to(device),
        )
        optimizer.zero_grad()
        logits = model(bseq, blen)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * lbl.size(0)
        n_seen += lbl.size(0)
    train_loss = total_loss / n_seen
    experiment_data["RandomClusterAssignment"]["SPR_BENCH"]["losses"]["train"].append(
        (epoch, train_loss)
    )

    # validation
    val_stats = evaluate(model, dev_loader, spr["dev"]["sequence"])
    experiment_data["RandomClusterAssignment"]["SPR_BENCH"]["losses"]["val"].append(
        (epoch, val_stats["loss"])
    )
    experiment_data["RandomClusterAssignment"]["SPR_BENCH"]["metrics"]["val"].append(
        (
            epoch,
            val_stats["CWA"],
            val_stats["SWA"],
            val_stats["HCSA"],
            val_stats["SNWA"],
        )
    )
    print(
        f'Epoch {epoch}: val_loss={val_stats["loss"]:.4f} HCSA={val_stats["HCSA"]:.3f} SNWA={val_stats["SNWA"]:.3f}'
    )

    # early stopping
    if val_stats["HCSA"] > best_hcs + 1e-6:
        best_hcs, best_state, since_best = (
            val_stats["HCSA"],
            copy.deepcopy(model.state_dict()),
            0,
        )
    else:
        since_best += 1
    if since_best >= patience:
        print("Early stopping.")
        break

if best_state:
    model.load_state_dict(best_state)

# ---------------------------------------------------- #
# 8. final evaluation                                   #
# ---------------------------------------------------- #
dev_final = evaluate(model, dev_loader, spr["dev"]["sequence"])
test_final = evaluate(model, test_loader, spr["test"]["sequence"])

exp = experiment_data["RandomClusterAssignment"]["SPR_BENCH"]
exp["predictions"]["dev"], exp["ground_truth"]["dev"] = (
    dev_final["preds"],
    dev_final["gts"],
)
exp["predictions"]["test"], exp["ground_truth"]["test"] = (
    test_final["preds"],
    test_final["gts"],
)

print(
    f'Dev  : CWA={dev_final["CWA"]:.3f} SWA={dev_final["SWA"]:.3f} HCSA={dev_final["HCSA"]:.3f} SNWA={dev_final["SNWA"]:.3f}'
)
print(
    f'Test : CWA={test_final["CWA"]:.3f} SWA={test_final["SWA"]:.3f} HCSA={test_final["HCSA"]:.3f} SNWA={test_final["SNWA"]:.3f}'
)

# ---------------------------------------------------- #
# 9. save experiment data                               #
# ---------------------------------------------------- #
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Saved experiment data to {working_dir}/experiment_data.npy")
