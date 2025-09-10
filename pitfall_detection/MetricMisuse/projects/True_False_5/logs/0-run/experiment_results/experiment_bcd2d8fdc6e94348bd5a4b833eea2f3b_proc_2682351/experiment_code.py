import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- experiment store ---------------
experiment_data = {
    "SYM_ONLY": {
        "SPR_BENCH": {
            "metrics": {"train_swa": [], "val_swa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}
exp_rec = experiment_data["SYM_ONLY"]["SPR_BENCH"]

# ---------- device & working dir -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
print(f"Using device: {device}")


# ---------- dataset loader -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- helpers ------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def build_shape_color_sets(dataset):
    shapes, colors = set(), set()
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok:
                shapes.add(tok[0])
                if len(tok) > 1:
                    colors.add(tok[1])
    colors.add("<none>")
    shapes = {"<pad>": 0, **{ch: i + 1 for i, ch in enumerate(sorted(shapes))}}
    colors = {"<pad>": 0, **{ch: i + 1 for i, ch in enumerate(sorted(colors))}}
    return shapes, colors


shape_map, color_map = build_shape_color_sets(spr["train"])
n_shape_sym = len({k for k in shape_map if k != "<pad>"})
n_color_sym = len({k for k in color_map if k != "<pad>"})
sym_dim = n_shape_sym + n_color_sym
print(f"sym_dim={sym_dim}")


# ---------- dataset class ------------------
class SPRDataset(Dataset):
    def __init__(self, split, shape_map, color_map):
        self.seq = split["sequence"]
        self.labels = split["label"]
        self.shape_map = shape_map
        self.color_map = color_map
        self.n_shape_sym = n_shape_sym
        self.n_color_sym = n_color_sym

    def symbolic_vec(self, seq):
        s_vec = np.zeros(self.n_shape_sym, dtype=np.float32)
        c_vec = np.zeros(self.n_color_sym, dtype=np.float32)
        for tok in seq.strip().split():
            if tok:
                if tok[0] in self.shape_map and tok[0] != "<pad>":
                    s_vec[self.shape_map[tok[0]] - 1] += 1
                if len(tok) > 1 and tok[1] in self.color_map and tok[1] != "<pad>":
                    c_vec[self.color_map[tok[1]] - 1] += 1
        total = max(len(seq.strip().split()), 1)
        return np.concatenate([s_vec, c_vec]) / total

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "sym_feats": torch.tensor(
                self.symbolic_vec(self.seq[idx]), dtype=torch.float32
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sequence_str": self.seq[idx],
        }


train_ds = SPRDataset(spr["train"], shape_map, color_map)
dev_ds = SPRDataset(spr["dev"], shape_map, color_map)
test_ds = SPRDataset(spr["test"], shape_map, color_map)


def collate_fn(batch):
    sym = torch.stack([b["sym_feats"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    seqs = [b["sequence_str"] for b in batch]
    return {"sym_feats": sym, "labels": labels, "sequence_str": seqs}


BATCH = 256
train_loader = DataLoader(
    train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(dev_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(
    test_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn
)


# ---------- symbolic-only model ------------
class SymbolicOnlyModel(nn.Module):
    def __init__(self, sym_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sym_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, sym_feats):
        return self.mlp(sym_feats)


num_classes = int(max(train_ds.labels)) + 1
model = SymbolicOnlyModel(sym_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- evaluation ---------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tot_loss, preds, gts, seqs = 0.0, [], [], []
    for batch in loader:
        sym = batch["sym_feats"].to(device)
        labels = batch["labels"].to(device)
        logits = model(sym)
        loss = criterion(logits, labels)
        tot_loss += loss.item() * labels.size(0)
        p = logits.argmax(-1).cpu().tolist()
        preds.extend(p)
        gts.extend(batch["labels"].cpu().tolist())
        seqs.extend(batch["sequence_str"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    return tot_loss / len(loader.dataset), swa, preds, gts, seqs


# ---------- training loop ------------------
MAX_EPOCHS, patience = 20, 4
best_val_swa, best_state, no_imp = -1.0, None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        sym = batch["sym_feats"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(sym)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * labels.size(0)
    train_loss = epoch_loss / len(train_loader.dataset)
    train_loss_eval, train_swa, _, _, _ = evaluate(train_loader)
    val_loss, val_swa, _, _, _ = evaluate(dev_loader)

    exp_rec["losses"]["train"].append(train_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["metrics"]["train_swa"].append(train_swa)
    exp_rec["metrics"]["val_swa"].append(val_swa)
    exp_rec["timestamps"].append(time.time())

    print(f"Epoch {epoch:02d}: val_loss={val_loss:.4f}  val_SWA={val_swa:.4f}")

    if val_swa > best_val_swa:
        best_val_swa, best_state, no_imp = (
            val_swa,
            {k: v.cpu() for k, v in model.state_dict().items()},
            0,
        )
    else:
        no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break

# ---------- test ---------------------------
model.load_state_dict(best_state)
test_loss, test_swa, test_preds, test_gts, _ = evaluate(test_loader)
print(f"TEST loss={test_loss:.4f}  SWA={test_swa:.4f}")

exp_rec["predictions"] = np.array(test_preds)
exp_rec["ground_truth"] = np.array(test_gts)

# ---------- save ---------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
