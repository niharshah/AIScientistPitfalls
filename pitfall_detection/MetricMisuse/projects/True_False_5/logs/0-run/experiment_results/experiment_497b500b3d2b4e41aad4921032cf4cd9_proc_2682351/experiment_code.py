"""
Ablation: No-Transformer-Encoder (Bag-of-Embeddings) for SPR_BENCH
This script is self-contained and mirrors the baseline pipeline, but the model
has no self-attention layers: token embeddings (shape+color+position) are
mean-pooled and fed—together with symbolic features—into an MLP classifier.
"""

import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- working dir & experiment record -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "NoTransformerEncoder": {
        "SPR_BENCH": {
            "metrics": {"train_swa": [], "val_swa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}
exp_rec = experiment_data["NoTransformerEncoder"]["SPR_BENCH"]

# ---------- device ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset loader -------------------------------------------------
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


# ---------- helper: accuracy ----------------------------------------------
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum((wi if t == p else 0) for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------- vocab build ----------------------------------------------------
def build_shape_color_sets(dataset):
    shapes, colors = set(), set()
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok:
                shapes.add(tok[0])
                if len(tok) > 1:
                    colors.add(tok[1])
    colors.add("<none>")
    shape_map = {"<pad>": 0, **{s: i + 1 for i, s in enumerate(sorted(shapes))}}
    color_map = {"<pad>": 0, **{c: i + 1 for i, c in enumerate(sorted(colors))}}
    return shape_map, color_map


shape_map, color_map = build_shape_color_sets(spr["train"])
n_shape_sym = len(shape_map) - 1
n_color_sym = len(color_map) - 1
sym_dim = n_shape_sym + n_color_sym
print(f"n_shapes={len(shape_map)}  n_colors={len(color_map)}")


# ---------- torch dataset --------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, split, shape_map, color_map):
        self.seq = split["sequence"]
        self.labels = split["label"]
        self.shape_map = shape_map
        self.color_map = color_map
        self.n_shape_sym = n_shape_sym
        self.n_color_sym = n_color_sym

    def encode_token(self, tok):
        s_idx = self.shape_map.get(tok[0], 0)
        c_idx = (
            self.color_map.get(tok[1], self.color_map["<none>"])
            if len(tok) > 1
            else self.color_map["<none>"]
        )
        return s_idx, c_idx

    def symbolic_vec(self, seq):
        s_vec = np.zeros(self.n_shape_sym, np.float32)
        c_vec = np.zeros(self.n_color_sym, np.float32)
        toks = seq.strip().split()
        for tok in toks:
            if tok:
                if tok[0] in self.shape_map and tok[0] != "<pad>":
                    s_vec[self.shape_map[tok[0]] - 1] += 1
                if len(tok) > 1 and tok[1] in self.color_map and tok[1] != "<pad>":
                    c_vec[self.color_map[tok[1]] - 1] += 1
        total = max(len(toks), 1)
        return np.concatenate([s_vec, c_vec]) / total

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq_str = self.seq[idx]
        shape_ids, color_ids = zip(
            *(self.encode_token(tok) for tok in seq_str.strip().split())
        )
        return {
            "shape_ids": torch.tensor(shape_ids, dtype=torch.long),
            "color_ids": torch.tensor(color_ids, dtype=torch.long),
            "sym_feats": torch.tensor(self.symbolic_vec(seq_str), dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sequence_str": seq_str,
        }


train_ds = SPRDataset(spr["train"], shape_map, color_map)
dev_ds = SPRDataset(spr["dev"], shape_map, color_map)
test_ds = SPRDataset(spr["test"], shape_map, color_map)


# ---------- collate --------------------------------------------------------
def collate_fn(batch):
    sh = nn.utils.rnn.pad_sequence(
        [b["shape_ids"] for b in batch], batch_first=True, padding_value=0
    )
    co = nn.utils.rnn.pad_sequence(
        [b["color_ids"] for b in batch], batch_first=True, padding_value=0
    )
    mask = sh != 0
    return {
        "shape_ids": sh,
        "color_ids": co,
        "attention_mask": mask,
        "sym_feats": torch.stack([b["sym_feats"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "sequence_str": [b["sequence_str"] for b in batch],
    }


BATCH = 256
train_loader = DataLoader(train_ds, BATCH, True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, BATCH, False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, BATCH, False, collate_fn=collate_fn)


# ---------- model: Bag-of-Embeddings --------------------------------------
class NeuralSymbolicBagEncoder(nn.Module):
    def __init__(self, n_shape, n_color, sym_dim, num_classes, d_model=64, max_len=64):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(n_color, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + sym_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, shape_ids, color_ids, attn_mask, sym_feats):
        B, T = shape_ids.shape
        pos = torch.arange(T, device=shape_ids.device).unsqueeze(0).expand(B, T)
        tok_emb = (
            self.shape_emb(shape_ids) + self.color_emb(color_ids) + self.pos_emb(pos)
        )
        masked = tok_emb * attn_mask.unsqueeze(-1)
        seq_emb = masked.sum(1) / attn_mask.sum(1, keepdim=True).clamp(min=1e-6)
        return self.mlp(torch.cat([seq_emb, sym_feats], dim=-1))


num_classes = int(max(train_ds.labels)) + 1
model = NeuralSymbolicBagEncoder(
    len(shape_map), len(color_map), sym_dim, num_classes
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- evaluation -----------------------------------------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, preds, gts, seqs = 0.0, [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(
            batch["shape_ids"],
            batch["color_ids"],
            batch["attention_mask"],
            batch["sym_feats"],
        )
        loss = criterion(logits, batch["labels"])
        total_loss += loss.item() * batch["labels"].size(0)
        p = logits.argmax(-1).cpu().tolist()
        g = batch["labels"].cpu().tolist()
        preds.extend(p)
        gts.extend(g)
        seqs.extend(batch["sequence_str"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    return total_loss / len(loader.dataset), swa, preds, gts


# ---------- training loop --------------------------------------------------
MAX_EPOCHS, patience = 20, 4
best_val_swa, best_state, no_improve = -1.0, None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(
            batch["shape_ids"],
            batch["color_ids"],
            batch["attention_mask"],
            batch["sym_feats"],
        )
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["labels"].size(0)
    train_loss = epoch_loss / len(train_loader.dataset)
    tr_loss_eval, tr_swa, _, _ = evaluate(train_loader)
    val_loss, val_swa, _, _ = evaluate(dev_loader)

    exp_rec["losses"]["train"].append(train_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["metrics"]["train_swa"].append(tr_swa)
    exp_rec["metrics"]["val_swa"].append(val_swa)
    exp_rec["timestamps"].append(time.time())

    print(f"Epoch {epoch:02d}: val_loss={val_loss:.4f}  val_SWA={val_swa:.4f}")

    if val_swa > best_val_swa:
        best_val_swa, best_state, no_improve = (
            val_swa,
            {k: v.cpu() for k, v in model.state_dict().items()},
            0,
        )
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# ---------- test -----------------------------------------------------------
model.load_state_dict(best_state)
test_loss, test_swa, test_preds, test_gts = evaluate(test_loader)
print(f"TEST loss={test_loss:.4f}  SWA={test_swa:.4f}")

exp_rec["predictions"] = np.array(test_preds)
exp_rec["ground_truth"] = np.array(test_gts)

# ---------- save -----------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
