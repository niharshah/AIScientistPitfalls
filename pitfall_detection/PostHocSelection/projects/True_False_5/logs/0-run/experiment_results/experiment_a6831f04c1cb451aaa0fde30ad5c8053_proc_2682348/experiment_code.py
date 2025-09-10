import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- experiment bookkeeping ----------
experiment_data = {
    "NoSymbolicVector": {
        "SPR_BENCH": {
            "metrics": {"train_swa": [], "val_swa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}
exp_rec = experiment_data["NoSymbolicVector"]["SPR_BENCH"]

# ---------- device --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper: load SPR-BENCH ----------
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


# ---------- vocab construction --------------
def build_shape_color_sets(dataset):
    shapes, colors = set(), set()
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok:
                shapes.add(tok[0])
                if len(tok) > 1:
                    colors.add(tok[1])
    colors.add("<none>")
    shapes = {"<pad>": 0, **{s: i + 1 for i, s in enumerate(sorted(shapes))}}
    colors = {"<pad>": 0, **{c: i + 1 for i, c in enumerate(sorted(colors))}}
    return shapes, colors


shape_map, color_map = build_shape_color_sets(spr["train"])
print(f"n_shapes={len(shape_map)}  n_colors={len(color_map)}")


# ---------- metric helper -------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ---------- torch Dataset -------------------
class SPRDataset(Dataset):
    def __init__(self, split, shape_map, color_map):
        self.seq = split["sequence"]
        self.labels = split["label"]
        self.shape_map, self.color_map = shape_map, color_map

    def encode_token(self, tok):
        s_idx = self.shape_map.get(tok[0], self.shape_map["<pad>"])
        if len(tok) > 1:
            c_idx = self.color_map.get(tok[1], self.color_map["<pad>"])
        else:
            c_idx = self.color_map.get("<none>", self.color_map["<pad>"])
        return s_idx, c_idx

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq_str = self.seq[idx]
        shape_ids, color_ids = zip(
            *[self.encode_token(t) for t in seq_str.strip().split()]
        )
        return {
            "shape_ids": torch.tensor(shape_ids, dtype=torch.long),
            "color_ids": torch.tensor(color_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sequence_str": seq_str,
        }


train_ds = SPRDataset(spr["train"], shape_map, color_map)
dev_ds = SPRDataset(spr["dev"], shape_map, color_map)
test_ds = SPRDataset(spr["test"], shape_map, color_map)


# ---------- collate -------------------------
def collate_fn(batch):
    shapes = [b["shape_ids"] for b in batch]
    colors = [b["color_ids"] for b in batch]
    pad_shape = nn.utils.rnn.pad_sequence(shapes, batch_first=True, padding_value=0)
    pad_color = nn.utils.rnn.pad_sequence(colors, batch_first=True, padding_value=0)
    mask = pad_shape != 0
    labels = torch.stack([b["labels"] for b in batch])
    seqs = [b["sequence_str"] for b in batch]
    return {
        "shape_ids": pad_shape,
        "color_ids": pad_color,
        "attention_mask": mask,
        "labels": labels,
        "sequence_str": seqs,
    }


BATCH = 256
train_loader = DataLoader(
    train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(dev_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(
    test_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn
)


# ---------- model w/o symbolic vector -------
class NoSymTransformer(nn.Module):
    def __init__(
        self,
        n_shape,
        n_color,
        num_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_len=64,
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(n_color, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, shape_ids, color_ids, attn_mask):
        B, T = shape_ids.size()
        pos = torch.arange(T, device=shape_ids.device).unsqueeze(0).expand(B, T)
        x = self.shape_emb(shape_ids) + self.color_emb(color_ids) + self.pos_emb(pos)
        enc = self.encoder(x, src_key_padding_mask=~attn_mask)
        seq_emb = (enc * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(
            1, keepdim=True
        ).clamp(min=1e-6)
        return self.mlp(seq_emb)


num_classes = int(max(train_ds.labels)) + 1
model = NoSymTransformer(len(shape_map), len(color_map), num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- evaluation ----------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tot_loss, preds, gts, seqs = 0.0, [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["shape_ids"], batch["color_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        tot_loss += loss.item() * batch["labels"].size(0)
        p = logits.argmax(-1).cpu().tolist()
        g = batch["labels"].cpu().tolist()
        preds.extend(p)
        gts.extend(g)
        seqs.extend(batch["sequence_str"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    return tot_loss / len(loader.dataset), swa, preds, gts, seqs


# ---------- training ------------------------
MAX_EPOCHS, patience = 20, 4
best_val_swa, best_state, no_imp = -1.0, None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["shape_ids"], batch["color_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * batch["labels"].size(0)
    train_loss = ep_loss / len(train_loader.dataset)
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

# ---------- test ----------------------------
model.load_state_dict(best_state)
test_loss, test_swa, test_preds, test_gts, _ = evaluate(test_loader)
print(f"TEST loss={test_loss:.4f}  SWA={test_swa:.4f}")

exp_rec["predictions"] = np.array(test_preds)
exp_rec["ground_truth"] = np.array(test_gts)

# ---------- save ----------------------------
os.makedirs("working", exist_ok=True)
np.save(os.path.join("working", "experiment_data.npy"), experiment_data)
