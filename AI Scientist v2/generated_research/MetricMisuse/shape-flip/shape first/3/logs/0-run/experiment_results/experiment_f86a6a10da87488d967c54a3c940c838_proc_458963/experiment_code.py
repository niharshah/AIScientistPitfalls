import os, pathlib, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------- mandatory boilerplate ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset utilities ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


# ---------- paths ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- build vocabularies ----------
shapes = set()
colors = set()
for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        if tok:
            shapes.add(tok[0])
            colors.add(tok[1] if len(tok) > 1 else " ")  # blank for missing colour
shape2id = {s: i + 2 for i, s in enumerate(sorted(shapes))}  # 0 PAD, 1 CLS
color2id = {c: i + 2 for i, c in enumerate(sorted(colors))}
shape2id["<PAD>"] = 0
shape2id["<CLS>"] = 1
color2id["<PAD>"] = 0
color2id["<CLS>"] = 1
id2label = {i: l for i, l in enumerate(sorted(set(spr["train"]["label"])))}
label2id = {l: i for i, l in id2label.items()}

# ---------- hyperparams ----------
MAX_LEN = 50  # including CLS
EMB_DIM = 64
N_HEADS = 4
N_LAYERS = 2
BATCH_SIZE = 256
EPOCHS = 15
PATIENCE = 3
LR = 3e-4


# ---------- dataset -> tensors ----------
def seq_to_ids(seq: str, max_len: int):
    toks = seq.split()[: max_len - 1]
    shape_ids = [shape2id["<CLS>"]]
    color_ids = [color2id["<CLS>"]]
    for tok in toks:
        shape_ids.append(shape2id.get(tok[0], 0))
        color_ids.append(color2id.get(tok[1] if len(tok) > 1 else " ", 0))
    pad_len = max_len - len(shape_ids)
    shape_ids.extend([shape2id["<PAD>"]] * pad_len)
    color_ids.extend([color2id["<PAD>"]] * pad_len)
    attn_mask = [1] * len(toks + ["CLS"]) + [0] * pad_len
    return shape_ids, color_ids, attn_mask


def collate(batch):
    sh, co, mask, feat, lab, seqs = [], [], [], [], [], []
    for ex in batch:
        s_ids, c_ids, m = seq_to_ids(ex["sequence"], MAX_LEN)
        sv = count_shape_variety(ex["sequence"])
        cv = count_color_variety(ex["sequence"])
        ln = len(ex["sequence"].split())
        feat.append([sv, cv, ln, sv / (ln + 1e-6), cv / (ln + 1e-6)])
        sh.append(s_ids)
        co.append(c_ids)
        mask.append(m)
        lab.append(label2id[ex["label"]])
        seqs.append(ex["sequence"])
    return (
        torch.tensor(sh).to(device),
        torch.tensor(co).to(device),
        torch.tensor(mask, dtype=torch.bool).to(device),
        torch.tensor(feat, dtype=torch.float32).to(device),
        torch.tensor(lab).to(device),
        seqs,
    )


train_loader = DataLoader(
    spr["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    spr["dev"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    spr["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class ShapeColorTransformer(nn.Module):
    def __init__(self, n_shapes, n_colors, emb_dim, n_heads, n_layers, feat_dim, n_cls):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, emb_dim, padding_idx=0)
        self.color_emb = nn.Embedding(n_colors, emb_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(MAX_LEN, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim + feat_dim, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )

    def forward(self, shape_ids, color_ids, attn_mask, feats):
        emb = self.shape_emb(shape_ids) + self.color_emb(color_ids) + self.pos_emb
        out = self.transformer(emb, src_key_padding_mask=~attn_mask)
        cls_vec = out[:, 0]  # CLS token
        logits = self.fc(torch.cat([cls_vec, feats], dim=-1))
        return logits


model = ShapeColorTransformer(
    len(shape2id),
    len(color2id),
    EMB_DIM,
    N_HEADS,
    N_LAYERS,
    feat_dim=5,
    n_cls=len(label2id),
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------- experiment tracker ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------- training ----------
best_val_swa = 0.0
no_improve = 0
for epoch in range(1, EPOCHS + 1):
    # -- train
    model.train()
    running = 0.0
    for sh, co, msk, feat, lab, _ in train_loader:
        optimizer.zero_grad()
        logits = model(sh, co, msk, feat)
        loss = criterion(logits, lab)
        loss.backward()
        optimizer.step()
        running += loss.item() * lab.size(0)
    train_loss = running / len(spr["train"])
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # -- validate
    model.eval()
    val_loss, y_true, y_pred, seqs = 0.0, [], [], []
    with torch.no_grad():
        for sh, co, msk, feat, lab, seq in dev_loader:
            logits = model(sh, co, msk, feat)
            val_loss += criterion(logits, lab).item() * lab.size(0)
            preds = logits.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[t.item()] for t in lab.cpu()])
            seqs.extend(seq)
    val_loss /= len(spr["dev"])
    val_swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA = {val_swa:.4f}")
    scheduler.step()

    # early stopping
    if val_swa > best_val_swa:
        best_val_swa = val_swa
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("Early stopping.")
            break

# ---------- test ----------
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
model.eval()
y_true, y_pred, seqs = [], [], []
with torch.no_grad():
    for sh, co, msk, feat, lab, seq in test_loader:
        logits = model(sh, co, msk, feat)
        preds = logits.argmax(1).cpu().tolist()
        y_pred.extend([id2label[p] for p in preds])
        y_true.extend([id2label[t.item()] for t in lab.cpu()])
        seqs.extend(seq)
test_swa = shape_weighted_accuracy(seqs, y_true, y_pred)
print(f"Test Shape-Weighted Accuracy (SWA): {test_swa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
