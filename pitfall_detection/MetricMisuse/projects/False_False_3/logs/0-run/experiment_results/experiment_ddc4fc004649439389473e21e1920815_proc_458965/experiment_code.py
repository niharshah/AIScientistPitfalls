import os, pathlib, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------- mandatory boilerplate ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- SPR helpers ----------
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
    w = [count_shape_variety(s) for s in seqs]
    return sum(v if t == p else 0 for v, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------- load data ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# build separate vocabularies for shapes and colours
all_shapes = {tok[0] for tok in spr["train"]["sequence"] for tok in tok.split()}
all_colors = {
    tok[1] for tok in spr["train"]["sequence"] for tok in tok.split() if len(tok) > 1
}
shape2id = {s: i + 2 for i, s in enumerate(sorted(all_shapes))}
color2id = {c: i + 2 for i, c in enumerate(sorted(all_colors))}
shape2id["<pad>"] = 0
shape2id["<unk>"] = 1
color2id["<pad>"] = 0
color2id["<unk>"] = 1

labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# ---------- tensorise ----------
MAX_LEN = 40


def seq_to_tensor(seq):
    shapes, colours = [], []
    for tok in seq.split()[:MAX_LEN]:
        shapes.append(shape2id.get(tok[0], 1))
        colours.append(color2id.get(tok[1] if len(tok) > 1 else "<unk>", 1))
    pad = MAX_LEN - len(shapes)
    shapes.extend([0] * pad)
    colours.extend([0] * pad)
    mask = [1] * (MAX_LEN - pad) + [0] * pad
    return shapes, colours, mask


def collate(batch):
    s_ids, c_ids, masks, feats, labels_t = [], [], [], [], []
    for ex in batch:
        s, c, m = seq_to_tensor(ex["sequence"])
        ln = sum(m)
        sv = count_shape_variety(ex["sequence"])
        cv = count_color_variety(ex["sequence"])
        feats.append([sv, cv, ln, sv / (ln + 1e-6)])
        s_ids.append(s)
        c_ids.append(c)
        masks.append(m)
        labels_t.append(label2id[ex["label"]])
    to_tensor = lambda x, tp=torch.long: torch.tensor(x, dtype=tp).to(device)
    return (
        to_tensor(s_ids),
        to_tensor(c_ids),
        to_tensor(masks, torch.float),
        to_tensor(feats, torch.float),
        to_tensor(labels_t),
    )


batch_size = 256
train_loader = DataLoader(
    spr["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    spr["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    spr["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class HybridTransformer(nn.Module):
    def __init__(self, n_shapes, n_colors, d_model=64, n_cls=10, symb_dim=4):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(n_colors, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Sequential(
            nn.Linear(d_model + symb_dim, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )

    def forward(self, s_ids, c_ids, mask, symb_feats):
        x = self.shape_emb(s_ids) + self.color_emb(c_ids)
        x = self.encoder(x, src_key_padding_mask=(1 - mask).bool())
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        out = torch.cat([pooled, symb_feats], dim=-1)
        return self.cls(out)


# ---------- experiment container ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------- training setup ----------
model = HybridTransformer(
    len(shape2id), len(color2id), d_model=64, n_cls=len(labels), symb_dim=4
).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# ---------- training loop ----------
best_swa, patience, no_improve, EPOCHS = 0.0, 3, 0, 15
for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    running_loss = 0.0
    for s_ids, c_ids, mask, feats, tgt in train_loader:
        optimizer.zero_grad()
        logits = model(s_ids, c_ids, mask, feats)
        loss = criterion(logits, tgt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * tgt.size(0)
    train_loss = running_loss / len(spr["train"])
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # validate
    model.eval()
    val_loss, y_true, y_pred, seqs_dev = 0.0, [], [], []
    with torch.no_grad():
        for idx, data in enumerate(dev_loader):
            s_ids, c_ids, mask, feats, tgt = data
            logits = model(s_ids, c_ids, mask, feats)
            val_loss += criterion(logits, tgt).item() * tgt.size(0)
            preds = logits.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[y] for y in tgt.cpu().tolist()])
            seqs_dev.extend(
                spr["dev"]["sequence"][
                    idx * batch_size : idx * batch_size + tgt.size(0)
                ]
            )
    val_loss /= len(spr["dev"])
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    val_swa = shape_weighted_accuracy(seqs_dev, y_true, y_pred)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | SWA={val_swa:.4f}"
    )
    scheduler.step()

    if val_swa > best_swa:
        best_swa, no_improve = val_swa, 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# ---------- test evaluation ----------
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
model.eval()
y_true_test, y_pred_test, seqs_test = [], [], []
with torch.no_grad():
    for idx, data in enumerate(test_loader):
        s_ids, c_ids, mask, feats, tgt = data
        logits = model(s_ids, c_ids, mask, feats)
        preds = logits.argmax(1).cpu().tolist()
        y_pred_test.extend([id2label[p] for p in preds])
        y_true_test.extend([id2label[y] for y in tgt.cpu().tolist()])
        seqs_test.extend(
            spr["test"]["sequence"][idx * batch_size : idx * batch_size + tgt.size(0)]
        )
test_swa = shape_weighted_accuracy(seqs_test, y_true_test, y_pred_test)
print(f"Test Shape-Weighted Accuracy (SWA): {test_swa:.4f}")

# save results
experiment_data["SPR_BENCH"]["predictions"] = y_pred_test
experiment_data["SPR_BENCH"]["ground_truth"] = y_true_test
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
