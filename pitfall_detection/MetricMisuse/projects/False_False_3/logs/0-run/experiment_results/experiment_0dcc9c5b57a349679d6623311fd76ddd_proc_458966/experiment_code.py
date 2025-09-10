import os, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------- mandatory boilerplate ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- data utilities ----------
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


# ---------- vocabulary for shapes / colours ----------
class SCVocab:
    def __init__(self, dataset):
        shapes = set()
        colors = set()
        for seq in dataset["train"]["sequence"]:
            for tok in seq.split():
                shapes.add(tok[0])
                colors.add(tok[1] if len(tok) > 1 else "-")
        self.sitos = ["<pad>"] + sorted(shapes)
        self.citos = ["<pad>", "-"] + sorted(colors - {"", "-"})
        self.sstoi = {t: i for i, t in enumerate(self.sitos)}
        self.cstoi = {t: i for i, t in enumerate(self.citos)}

    def encode(self, seq, max_len):
        s_ids, c_ids = [], []
        for tok in seq.split()[:max_len]:
            s_ids.append(self.sstoi.get(tok[0], 0))
            c_ids.append(self.cstoi.get(tok[1] if len(tok) > 1 else "-", 0))
        pad = max_len - len(s_ids)
        s_ids += [0] * pad
        c_ids += [0] * pad
        mask = [1] * (len(s_ids) - pad) + [0] * pad
        return s_ids, c_ids, mask

    @property
    def n_shapes(self):
        return len(self.sitos)

    @property
    def n_colors(self):
        return len(self.citos)


# ---------- model ----------
class NeuralSymbolicTransformer(nn.Module):
    def __init__(
        self, n_shape, n_color, d_model, symb_dim, n_cls, n_heads=4, n_layers=2
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, 16, padding_idx=0)
        self.color_emb = nn.Embedding(n_color, 8, padding_idx=0)
        self.proj = nn.Linear(24, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.symb_fc = nn.Sequential(nn.Linear(symb_dim, 16), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 16, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )

    def forward(self, s_ids, c_ids, mask, symb_feats):
        x = torch.cat(
            [self.shape_emb(s_ids), self.color_emb(c_ids)], dim=-1
        )  # [B,L,24]
        x = self.proj(x) + self.pos_emb[:, : x.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~mask.bool())
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)  # mean pool
        symb = self.symb_fc(symb_feats)
        out = self.classifier(torch.cat([x, symb], dim=-1))
        return out


# ---------- load data ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
vocab = SCVocab(spr)

labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}

MAX_LEN = 40


def collate(batch):
    s_ids, c_ids, mask, lab, sfeat = [], [], [], [], []
    for ex in batch:
        s, c, m = vocab.encode(ex["sequence"], MAX_LEN)
        s_ids.append(s)
        c_ids.append(c)
        mask.append(m)
        ln = len(ex["sequence"].split())
        sv, cv = count_shape_variety(ex["sequence"]), count_color_variety(
            ex["sequence"]
        )
        sfeat.append([sv, cv, ln, sv / (ln + 1e-6), cv / (ln + 1e-6)])
        lab.append(lab2id[ex["label"]])
    return (
        torch.tensor(s_ids).to(device),
        torch.tensor(c_ids).to(device),
        torch.tensor(mask, dtype=torch.bool).to(device),
        torch.tensor(sfeat, dtype=torch.float32).to(device),
        torch.tensor(lab).to(device),
    )


train_loader = DataLoader(
    spr["train"], batch_size=256, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(spr["dev"], batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(spr["test"], batch_size=256, shuffle=False, collate_fn=collate)

# ---------- experiment container ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------- model / optimiser / loss ----------
model = NeuralSymbolicTransformer(
    vocab.n_shapes, vocab.n_colors, d_model=32, symb_dim=5, n_cls=len(labels)
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# ---------- training ----------
best_swa = 0
patience = 3
no_imp = 0
EPOCHS = 15
for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0
    for s, c, m, feat, l in train_loader:
        optimizer.zero_grad()
        out = model(s, c, m, feat)
        loss = criterion(out, l)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * l.size(0)
    tr_loss /= len(spr["train"])
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # ---- validation
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []
    seqs = []
    with torch.no_grad():
        for (s, c, m, feat, l), idx in zip(dev_loader, range(len(dev_loader))):
            out = model(s, c, m, feat)
            loss = criterion(out, l)
            val_loss += loss.item() * l.size(0)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend([id2lab[p] for p in preds])
            y_true.extend([id2lab[i] for i in l.cpu().tolist()])
            seqs.extend(spr["dev"]["sequence"][idx * 256 : idx * 256 + l.size(0)])
    val_loss /= len(spr["dev"])
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(swa)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA = {swa:.4f}")
    if swa > best_swa:
        best_swa = swa
        no_imp = 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
    else:
        no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break

# ---------- testing ----------
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
model.eval()
y_true = []
y_pred = []
seqs = []
with torch.no_grad():
    for (s, c, m, feat, l), idx in zip(test_loader, range(len(test_loader))):
        out = model(s, c, m, feat)
        preds = out.argmax(1).cpu().tolist()
        y_pred.extend([id2lab[p] for p in preds])
        y_true.extend([id2lab[i] for i in l.cpu().tolist()])
        seqs.extend(spr["test"]["sequence"][idx * 256 : idx * 256 + l.size(0)])
test_swa = shape_weighted_accuracy(seqs, y_true, y_pred)
print(f"Test Shape-Weighted Accuracy (SWA): {test_swa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
