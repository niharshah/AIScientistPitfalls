import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- experiment dict ----------
experiment_data = {
    "NoShapeEmb": {
        "SPR_BENCH": {
            "metrics": {"train_swa": [], "val_swa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}
exp_rec = experiment_data["NoShapeEmb"]["SPR_BENCH"]

# ---------------- working dir --------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- load dataset -------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
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


# ---------------- vocab build --------------
def build_shape_color_sets(data):
    shapes, colors = set(), set()
    for seq in data["sequence"]:
        for tok in seq.strip().split():
            shapes.add(tok[0])
            colors.add(tok[1] if len(tok) > 1 else "<none>")
    shapes = {"<pad>": 0, **{s: i + 1 for i, s in enumerate(sorted(shapes))}}
    colors = {"<pad>": 0, **{c: i + 1 for i, c in enumerate(sorted(colors))}}
    return shapes, colors


shape_map, color_map = build_shape_color_sets(spr["train"])
print("n_shapes", len(shape_map), "n_colors", len(color_map))

n_shape_sym = len(shape_map) - 1
n_color_sym = len(color_map) - 1
sym_dim = n_shape_sym + n_color_sym


# ---------------- dataset ------------------
class SPRDataset(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.labels = split["label"]

    def encode_tok(self, tok):
        s_id = shape_map.get(tok[0], 0)
        c_id = color_map.get(tok[1] if len(tok) > 1 else "<none>", 0)
        return s_id, c_id

    def sym_vec(self, seq):
        s_vec = np.zeros(n_shape_sym, np.float32)
        c_vec = np.zeros(n_color_sym, np.float32)
        toks = seq.strip().split()
        for t in toks:
            if t:
                si = shape_map.get(t[0], 0) - 1
                if si >= 0:
                    s_vec[si] += 1
                ci = color_map.get(t[1] if len(t) > 1 else "<none>", 0) - 1
                if ci >= 0:
                    c_vec[ci] += 1
        denom = max(len(toks), 1)
        return np.concatenate([s_vec, c_vec]) / denom

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s_ids, c_ids = zip(*[self.encode_tok(t) for t in self.seq[idx].strip().split()])
        return {
            "shape_ids": torch.tensor(s_ids, dtype=torch.long),
            "color_ids": torch.tensor(c_ids, dtype=torch.long),
            "sym": torch.tensor(self.sym_vec(self.seq[idx]), dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_str": self.seq[idx],
        }


train_ds, dev_ds, test_ds = (
    SPRDataset(spr["train"]),
    SPRDataset(spr["dev"]),
    SPRDataset(spr["test"]),
)


# ---------------- collate ------------------
def collate(batch):
    shape = nn.utils.rnn.pad_sequence(
        [b["shape_ids"] for b in batch], batch_first=True, padding_value=0
    )
    color = nn.utils.rnn.pad_sequence(
        [b["color_ids"] for b in batch], batch_first=True, padding_value=0
    )
    mask = shape != 0
    sym = torch.stack([b["sym"] for b in batch])
    lab = torch.stack([b["label"] for b in batch])
    seqs = [b["seq_str"] for b in batch]
    return {
        "shape_ids": shape,
        "color_ids": color,
        "mask": mask,
        "sym": sym,
        "label": lab,
        "seqs": seqs,
    }


BATCH = 256
train_loader = DataLoader(train_ds, BATCH, True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, BATCH, False, collate_fn=collate)
test_loader = DataLoader(test_ds, BATCH, False, collate_fn=collate)


# ---------------- metrics ------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def shape_weighted_accuracy(seqs, y, g):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if yy == pp else 0 for wt, yy, pp in zip(w, y, g)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------------- model --------------------
class NeuralSymbolicTransformer(nn.Module):
    def __init__(
        self,
        n_shape,
        n_color,
        sym_dim,
        num_cls,
        d_model=64,
        nhead=4,
        layers=2,
        max_len=64,
        use_shape_emb=False,
    ):
        super().__init__()
        # keep a frozen shape embedding just for API completeness
        self.use_shape_emb = use_shape_emb
        self.shape_emb = nn.Embedding(n_shape, d_model, padding_idx=0)
        for p in self.shape_emb.parameters():
            p.requires_grad = False
        self.color_emb = nn.Embedding(n_color, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + sym_dim, 128), nn.ReLU(), nn.Linear(128, num_cls)
        )

    def forward(self, shape_ids, color_ids, mask, sym):
        B, T = shape_ids.shape
        pos = torch.arange(T, device=shape_ids.device).unsqueeze(0).expand(B, T)
        tok = self.color_emb(color_ids) + self.pos_emb(pos)
        if self.use_shape_emb:
            tok = tok + self.shape_emb(shape_ids)
        enc = self.encoder(tok, src_key_padding_mask=~mask)
        seq_vec = (enc * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(
            min=1e-6
        )
        return self.mlp(torch.cat([seq_vec, sym], -1))


num_classes = int(max(train_ds.labels)) + 1
model = NeuralSymbolicTransformer(
    len(shape_map), len(color_map), sym_dim, num_classes, use_shape_emb=False
).to(device)

crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], 1e-3)


# ---------------- evaluation ---------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tot, preds, gts, seqs = 0.0, [], [], []
    for b in loader:
        b = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in b.items()
        }
        out = model(b["shape_ids"], b["color_ids"], b["mask"], b["sym"])
        loss = crit(out, b["label"])
        tot += loss.item() * b["label"].size(0)
        p = out.argmax(-1).cpu().tolist()
        preds += p
        gts += b["label"].cpu().tolist()
        seqs += b["seqs"]
    swa = shape_weighted_accuracy(seqs, gts, preds)
    return tot / len(loader.dataset), swa, preds, gts


# ---------------- train --------------------
best_swa, best_state, no_imp = -1.0, None, 0
MAX_EPOCHS, PATIENCE = 20, 4

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for b in train_loader:
        b = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in b.items()
        }
        opt.zero_grad()
        out = model(b["shape_ids"], b["color_ids"], b["mask"], b["sym"])
        loss = crit(out, b["label"])
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * b["label"].size(0)
    tr_loss = epoch_loss / len(train_loader.dataset)
    tr_eval_loss, tr_swa, _, _ = evaluate(train_loader)
    val_loss, val_swa, _, _ = evaluate(dev_loader)

    exp_rec["losses"]["train"].append(tr_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["metrics"]["train_swa"].append(tr_swa)
    exp_rec["metrics"]["val_swa"].append(val_swa)
    exp_rec["timestamps"].append(time.time())

    print(f"Ep{epoch:02d} val_loss={val_loss:.4f}  val_SWA={val_swa:.4f}")
    if val_swa > best_swa:
        best_swa = val_swa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print("Early stop")
            break

# ---------------- test ---------------------
model.load_state_dict(best_state)
test_loss, test_swa, preds, gts = evaluate(test_loader)
print(f"TEST loss={test_loss:.4f}  SWA={test_swa:.4f}")

exp_rec["predictions"] = np.array(preds)
exp_rec["ground_truth"] = np.array(gts)

# ---------------- save ---------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
