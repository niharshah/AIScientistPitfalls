import os, pathlib, random, time, numpy as np, torch, warnings
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------- initial boilerplate ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- utility metrics ----------------
def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ---------------- data loading ----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({s: _ld(f"{s}.csv") for s in ["train", "dev", "test"]})


def get_dataset():
    try:
        root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        ds = load_spr_bench(root)
        print("Loaded official SPR_BENCH")
    except Exception as e:
        print("Failed official load, generating toy data.", e)
        shapes, colors = "ABCD", "abcd"

        def gen(n):
            data = [
                {
                    "id": i,
                    "sequence": " ".join(
                        random.choice(shapes) + random.choice(colors)
                        for _ in range(random.randint(3, 9))
                    ),
                    "label": random.choice(["yes", "no"]),
                }
                for i in range(n)
            ]
            return load_dataset("json", data_files={"train": data}, split="train")

        ds = DatasetDict()
        ds["train"], ds["dev"], ds["test"] = gen(2000), gen(500), gen(500)
    return ds


ds = get_dataset()

# ---------------- vocab creation ----------------
shapes = set()
colors = set()
labels = set()
for ex in ds["train"]:
    for tok in ex["sequence"].split():
        shapes.add(tok[0])
        colors.add(tok[1] if len(tok) > 1 else "#")
    labels.add(ex["label"])
shape2id = {s: i + 1 for i, s in enumerate(sorted(shapes))}
color2id = {c: i + 1 for i, c in enumerate(sorted(colors))}
label2id = {l: i for i, l in enumerate(sorted(labels))}
id2label = {v: k for k, v in label2id.items()}
num_shapes = len(shape2id) + 1
num_colors = len(color2id) + 1
num_classes = len(label2id)
print(f"{num_shapes-1} shapes, {num_colors-1} colors, {num_classes} classes")


# ---------------- torch dataset ----------------
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.lbl = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        shape_ids = [shape2id[t[0]] for t in self.seq[idx].split()]
        color_ids = [
            color2id[t[1] if len(t) > 1 else "#"] for t in self.seq[idx].split()
        ]
        return {
            "shape": torch.tensor(shape_ids),
            "color": torch.tensor(color_ids),
            "len": torch.tensor(len(shape_ids)),
            "n_shape": torch.tensor(count_shape_variety(self.seq[idx])),
            "label": torch.tensor(self.lbl[idx]),
            "raw": self.seq[idx],
        }


def collate(batch):
    maxlen = max(b["len"] for b in batch).item()
    shp = torch.zeros(len(batch), maxlen, dtype=torch.long)
    clr = torch.zeros(len(batch), maxlen, dtype=torch.long)
    msk = torch.zeros(len(batch), maxlen, dtype=torch.bool)
    for i, b in enumerate(batch):
        l = b["len"]
        shp[i, :l] = b["shape"]
        clr[i, :l] = b["color"]
        msk[i, :l] = 1
    return {
        "shape": shp,
        "color": clr,
        "mask": msk,
        "n_shape": torch.stack([b["n_shape"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw": [b["raw"] for b in batch],
    }


batch_size = 128
train_loader = DataLoader(
    SPRTorch(ds["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(ds["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(ds["test"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------------- model ----------------
class ShapeColorTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, nlayers=1, ff=128, dropout=0.1):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(num_colors, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(200, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, nlayers)
        self.fc = nn.Linear(d_model + 1, num_classes)  # +1 for n_shape (symbolic)

    def forward(self, shape, color, mask, n_shape):
        B, L = shape.size()
        pos_ids = torch.arange(L, device=shape.device).unsqueeze(0).expand(B, L)
        x = self.shape_emb(shape) + self.color_emb(color) + self.pos_emb(pos_ids)
        x = self.enc(x, src_key_padding_mask=~mask)
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        feats = torch.cat([x, n_shape.unsqueeze(-1).float()], -1)
        return self.fc(feats)


# ---------------- training helpers ----------------
def evaluate(model, loader):
    model.eval()
    all_y, all_p, all_seq, total_loss, ntok = [], [], [], 0, 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(bt["shape"], bt["color"], bt["mask"], bt["n_shape"])
            loss = ce(logits, bt["label"])
            total_loss += loss.item() * bt["label"].size(0)
            ntok += bt["label"].size(0)
            all_y.extend(bt["label"].cpu().numpy())
            all_p.extend(logits.argmax(-1).cpu().numpy())
            all_seq.extend(batch["raw"])
    return total_loss / ntok, shape_weighted_accuracy(all_seq, all_y, all_p)


# ---------------- hyperparameter sweep ----------------
search_space = [(64, 1), (64, 2), (128, 1)]
experiment_data = {"runs": {}}

for d_model, n_layers in search_space:
    tag = f"d{d_model}_l{n_layers}"
    print(f"\n=== Config {tag} ===")
    model = ShapeColorTransformer(d_model=d_model, nlayers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_val, pat, patience = float("inf"), 0, 3
    run = {
        "losses": {"train": [], "val": []},
        "metrics": {"val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    for epoch in range(1, 31):
        model.train()
        epoch_loss = 0
        nex = 0
        for batch in train_loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(bt["shape"], bt["color"], bt["mask"], bt["n_shape"])
            loss = ce(logits, bt["label"])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * bt["label"].size(0)
            nex += bt["label"].size(0)
        train_loss = epoch_loss / nex
        val_loss, val_swa = evaluate(model, dev_loader)
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}   SWA = {val_swa:.4f}")
        run["losses"]["train"].append(train_loss)
        run["losses"]["val"].append(val_loss)
        run["metrics"]["val"].append(val_swa)
        run["epochs"].append(epoch)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = model.state_dict()
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            print("Early stopping.")
            break
    model.load_state_dict(best_state)
    test_loss, test_swa = evaluate(model, test_loader)
    print(f"TEST  loss={test_loss:.4f}   SWA={test_swa:.4f}")
    run["metrics"]["test"] = test_swa
    experiment_data["runs"][tag] = run

# ---------------- save all ----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
