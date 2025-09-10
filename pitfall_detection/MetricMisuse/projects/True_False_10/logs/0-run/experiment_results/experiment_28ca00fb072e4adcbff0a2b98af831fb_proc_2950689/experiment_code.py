import os, pathlib, random, time, numpy as np, torch, warnings
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------- required boilerplate ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- metric helpers ----------------
def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ---------------- SPR load (with toy fallback) ----------------
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
        print("Official dataset missing; using synthetic toy.", e)
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
        ds["train"], ds["dev"], ds["test"] = gen(4000), gen(1000), gen(1000)
    return ds


ds = get_dataset()

# ---------------- vocab build ----------------
shapes, colors, labels = set(), set(), set()
for ex in ds["train"]:
    for tok in ex["sequence"].split():
        shapes.add(tok[0])
        colors.add(tok[1])
    labels.add(ex["label"])
shape2id = {s: i + 1 for i, s in enumerate(sorted(shapes))}
color2id = {c: i + 1 for i, c in enumerate(sorted(colors))}
label2id = {l: i for i, l in enumerate(sorted(labels))}
id2label = {v: k for k, v in label2id.items()}
num_shapes, num_colors, num_classes = (
    len(shape2id) + 1,
    len(color2id) + 1,
    len(label2id),
)
print(f"Vocab: {num_shapes-1} shapes, {num_colors-1} colors, {num_classes} classes")


# ---------------- Torch dataset ----------------
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seq, self.lbl = split["sequence"], [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        toks = self.seq[idx].split()
        shape_ids = [shape2id[t[0]] for t in toks]
        color_ids = [color2id[t[1]] for t in toks]
        # symbolic stats
        shape_cnt = np.bincount(shape_ids, minlength=num_shapes)
        color_cnt = np.bincount(color_ids, minlength=num_colors)
        feats = np.concatenate(
            [
                shape_cnt,
                color_cnt,
                [
                    len(toks),
                    len(set([t[0] for t in toks])),
                    len(set([t[1] for t in toks])),
                ],
            ]
        ).astype(np.float32)
        return {
            "shape": torch.tensor(shape_ids),
            "color": torch.tensor(color_ids),
            "len": torch.tensor(len(toks)),
            "sym": torch.tensor(feats),
            "label": torch.tensor(self.lbl[idx]),
            "raw": self.seq[idx],
        }


def collate(batch):
    maxlen = max(b["len"] for b in batch).item()
    shp = torch.zeros(len(batch), maxlen, dtype=torch.long)
    clr = torch.zeros_like(shp)
    msk = torch.zeros_like(shp, dtype=torch.bool)
    for i, b in enumerate(batch):
        l = b["len"]
        shp[i, :l] = b["shape"]
        clr[i, :l] = b["color"]
        msk[i, :l] = 1
    return {
        "shape": shp,
        "color": clr,
        "mask": msk,
        "sym": torch.stack([b["sym"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw": [b["raw"] for b in batch],
    }


train_loader = DataLoader(
    SPRTorch(ds["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(ds["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(ds["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ---------------- model ----------------
class NeuroSymbolic(nn.Module):
    def __init__(
        self,
        d_model=64,
        nhead=4,
        nlayers=1,
        ff=128,
        drop=0.1,
        sym_dim=None,
        ablate=None,
    ):
        super().__init__()
        self.ablate = ablate  # None | 'neural' | 'symbolic'
        self.shape_emb = nn.Embedding(num_shapes, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(num_colors, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(200, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, drop, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.sym_mlp = nn.Sequential(
            nn.Linear(sym_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
        self.fc_neural = nn.Linear(d_model, num_classes)
        self.alpha = nn.Parameter(torch.tensor(0.0))  # blending scalar

    def forward(self, shape, color, mask, sym):
        logits_sym = self.sym_mlp(sym)
        if self.ablate == "symbolic":
            logits_sym *= 0
        if self.ablate == "neural":
            logits_neural = torch.zeros_like(logits_sym)
        else:
            B, L = shape.size()
            pos = torch.arange(L, device=shape.device).unsqueeze(0).expand(B, L)
            x = self.shape_emb(shape) + self.color_emb(color) + self.pos_emb(pos)
            x = self.enc(x, src_key_padding_mask=~mask)
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            logits_neural = self.fc_neural(x)
        alpha = torch.sigmoid(self.alpha)  # 0..1
        return alpha * logits_neural + (1 - alpha) * logits_sym


sym_dim = (num_shapes) + (num_colors) + 3
model = NeuroSymbolic(sym_dim=sym_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# ---------------- experiment store ----------------
experiment_data = {
    "neuro_sym": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------------- train / eval funcs ----------------
def evaluate(model, loader):
    model.eval()
    all_y, all_p, all_seq = [], [], []
    total = 0
    loss_sum = 0
    with torch.no_grad():
        for btch in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in btch.items()
            }
            logits = model(bt["shape"], bt["color"], bt["mask"], bt["sym"].float())
            loss = criterion(logits, bt["label"])
            loss_sum += loss.item() * bt["label"].size(0)
            total += bt["label"].size(0)
            all_y.extend(bt["label"].cpu().numpy())
            all_p.extend(logits.argmax(-1).cpu().numpy())
            all_seq.extend(btch["raw"])
    return loss_sum / total, shape_weighted_accuracy(all_seq, all_y, all_p)


best_val = float("inf")
patience, pat = 4, 0
for epoch in range(1, 31):
    model.train()
    tr_loss_sum, n = 0, 0
    for btch in train_loader:
        bt = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in btch.items()
        }
        optimizer.zero_grad()
        logits = model(bt["shape"], bt["color"], bt["mask"], bt["sym"].float())
        loss = criterion(logits, bt["label"])
        loss.backward()
        optimizer.step()
        tr_loss_sum += loss.item() * bt["label"].size(0)
        n += bt["label"].size(0)
    train_loss = tr_loss_sum / n
    val_loss, val_swa = evaluate(model, dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA = {val_swa:.4f}")
    experiment_data["neuro_sym"]["losses"]["train"].append(train_loss)
    experiment_data["neuro_sym"]["losses"]["val"].append(val_loss)
    experiment_data["neuro_sym"]["metrics"]["val"].append(val_swa)
    experiment_data["neuro_sym"]["epochs"].append(epoch)
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
print(f"TEST: loss={test_loss:.4f} | SWA={test_swa:.4f}")
experiment_data["neuro_sym"]["metrics"]["test"] = test_swa

# ---------------- save metrics ----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
