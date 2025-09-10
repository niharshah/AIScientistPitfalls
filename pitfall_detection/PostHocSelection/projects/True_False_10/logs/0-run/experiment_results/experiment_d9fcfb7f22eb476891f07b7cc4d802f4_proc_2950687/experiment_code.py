import os, pathlib, random, time, numpy as np, torch, warnings
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- reproducibility ----------
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_shape_variety(seq):  # number of distinct shapes (first char)
    return len(set(tok[0] for tok in seq.split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------- data loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def fallback_dataset(n_train=1500, n_dev=300, n_test=300):
    shapes, colors = "ABCD", "abcd"

    def gen(n):
        data = [
            {
                "id": i,
                "sequence": " ".join(
                    random.choice(shapes) + random.choice(colors)
                    for _ in range(random.randint(3, 10))
                ),
                "label": random.choice(["yes", "no"]),
            }
            for i in range(n)
        ]
        return load_dataset("json", data_files={"train": data}, split="train")

    return DatasetDict(train=gen(n_train), dev=gen(n_dev), test=gen(n_test))


try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic data:", e)
    dsets = fallback_dataset()

# ---------- vocab construction ----------
shape_set, color_set, label_set = set(), set(), set()
for ex in dsets["train"]:
    toklist = ex["sequence"].split()
    shape_set.update(t[0] for t in toklist if t)
    color_set.update(t[1] if len(t) > 1 else "_" for t in toklist)
    label_set.add(ex["label"])
shape2id = {s: i + 1 for i, s in enumerate(sorted(shape_set))}  # 0 = PAD
color2id = {c: i + 1 for i, c in enumerate(sorted(color_set))}
label2id = {l: i for i, l in enumerate(sorted(label_set))}
id2label = {v: k for k, v in label2id.items()}


# ---------- torch dataset ----------
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        shape_ids = [shape2id.get(t[0], 0) for t in toks]
        color_ids = [color2id.get(t[1] if len(t) > 1 else "_", 0) for t in toks]
        return {
            "shape_ids": torch.tensor(shape_ids, dtype=torch.long),
            "color_ids": torch.tensor(color_ids, dtype=torch.long),
            "len": torch.tensor(len(shape_ids)),
            "n_shape": torch.tensor(count_shape_variety(self.seqs[idx])),
            "label": torch.tensor(self.labels[idx]),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    max_len = max(b["len"] for b in batch).item()
    pad_shape = torch.zeros(len(batch), max_len, dtype=torch.long)
    pad_color = torch.zeros_like(pad_shape)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        l = b["len"]
        pad_shape[i, :l] = b["shape_ids"]
        pad_color[i, :l] = b["color_ids"]
        mask[i, :l] = 1
    return {
        "shape_ids": pad_shape,
        "color_ids": pad_color,
        "mask": mask,
        "n_shape": torch.tensor([b["n_shape"] for b in batch], dtype=torch.float),
        "label": torch.tensor([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


batch_size = 256
train_loader = DataLoader(
    SPRTorch(dsets["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(dsets["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(dsets["test"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class NeuralSymbolicSPR(nn.Module):
    def __init__(self, n_shape_tok, n_color_tok, dims=64, heads=4, layers=2, num_cls=2):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape_tok, dims // 2, padding_idx=0)
        self.color_emb = nn.Embedding(n_color_tok, dims // 2, padding_idx=0)
        self.pos_emb = nn.Embedding(128, dims)  # max length 128
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dims,
            nhead=heads,
            dim_feedforward=dims * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(dims + 1, num_cls)  # +1 symbolic feature n_shape

    def forward(self, shape_ids, color_ids, mask, n_shape_feat):
        tok_emb = torch.cat(
            [self.shape_emb(shape_ids), self.color_emb(color_ids)], dim=-1
        )
        pos = torch.arange(tok_emb.size(1), device=tok_emb.device).unsqueeze(0)
        tok_emb = tok_emb + self.pos_emb(pos)
        enc = self.encoder(tok_emb, src_key_padding_mask=~mask)
        pooled = (enc * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(
            min=1
        )
        out = self.fc(torch.cat([pooled, n_shape_feat.unsqueeze(-1)], dim=-1))
        return out


# ---------- helpers ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    all_seq, all_true, all_pred, losses = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(b["shape_ids"], b["color_ids"], b["mask"], b["n_shape"])
            loss = criterion(logits, b["label"])
            preds = logits.argmax(-1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(b["label"].cpu().numpy())
            all_seq.extend(batch["raw_seq"])
            losses.append(loss.item() * b["label"].size(0))
    return (
        np.sum(losses) / len(all_true),
        shape_weighted_accuracy(all_seq, all_true, all_pred),
        all_true,
        all_pred,
        all_seq,
    )


# ---------- training ----------
dim_options = [64, 96]  # quick sweep
experiment_data = {"runs": {}}

for dims in dim_options:
    print(f"\n=== Model dim {dims} ===")
    model = NeuralSymbolicSPR(
        len(shape2id) + 1, len(color2id) + 1, dims=dims, num_cls=len(label2id)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    run_key = f"dim_{dims}"
    experiment_data["runs"][run_key] = {
        "losses": {"train": [], "val": []},
        "SWA": {"val": []},
    }
    best_val, patience, bad = 1e9, 3, 0
    for epoch in range(1, 31):
        model.train()
        tot, n = 0, 0
        for batch in train_loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(b["shape_ids"], b["color_ids"], b["mask"], b["n_shape"])
            loss = criterion(logits, b["label"])
            loss.backward()
            optimizer.step()
            tot += loss.item() * b["label"].size(0)
            n += b["label"].size(0)
        train_loss = tot / n
        val_loss, val_swa, *_ = evaluate(model, dev_loader)
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  SWA = {val_swa:.4f}")
        experiment_data["runs"][run_key]["losses"]["train"].append(train_loss)
        experiment_data["runs"][run_key]["losses"]["val"].append(val_loss)
        experiment_data["runs"][run_key]["SWA"]["val"].append(val_swa)
        if val_loss < best_val - 1e-4:
            best_val, best_state = val_loss, model.state_dict()
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print("Early stopping.")
            break
    model.load_state_dict(best_state)
    test_loss, test_swa, y_t, y_p, seqs = evaluate(model, test_loader)
    experiment_data["runs"][run_key]["test"] = {"loss": test_loss, "SWA": test_swa}
    print(f"TEST: loss={test_loss:.4f}  SWA={test_swa:.4f}")

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
