import os, pathlib, random, time, numpy as np, torch, warnings
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ──────────────────────────────────────────────────────────
# housekeeping & globals
# ──────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

SEED = 13
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ──────────────────────────────────────────────────────────
# utils: metrics & dataset loader
# ──────────────────────────────────────────────────────────
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(file):
        return load_dataset(
            "csv", data_files=str(root / file), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {split: _load(f"{split}.csv") for split in ["train", "dev", "test"]}
    )


def get_dataset():
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        dset = load_spr_bench(DATA_PATH)
        print("Loaded SPR_BENCH.")
    except Exception as e:
        print("Falling back to synthetic toy data.", e)
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

        dset = DatasetDict()
        dset["train"], dset["dev"], dset["test"] = gen(1500), gen(300), gen(300)
    return dset


dset = get_dataset()

# ──────────────────────────────────────────────────────────
# vocabulary
# ──────────────────────────────────────────────────────────
all_tokens = set()
all_labels = set()
for ex in dset["train"]:
    all_tokens.update(ex["sequence"].split())
    all_labels.add(ex["label"])
tok2id = {t: i + 1 for i, t in enumerate(sorted(all_tokens))}
label2id = {l: i for i, l in enumerate(sorted(all_labels))}
id2label = {v: k for k, v in label2id.items()}
VOCAB = len(tok2id) + 1
N_CLASS = len(label2id)
print(f"Vocab={VOCAB-1}, classes={N_CLASS}")


# ──────────────────────────────────────────────────────────
# torch dataset
# ──────────────────────────────────────────────────────────
class SPRTorch(Dataset):
    def __init__(self, split):
        self.data = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]["sequence"]
        lbl = self.data[idx]["label"]
        ids = [tok2id[t] for t in seq.split()]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "shape_var": torch.tensor(count_shape_variety(seq)),
            "color_var": torch.tensor(count_color_variety(seq)),
            "length": torch.tensor(len(ids)),
            "label": torch.tensor(label2id[lbl], dtype=torch.long),
            "raw_seq": seq,
        }


def collate(batch):
    L = max(x["length"] for x in batch).item()
    padded = torch.zeros(len(batch), L, dtype=torch.long)
    mask = torch.zeros(len(batch), L, dtype=torch.bool)
    for i, b in enumerate(batch):
        l = b["length"]
        padded[i, :l] = b["input_ids"]
        mask[i, :l] = 1
    feats = torch.stack(
        [
            torch.tensor(
                [b["shape_var"], b["color_var"], b["length"]], dtype=torch.float
            )
            for b in batch
        ]
    )
    return {
        "input_ids": padded,
        "mask": mask,
        "features": feats,
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


batch_size = 128
train_loader = DataLoader(
    SPRTorch(dset["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(dset["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(dset["test"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ──────────────────────────────────────────────────────────
# models: symbolic, neural, hybrid
# ──────────────────────────────────────────────────────────
class SymbolicMLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat):
        return feat


class NeuralEncoder(nn.Module):
    def __init__(self, vocab, emb_dim=64, nhead=4, nlayers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            emb_dim, nhead, dim_feedforward=emb_dim * 2, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)

    def forward(self, ids, mask):
        x = self.emb(ids)
        key_mask = ~mask
        h = self.enc(x, src_key_padding_mask=key_mask)
        return h.masked_fill(key_mask.unsqueeze(-1), 0).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp(min=1)


class SPRClassifier(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode  # 'neural'|'symbolic'|'hybrid'
        if mode in ["neural", "hybrid"]:
            self.neural = NeuralEncoder(VOCAB)
        if mode in ["symbolic", "hybrid"]:
            self.symb_mlp = nn.Sequential(
                nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16)
            )
        in_dim = {"neural": 64, "symbolic": 16, "hybrid": 80}[mode]
        self.classifier = nn.Linear(in_dim, N_CLASS)

    def forward(self, ids, mask, feat):
        reps = []
        if self.mode in ["neural", "hybrid"]:
            reps.append(self.neural(ids, mask))
        if self.mode in ["symbolic", "hybrid"]:
            reps.append(self.symb_mlp(feat))
        rep = torch.cat(reps, dim=-1)
        return self.classifier(rep)


# ──────────────────────────────────────────────────────────
# training / evaluation helpers
# ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum = n = 0
    ys, yp, seqs = [], [], []
    for batch in loader:
        b = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(b["input_ids"], b["mask"], b["features"])
        loss = criterion(logits, b["label"])
        loss_sum += loss.item() * b["label"].size(0)
        n += b["label"].size(0)
        yp.extend(logits.argmax(-1).cpu().numpy())
        ys.extend(b["label"].cpu().numpy())
        seqs.extend(batch["raw_seq"])
    return loss_sum / n, shape_weighted_accuracy(seqs, ys, yp)


def train_model(mode, max_epochs=15, patience=3):
    model = SPRClassifier(mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val = float("inf")
    wait = 0
    best_state = None
    run = {"losses": {"train": [], "val": []}, "metrics": {"val": []}, "test": None}
    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_loss = m = 0
        for batch in train_loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(b["input_ids"], b["mask"], b["features"])
            loss = criterion(logits, b["label"])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * b["label"].size(0)
            m += b["label"].size(0)
        train_loss = ep_loss / m
        val_loss, val_swa = evaluate(model, dev_loader)
        print(
            f"{mode.upper()} Epoch {epoch}: val_loss={val_loss:.4f} SWA={val_swa:.4f}"
        )
        run["losses"]["train"].append(train_loss)
        run["losses"]["val"].append(val_loss)
        run["metrics"]["val"].append(val_swa)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
        if wait >= patience:
            print("Early stop.")
            break
    model.load_state_dict(best_state)
    test_loss, test_swa = evaluate(model, test_loader)
    run["test"] = {"loss": test_loss, "SWA": test_swa}
    print(f"{mode.upper()} TEST: loss={test_loss:.4f} SWA={test_swa:.4f}")
    return run


# ──────────────────────────────────────────────────────────
# run experiments
# ──────────────────────────────────────────────────────────
experiment_data = {"neural": {}, "symbolic": {}, "hybrid": {}}
for mode in ["neural", "symbolic", "hybrid"]:
    experiment_data[mode] = train_model(mode)

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
