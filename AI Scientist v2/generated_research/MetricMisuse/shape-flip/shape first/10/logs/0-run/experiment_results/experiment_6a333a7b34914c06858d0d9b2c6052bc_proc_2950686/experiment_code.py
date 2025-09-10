import os, pathlib, random, time, numpy as np, torch, warnings
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

warnings.filterwarnings("ignore")
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------- helper metrics ----------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


# ---------- load SPR_BENCH or fall back to toy ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _ld(s) for s in ["train", "dev", "test"]})


try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic toy data.", e)

    def _gen(n):
        shapes, colors = "ABCD", "abcd"
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

    dsets = DatasetDict({"train": _gen(2000), "dev": _gen(400), "test": _gen(400)})

# ---------- vocab ----------
tokens = set()
labels = set()
for ex in dsets["train"]:
    tokens.update(ex["sequence"].split())
    labels.add(ex["label"])
tok2id = {tok: i + 1 for i, tok in enumerate(sorted(tokens))}
label2id = {lab: i for i, lab in enumerate(sorted(labels))}
id2label = {v: k for k, v in label2id.items()}
vocab_size = len(tok2id) + 1
num_classes = len(label2id)
print(f"Vocab={vocab_size-1}, classes={num_classes}")


# ---------- torch dataset ----------
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.lbl = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx].split()
        ids = [tok2id[t] for t in seq]
        n_shape, count_shape = count_shape_variety(self.seqs[idx]), count_color_variety(
            self.seqs[idx]
        )
        all_same_shape, all_same_color = int(n_shape == 1), int(count_shape == 1)
        return {
            "input_ids": torch.tensor(ids),
            "length": torch.tensor(len(ids)),
            "symb": torch.tensor(
                [n_shape, count_shape, all_same_shape, all_same_color],
                dtype=torch.float,
            ),
            "label": torch.tensor(self.lbl[idx]),
            "raw": self.seqs[idx],
        }


def collate(batch):
    max_len = max(b["length"] for b in batch).item()
    pad_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros_like(pad_ids, dtype=torch.bool)
    for i, b in enumerate(batch):
        pad_ids[i, : b["length"]] = b["input_ids"]
        mask[i, : b["length"]] = 1
    return {
        "input_ids": pad_ids,
        "mask": mask,
        "symb": torch.stack([b["symb"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw": [b["raw"] for b in batch],
    }


BATCH = 128
train_loader = DataLoader(
    SPRTorch(dsets["train"]), batch_size=BATCH, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(dsets["dev"]), batch_size=BATCH, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(dsets["test"]), batch_size=BATCH, shuffle=False, collate_fn=collate
)


# ---------- models ----------
class NeuroSymbolic(nn.Module):
    def __init__(self, use_embed=True, use_symb=True):
        super().__init__()
        self.use_embed = use_embed
        self.use_symb = use_symb
        dim_embed = 64
        in_dim = (dim_embed if use_embed else 0) + (4 if use_symb else 0)
        self.emb = (
            nn.Embedding(vocab_size, dim_embed, padding_idx=0) if use_embed else None
        )
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, ids, mask, symb):
        feats = []
        if self.use_embed:
            em = self.emb(ids)
            avg = (em * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(
                -1
            )
            feats.append(avg)
        if self.use_symb:
            feats.append(symb)
        x = torch.cat(feats, -1)
        return self.fc(x)


# ---------- train / evaluate ----------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, opt=None):
    train = opt is not None
    if train:
        model.train()
    else:
        model.eval()
    tot_loss = n = 0
    all_s, all_t, all_p = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["mask"], batch["symb"])
        loss = criterion(logits, batch["label"])
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * batch["label"].size(0)
        n += batch["label"].size(0)
        if not train:
            preds = logits.argmax(-1).cpu().numpy()
            all_p.extend(preds)
            all_t.extend(batch["label"].cpu().numpy())
            all_s.extend(batch["raw"])
    if train:
        return tot_loss / n
    else:
        swa = shape_weighted_accuracy(all_s, all_t, all_p)
        return tot_loss / n, swa


# ---------- experiment ----------
variants = {
    "neuro_symbolic": (True, True),
    "neural_only": (True, False),
    "symbolic_only": (False, True),
}
experiment_data = {}
MAX_EPOCH = 20
PATIENCE = 3
for name, (use_emb, use_symb) in variants.items():
    print(f"\n=== {name} ===")
    model = NeuroSymbolic(use_emb, use_symb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val = np.inf
    wait = 0
    exp = {"losses": {"train": [], "val": []}, "metrics": {"val": [], "test": None}}
    for ep in range(1, MAX_EPOCH + 1):
        tr_loss = run_epoch(model, train_loader, opt)
        val_loss, val_swa = run_epoch(model, dev_loader)
        print(f"Epoch {ep}: validation_loss = {val_loss:.4f}, SWA = {val_swa:.4f}")
        exp["losses"]["train"].append(tr_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["val"].append(val_swa)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break
    model.load_state_dict(best_state)
    tst_loss, tst_swa = run_epoch(model, test_loader)
    print(f"TEST: loss={tst_loss:.4f}, SWA={tst_swa:.4f}")
    exp["metrics"]["test"] = tst_swa
    experiment_data[name] = exp

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
