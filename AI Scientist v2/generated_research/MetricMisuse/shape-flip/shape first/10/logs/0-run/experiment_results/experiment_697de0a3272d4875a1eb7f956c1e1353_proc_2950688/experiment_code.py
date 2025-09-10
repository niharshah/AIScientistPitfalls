import os, random, time, copy, warnings, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# --- mandatory working dir ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- reproducibility & device ---
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========== Utility metrics (SWA only) ==========
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred) -> float:
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ========== Data loading with fallback ==========
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def get_dataset():
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        ds = load_spr_bench(DATA_PATH)
        print("Loaded official SPR_BENCH.")
    except Exception as e:
        print("Falling back to synthetic mini-benchmark.", e)
        shapes, colours = "ABCD", "abcd"

        def synth(n):
            rows = [
                {
                    "id": i,
                    "sequence": " ".join(
                        random.choice(shapes) + random.choice(colours)
                        for _ in range(random.randint(3, 10))
                    ),
                    "label": random.choice(["yes", "no"]),
                }
                for i in range(n)
            ]
            return load_dataset("json", data_files={"train": rows}, split="train")

        ds = DatasetDict()
        ds["train"], ds["dev"], ds["test"] = synth(1500), synth(300), synth(300)
    return ds


dset = get_dataset()

# ========== Vocabulary & label maps ==========
vocab = set()
labels = set()
for split in dset.values():
    for ex in split:
        vocab.update(ex["sequence"].split())
        labels.add(ex["label"])
tok2id = {tok: i + 1 for i, tok in enumerate(sorted(vocab))}  # 0=PAD
label2id = {l: i for i, l in enumerate(sorted(labels))}
id2label = {v: k for k, v in label2id.items()}
VOCAB_SIZE = len(tok2id) + 1
NUM_CLS = len(label2id)
print(f"Vocab={VOCAB_SIZE-1}, Classes={NUM_CLS}")

# compute max shape count for normalisation
max_shape_train = max(count_shape_variety(s) for s in dset["train"]["sequence"])


# ========== Torch Dataset ==========
class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.rows = hf_split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        seq = self.rows[idx]["sequence"]
        ids = [tok2id[tok] for tok in seq.split()]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(len(ids)),
            "n_shape": torch.tensor(count_shape_variety(seq), dtype=torch.float),
            "label": torch.tensor(label2id[self.rows[idx]["label"]], dtype=torch.long),
            "raw_seq": seq,
        }


def collate(batch):
    max_len = max(b["len"] for b in batch).item()
    pad_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        pad_ids[i, : b["len"]] = b["input_ids"]
        mask[i, : b["len"]] = 1
    return {
        "input_ids": pad_ids,
        "mask": mask,
        "n_shape": torch.stack([b["n_shape"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


bs = 128
train_loader = DataLoader(
    SPRTorch(dset["train"]), batch_size=bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(dset["dev"]), batch_size=bs, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(dset["test"]), batch_size=bs, shuffle=False, collate_fn=collate
)


# ========== Models ==========
class NeuroSymbolic(nn.Module):
    def __init__(self, use_symbolic: bool = True):
        super().__init__()
        self.use_symbolic = use_symbolic
        self.emb = nn.Embedding(VOCAB_SIZE, 64, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cls = nn.Linear(64 + (1 if use_symbolic else 0), NUM_CLS)

    def forward(self, ids, mask, n_shape):
        x = self.emb(ids)  # [B,T,64]
        x = self.enc(x, src_key_padding_mask=~mask)  # Transformer expects True=pad
        rep = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        if self.use_symbolic:
            feat = (n_shape / max_shape_train).unsqueeze(-1)  # normalised
            rep = torch.cat([rep, feat], -1)
        return self.cls(rep)


class SymbolicOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, NUM_CLS)

    def forward(self, ids, mask, n_shape):
        feat = (n_shape / max_shape_train).unsqueeze(-1)
        return self.lin(feat)


# ========== Train / Eval helpers ==========
ce_loss = nn.CrossEntropyLoss()


def step_batch_to_device(batch):
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss = n = 0
    all_p = []
    all_t = []
    all_seq = []
    for batch in loader:
        batch = step_batch_to_device(batch)
        logits = model(batch["input_ids"], batch["mask"], batch["n_shape"])
        loss = ce_loss(logits, batch["label"])
        tot_loss += loss.item() * batch["label"].size(0)
        n += batch["label"].size(0)
        preds = logits.argmax(-1).cpu().numpy()
        all_p.extend(preds)
        all_t.extend(batch["label"].cpu().numpy())
        all_seq.extend(batch["raw_seq"])
    return tot_loss / n, shape_weighted_accuracy(all_seq, all_t, all_p)


def train_model(
    model_name, model, train_loader, dev_loader, epochs=15, lr=1e-3, patience=3
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    wait = 0
    best_state = None
    record = {
        "losses": {"train": [], "val": []},
        "metrics": {"val": []},
        "timestamps": [],
    }
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0
        m = 0
        for batch in train_loader:
            batch = step_batch_to_device(batch)
            opt.zero_grad()
            logits = model(batch["input_ids"], batch["mask"], batch["n_shape"])
            loss = ce_loss(logits, batch["label"])
            loss.backward()
            opt.step()
            tot += loss.item() * batch["label"].size(0)
            m += batch["label"].size(0)
        tr_loss = tot / m
        val_loss, val_swa = evaluate(model, dev_loader)
        print(
            f"{model_name} Epoch {ep}: validation_loss = {val_loss:.4f}, SWA = {val_swa:.4f}"
        )
        record["losses"]["train"].append(tr_loss)
        record["losses"]["val"].append(val_loss)
        record["metrics"]["val"].append(val_swa)
        record["timestamps"].append(time.time())
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stop.")
                break
    model.load_state_dict(best_state)
    test_loss, test_swa = evaluate(model, test_loader)
    print(f"{model_name} TEST: loss={test_loss:.4f}, SWA={test_swa:.4f}")
    record["losses"]["test"] = test_loss
    record["metrics"]["test"] = test_swa
    return record


# ========== Run experiments ==========
experiment_data = {"neuro_symbolic": {}, "neural_only": {}, "symbolic_only": {}}

# 1. Neuro-symbolic (Transformer + symbolic count)
experiment_data["neuro_symbolic"] = train_model(
    "NeuroSym", NeuroSymbolic(use_symbolic=True), train_loader, dev_loader
)

# 2. Neural-only (Transformer, no symbolic)
experiment_data["neural_only"] = train_model(
    "NeuralOnly", NeuroSymbolic(use_symbolic=False), train_loader, dev_loader
)

# 3. Symbolic-only (logistic on counts)
experiment_data["symbolic_only"] = train_model(
    "SymbolicOnly", SymbolicOnly(), train_loader, dev_loader
)

# ========== Save ==========
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to working/experiment_data.npy")
