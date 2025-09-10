import os, pathlib, random, string, numpy as np, torch, math
from typing import Dict, List
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# house-keeping & GPU setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# utility: find SPR_BENCH or create tiny synthetic fallback
def find_spr_path() -> pathlib.Path:
    cand = [
        os.getenv("SPR_DATA_DIR", ""),
        "./SPR_BENCH",
        "./data/SPR_BENCH",
        "/datasets/SPR_BENCH",
        "/workspace/SPR_BENCH",
    ]
    for p in cand:
        if p and pathlib.Path(p).exists():
            return pathlib.Path(p).resolve()
    return None


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def create_synthetic_spr(
    n_train=2000, n_val=500, n_test=500, vocab=list(string.ascii_lowercase[:8])
):
    def rand_seq():
        ln = random.randint(10, 30)
        s = "".join(random.choices(vocab, k=ln))
        # a toy rule: label = parity of 'a' count (0/1)
        lab = s.count("a") % 2
        return {"sequence": s, "label": lab}

    from datasets import Dataset

    train = Dataset.from_list([rand_seq() for _ in range(n_train)])
    dev = Dataset.from_list([rand_seq() for _ in range(n_val)])
    test = Dataset.from_list([rand_seq() for _ in range(n_test)])
    return DatasetDict({"train": train, "dev": dev, "test": test})


spr_path = find_spr_path()
if spr_path:
    print(f"Loading SPR_BENCH from {spr_path}")
    try:
        spr = load_spr_bench(spr_path)
    except Exception as e:
        print(f"Failed to load official data ({e}); falling back to synthetic.")
        spr = create_synthetic_spr()
else:
    print("SPR_BENCH not found; using synthetic dataset.")
    spr = create_synthetic_spr()

print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------------
# vocabulary & encoding helpers
PAD, UNK = "<pad>", "<unk>"
char_set = set(ch for ex in spr["train"] for ch in ex["sequence"])
itos = [PAD, UNK] + sorted(list(char_set))
stoi = {ch: i for i, ch in enumerate(itos)}
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
max_len = 128  # truncate / pad length


def encode_seq(seq: str) -> List[int]:
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


def symbolic_features(seq: str) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in seq:
        vec[stoi.get(ch, stoi[UNK])] += 1.0
    if len(seq) > 0:
        vec /= len(seq)
    return vec


# ------------------------------------------------------------------
# torch Dataset
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        input_ids = torch.tensor(encode_seq(row["sequence"]), dtype=torch.long)
        attn_mask = (input_ids != stoi[PAD]).long()
        sym_vec = torch.tensor(symbolic_features(row["sequence"]), dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "sym_vec": sym_vec,
            "labels": label,
        }


# ------------------------------------------------------------------
# models
class TinyTransformer(nn.Module):
    def __init__(self, vocab, n_cls, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=stoi[PAD])
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_cls)

    def forward(self, input_ids, attention_mask, sym_vec=None):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1), :]
        x = self.enc(x, src_key_padding_mask=~attention_mask.bool())
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.fc(pooled)


class SymbolicTransformer(nn.Module):
    def __init__(self, vocab, n_cls, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.trans = TinyTransformer(vocab, n_cls, d_model, n_heads, n_layers)
        self.sym_proj = nn.Sequential(
            nn.Linear(vocab, d_model), nn.ReLU(), nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(2 * d_model, n_cls)

    def forward(self, input_ids, attention_mask, sym_vec):
        trans_emb = (
            self.trans.embed(input_ids) + self.trans.pos[:, : input_ids.size(1), :]
        )
        trans_emb = self.trans.enc(
            trans_emb, src_key_padding_mask=~attention_mask.bool()
        )
        pooled = (trans_emb * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        sym_emb = self.sym_proj(sym_vec)
        combined = torch.cat([pooled, sym_emb], dim=-1)
        return self.classifier(combined)


# ------------------------------------------------------------------
# training / evaluation helpers
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(
                batch["input_ids"], batch["attention_mask"], batch.get("sym_vec")
            )
            loss = criterion(out, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(out.argmax(-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    macroF1 = f1_score(gts, preds, average="macro")
    return avg_loss, macroF1, preds, gts


def train_model(
    model_name: str, model_cls, train_set, dev_set, epochs=3, bs=128, lr=3e-4
) -> Dict:
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=bs)
    model = model_cls(vocab_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    stats = {"losses": {"train": [], "val": []}, "metrics": {"train": [], "val": []}}
    for ep in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_loader, criterion)
        stats["losses"]["train"].append(tr_loss)
        stats["losses"]["val"].append(val_loss)
        stats["metrics"]["train"].append(tr_f1)
        stats["metrics"]["val"].append(val_f1)
        print(
            f"{model_name} | Epoch {ep}: val_loss={val_loss:.4f} val_macroF1={val_f1:.4f}"
        )
    return stats, model


# ------------------------------------------------------------------
# execute experiments
experiment_data = {
    "TinyTransformer": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    },
    "SymbolicTransformer": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    },
}

train_set = SPRTorchDataset(spr["train"])
dev_set = SPRTorchDataset(spr["dev"])

tiny_stats, _ = train_model("Tiny", TinyTransformer, train_set, dev_set)
sym_stats, _ = train_model("Symbolic", SymbolicTransformer, train_set, dev_set)

experiment_data["TinyTransformer"] = tiny_stats
experiment_data["SymbolicTransformer"] = sym_stats

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(
    "\nFinal Dev MacroF1 -> Tiny:",
    tiny_stats["metrics"]["val"][-1],
    "| Symbolic:",
    sym_stats["metrics"]["val"][-1],
)
