import os, pathlib, numpy as np, torch, matplotlib

matplotlib.use("Agg")

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


# ============== DATA LOADING ==============
def pick_spr_root() -> pathlib.Path:
    cand = [
        os.getenv("SPR_DATA_PATH", ""),
        "./SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cand:
        if p and pathlib.Path(p).exists():
            return pathlib.Path(p)
    raise FileNotFoundError("SPR_BENCH dataset not found. Set SPR_DATA_PATH env var.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


spr_root = pick_spr_root()
spr = load_spr_bench(spr_root)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# ============== VOCAB & ENCODING ==============
PAD, UNK = "<pad>", "<unk>"
char_set = set(ch for ex in spr["train"] for ch in ex["sequence"])
itos = [PAD, UNK] + sorted(list(char_set))
stoi = {ch: i for i, ch in enumerate(itos)}
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
max_len = 128


def encode(seq: str):
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


def bag_of_symbols(seq: str):
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in seq:
        vec[stoi.get(ch, stoi[UNK])] += 1.0
    return vec


# ============== DATASET ==============
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        ids = torch.tensor(encode(row["sequence"]), dtype=torch.long)
        mask = (ids != stoi[PAD]).long()
        boc = torch.tensor(bag_of_symbols(row["sequence"]), dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "symbol_counts": boc,
            "labels": label,
        }


# ============== MODELS ==============
class TinyTransformer(nn.Module):
    def __init__(self, symbolic=False, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.symbolic = symbolic
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        if symbolic:
            self.sym_proj = nn.Linear(vocab_size, d_model)
            self.fc = nn.Linear(d_model * 2, num_classes)
        else:
            self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask, symbol_counts=None):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        if self.symbolic:
            sym = self.sym_proj(symbol_counts)
            pooled = torch.cat([pooled, sym], dim=-1)
        return self.fc(pooled)


# ============== TRAIN / EVAL ==============
def run_loader(model, loader, crit, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(
                batch["input_ids"], batch["attention_mask"], batch.get("symbol_counts")
            )
            loss = crit(out, batch["labels"])
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds.extend(out.argmax(-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    return total_loss / len(loader.dataset), f1_score(gts, preds, average="macro")


# ============== EXPERIMENT ==============
batch_size = 128
epochs = 3
criterion = nn.CrossEntropyLoss()


def make_loader(split):
    return DataLoader(
        SPRTorchDataset(spr[split]), batch_size=batch_size, shuffle=(split == "train")
    )


train_loader = make_loader("train")
dev_loader = make_loader("dev")
test_loader = make_loader("test")

experiment_data = {
    "baseline": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    },
    "neurosymbolic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    },
}


def train_and_eval(symbolic=False, tag="baseline"):
    model = TinyTransformer(symbolic).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best_val = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1 = run_loader(model, train_loader, criterion, optimizer)
        val_loss, val_f1 = run_loader(model, dev_loader, criterion)
        experiment_data[tag]["losses"]["train"].append(tr_loss)
        experiment_data[tag]["losses"]["val"].append(val_loss)
        experiment_data[tag]["metrics"]["train"].append(tr_f1)
        experiment_data[tag]["metrics"]["val"].append(val_f1)
        print(
            f"[{tag}] Epoch {epoch}: val_loss={val_loss:.4f} val_macroF1={val_f1:.4f}"
        )
        if val_f1 > best_val:
            best_model_state = model.state_dict()
            best_val = val_f1
    model.load_state_dict(best_model_state)
    test_loss, test_f1 = run_loader(model, test_loader, criterion)
    print(f"[{tag}] TEST macroF1={test_f1:.4f}")
    experiment_data[tag]["test_macroF1"] = test_f1


train_and_eval(symbolic=False, tag="baseline")
train_and_eval(symbolic=True, tag="neurosymbolic")

# ============== SAVE & PLOT ==============
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

for tag in ["baseline", "neurosymbolic"]:
    plt.figure()
    plt.plot(experiment_data[tag]["metrics"]["val"], label="val_macroF1")
    plt.title(f"{tag} validation F1")
    plt.xlabel("epoch")
    plt.ylabel("MacroF1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{tag}_f1.png"))
    plt.close()
