import os, pathlib, numpy as np, torch, matplotlib, math, random, time

matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

# -------------------- house-keeping & device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- dataset loading --------------------------
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# -------------------- vocabulary & encoding --------------------
PAD, UNK = "<pad>", "<unk>"
char_set = set(ch for ex in spr["train"] for ch in ex["sequence"])
itos = [PAD, UNK] + sorted(list(char_set))
stoi = {ch: i for i, ch in enumerate(itos)}
vocab_size = len(itos)
max_len = min(128, max(len(seq) for seq in spr["train"]["sequence"]))
num_classes = len(set(spr["train"]["label"]))


def encode_seq(seq: str) -> list[int]:
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


def symbolic_counts(seq: str) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in seq:
        idx = stoi.get(ch, stoi[UNK])
        vec[idx] += 1.0
    if len(seq) > 0:
        vec /= len(seq)  # length normalisation
    return vec


# -------------------- torch dataset ----------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = torch.tensor(encode_seq(row["sequence"]), dtype=torch.long)
        attn = (ids != stoi[PAD]).long()
        label = torch.tensor(row["label"], dtype=torch.long)
        sym = torch.tensor(symbolic_counts(row["sequence"]), dtype=torch.float32)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "symbolic": sym,
            "labels": label,
        }


# -------------------- models -----------------------------------
class TinyTransformer(nn.Module):
    def __init__(
        self,
        with_symbolic: bool = False,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.with_symbolic = with_symbolic
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        if with_symbolic:
            self.sym_proj = nn.Sequential(
                nn.Linear(vocab_size, d_model), nn.ReLU(), nn.Dropout(0.1)
            )
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask, symbolic=None):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1), :]
        x = self.enc(x, src_key_padding_mask=~attention_mask.bool())
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        if self.with_symbolic and symbolic is not None:
            sym_repr = self.sym_proj(symbolic)
            pooled = pooled + sym_repr  # simple fusion
        return self.cls(pooled)


# -------------------- train / eval loops -----------------------
def run_epoch(model, loader, criterion, optimiser=None):
    train = optimiser is not None
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out = model(batch["input_ids"], batch["attention_mask"], batch["symbolic"])
        loss = criterion(out, batch["labels"])
        if train:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(-1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# -------------------- experiment --------------------------------
experiment_data = {"Baseline": {}, "SymbolicAug": {}}
batch_size = 128
epochs = 5
criterion = nn.CrossEntropyLoss()


def train_variant(name: str, with_symbolic: bool):
    print(f"\n=== Training {name} (symbolic={with_symbolic}) ===")
    train_loader = DataLoader(
        SPRTorchDataset(spr["train"]), batch_size=batch_size, shuffle=True
    )
    dev_loader = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=batch_size)
    model = TinyTransformer(with_symbolic=with_symbolic).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    stats = {
        "epochs": [],
        "losses": {"train": [], "val": []},
        "metrics": {"train_f1": [], "val_f1": []},
        "predictions": [],
        "ground_truth": [],
    }

    for ep in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, val_preds, val_gts = run_epoch(model, dev_loader, criterion)
        stats["epochs"].append(ep)
        stats["losses"]["train"].append(tr_loss)
        stats["losses"]["val"].append(val_loss)
        stats["metrics"]["train_f1"].append(tr_f1)
        stats["metrics"]["val_f1"].append(val_f1)
        if ep == epochs:
            stats["predictions"] = val_preds
            stats["ground_truth"] = val_gts
        print(
            f"Epoch {ep}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, val_macroF1={val_f1:.4f}"
        )

    # plots
    plt.figure()
    plt.plot(stats["epochs"], stats["losses"]["train"], label="train")
    plt.plot(stats["epochs"], stats["losses"]["val"], label="val")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss {name}")
    plt.savefig(os.path.join(working_dir, f"loss_{name}.png"))
    plt.close()

    plt.figure()
    plt.plot(stats["epochs"], stats["metrics"]["val_f1"], label="val_macro_f1")
    plt.xlabel("epoch")
    plt.ylabel("Macro F1")
    plt.title(f"MacroF1 {name}")
    plt.savefig(os.path.join(working_dir, f"f1_{name}.png"))
    plt.close()

    experiment_data[name] = stats
    print(f"Best Val MacroF1 for {name}: {max(stats['metrics']['val_f1']):.4f}")


train_variant("Baseline", with_symbolic=False)
train_variant("SymbolicAug", with_symbolic=True)

# -------------------- save experiment data ----------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nAll experiments complete. Data saved to working/experiment_data.npy")
