import os, pathlib, random, json, math, time
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef

# ----------  experiment container -------------------------------
experiment_data = {"NUM_LSTM_LAYERS": {"SPR_BENCH": {}}}

# ----------  misc setup -----------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
print("Using device:", device)


# ----------  load / synthesise  SPR_BENCH -----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def maybe_load_real_dataset() -> DatasetDict:
    root = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    if root.exists():
        print(f"Loading real SPR_BENCH from {root}")
        return load_spr_bench(root)

    print("Real dataset not found – generating synthetic parity dataset.")
    from datasets import Dataset as HFDataset

    def make_split(n):
        syms = list("ABCDEFGH")
        seqs, labels = [], []
        for _ in range(n):
            seq = "".join(random.choice(syms) for _ in range(random.randint(5, 12)))
            labels.append(int(seq.count("A") % 2 == 0))
            seqs.append(seq)
        return HFDataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    return DatasetDict(
        {"train": make_split(2000), "dev": make_split(500), "test": make_split(500)}
    )


spr = maybe_load_real_dataset()
print("Dataset splits:", spr.keys())

# ----------  vocab / encoding -----------------------------------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
pad_idx, max_len = 0, min(40, max(len(s) for s in spr["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    ids.extend([pad_idx] * (max_len - len(ids)))
    return ids


# ----------  torch Datasets -------------------------------------
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seq, self.lab = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "y": torch.tensor(self.lab[idx], dtype=torch.float32),
        }


train_ds, val_ds, test_ds = (
    SPRTorch(spr["train"]),
    SPRTorch(spr["dev"]),
    SPRTorch(spr["test"]),
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)


# ----------  model ----------------------------------------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        pooled = out.mean(1)
        return self.fc(pooled).squeeze(1)


# ----------  training / evaluation utilities --------------------
def run_epoch(model, loader, criterion, optim=None):
    is_train = optim is not None
    model.train() if is_train else model.eval()
    tot_loss, preds, truths = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if is_train:
            optim.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        if is_train:
            loss.backward()
            optim.step()
        tot_loss += loss.item() * batch["x"].size(0)
        preds.extend((torch.sigmoid(logits).detach().cpu().numpy() > 0.5).tolist())
        truths.extend(batch["y"].detach().cpu().numpy().tolist())
    return tot_loss / len(loader.dataset), matthews_corrcoef(truths, preds)


# ----------  hyperparameter sweep --------------------------------
num_layers_options = [1, 2, 3]
epochs = 5
criterion = nn.BCEWithLogitsLoss()

for n_layers in num_layers_options:
    tag = f"{n_layers}_layer"
    experiment_data["NUM_LSTM_LAYERS"]["SPR_BENCH"][tag] = {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
    }

    model = CharBiLSTM(len(vocab), n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n=== Training model with {n_layers} LSTM layer(s) ===")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_mcc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_mcc = run_epoch(model, val_loader, criterion)

        ed = experiment_data["NUM_LSTM_LAYERS"]["SPR_BENCH"][tag]
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(vl_loss)
        ed["metrics"]["train_MCC"].append(tr_mcc)
        ed["metrics"]["val_MCC"].append(vl_mcc)
        ed["epochs"].append(epoch)

        print(
            f"Layer {n_layers} | Ep {epoch}: train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, val_MCC={vl_mcc:.4f}"
        )

    # ---------------- test evaluation -------------------
    tl_loss, tl_mcc = run_epoch(model, test_loader, criterion)
    ed["test_MCC"] = tl_mcc
    print(f"Layer {n_layers}: Test MCC = {tl_mcc:.4f}")

# ----------  save results / plots -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# plot MCC curves
plt.figure()
for n_layers in num_layers_options:
    tag = f"{n_layers}_layer"
    mccs = experiment_data["NUM_LSTM_LAYERS"]["SPR_BENCH"][tag]["metrics"]["val_MCC"]
    plt.plot(range(1, len(mccs) + 1), mccs, label=f"{n_layers} layer(s)")
plt.xlabel("Epoch")
plt.ylabel("Validation MCC")
plt.legend()
plt.title("MCC vs epochs")
plt.savefig(os.path.join(working_dir, "mcc_curve.png"))
plt.close()

# plot Loss curves (validation)
plt.figure()
for n_layers in num_layers_options:
    tag = f"{n_layers}_layer"
    vloss = experiment_data["NUM_LSTM_LAYERS"]["SPR_BENCH"][tag]["losses"]["val"]
    plt.plot(range(1, len(vloss) + 1), vloss, label=f"{n_layers} layer(s)")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.title("Val Loss vs epochs")
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
