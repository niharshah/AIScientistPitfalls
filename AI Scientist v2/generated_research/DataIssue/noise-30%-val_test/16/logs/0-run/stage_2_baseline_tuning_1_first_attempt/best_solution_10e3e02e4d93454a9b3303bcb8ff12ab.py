import os, pathlib, random, time, json, math
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef

# --------------------------------------------------------------------------- #
#                               bookkeeping                                   #
# --------------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"batch_size_sweep": {"SPR_BENCH": {}}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------------------------------- #
#                         dataset (real or synthetic)                         #
# --------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name: str):
        return load_dataset(
            "csv", data_files=str(root / csv_name), split="train", cache_dir=".cache"
        )

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


def build_synth_split(n: int):
    syms = list("ABCDEFGH")
    seqs, labels = [], []
    for _ in range(n):
        s = "".join(random.choice(syms) for _ in range(random.randint(5, 12)))
        labels.append(int(s.count("A") % 2 == 0))
        seqs.append(s)
    return {"id": list(range(n)), "sequence": seqs, "label": labels}


def maybe_load_data() -> DatasetDict:
    root = pathlib.Path(os.getenv("SPR_PATH", "/nonexistent_path"))
    if root.exists():
        print("Loading real SPR_BENCH from", root)
        return load_spr_bench(root)

    print("Real dataset not found. Using synthetic toy data.")
    from datasets import Dataset as HFDataset

    d = DatasetDict()
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        d[split] = HFDataset.from_dict(build_synth_split(n))
    return d


spr_bench = maybe_load_data()
print("Loaded splits:", spr_bench.keys())

# --------------------------------------------------------------------------- #
#                                vocabulary                                   #
# --------------------------------------------------------------------------- #
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
pad_idx = 0
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 reserved
itos = {i: ch for ch, i in stoi.items()}
max_len = min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq: str):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


class SPRTorch(Dataset):
    def __init__(self, hf_dataset):
        self.seqs = hf_dataset["sequence"]
        self.labels = hf_dataset["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


train_ds, val_ds, test_ds = (SPRTorch(spr_bench[s]) for s in ["train", "dev", "test"])


# --------------------------------------------------------------------------- #
#                                   model                                     #
# --------------------------------------------------------------------------- #
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        pooled = out.mean(1)
        return self.fc(pooled).squeeze(1)


# --------------------------------------------------------------------------- #
#                              training routine                               #
# --------------------------------------------------------------------------- #
def run_experiment(batch_size: int, epochs: int = 5):
    print(f"\n=== Training with batch_size={batch_size} ===")
    data = {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
        "test_MCC": None,
    }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = CharBiLSTM(len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        tic = time.time()
        model.train()
        tr_losses, preds, truths = 0.0, [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            tr_losses += loss.item() * batch["x"].size(0)
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            truths.extend(batch["y"].cpu().numpy())

        train_loss = tr_losses / len(train_ds)
        train_mcc = matthews_corrcoef(truths, preds)

        model.eval()
        val_losses, v_preds, v_truths = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["x"])
                val_losses += criterion(logits, batch["y"]).item() * batch["x"].size(0)
                v_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                v_truths.extend(batch["y"].cpu().numpy())
        val_loss = val_losses / len(val_ds)
        val_mcc = matthews_corrcoef(v_truths, v_preds)

        data["losses"]["train"].append(train_loss)
        data["losses"]["val"].append(val_loss)
        data["metrics"]["train_MCC"].append(train_mcc)
        data["metrics"]["val_MCC"].append(val_mcc)
        data["epochs"].append(epoch)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_MCC={val_mcc:.4f}, time={(time.time()-tic):.1f}s"
        )

    # -------------- final test evaluation ----------------
    model.eval()
    t_preds, t_truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            t_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            t_truths.extend(batch["y"].cpu().numpy())
    test_mcc = matthews_corrcoef(t_truths, t_preds)
    data["predictions"] = t_preds
    data["ground_truth"] = t_truths
    data["test_MCC"] = test_mcc
    print(f"Test MCC (bs={batch_size}): {test_mcc:.4f}")
    return data


# --------------------------------------------------------------------------- #
#                           hyper-parameter sweep                             #
# --------------------------------------------------------------------------- #
batch_sizes = [32, 64, 128, 256, 512]
best_val_mcc = {}

for bs in batch_sizes:
    res = run_experiment(bs)
    experiment_data["batch_size_sweep"]["SPR_BENCH"][f"bs_{bs}"] = res
    best_val_mcc[bs] = max(res["metrics"]["val_MCC"])

# --------------------------------------------------------------------------- #
#                          save & simple visualisation                        #
# --------------------------------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

plt.figure()
plt.bar([str(b) for b in batch_sizes], [best_val_mcc[b] for b in batch_sizes])
plt.xlabel("Batch size")
plt.ylabel("Best Val MCC")
plt.title("Batch-size sweep on SPR_BENCH")
plt.savefig(os.path.join(working_dir, "bs_vs_mcc.png"))
print("All done. Data saved to", os.path.join(working_dir, "experiment_data.npy"))
