import os, pathlib, random, time, json, math, warnings

warnings.filterwarnings("ignore")
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

# --------------------- working dir -------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------- experiment container ----------------------------------
experiment_data = {
    "WEIGHT_DECAY": {"SPR_BENCH": {}}  # individual runs keyed by weight-decay value
}

# --------------------- device -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------- dataset helpers ---------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper to read a split csv
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for spl in ["train", "dev", "test"]:
        d[spl if spl != "dev" else "dev"] = _load(f"{spl}.csv")
    return d


def maybe_load_real_dataset() -> DatasetDict:
    env_path = os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    root = pathlib.Path(env_path)
    if root.exists():
        print(f"Loading real SPR_BENCH from {root}")
        return load_spr_bench(root)

    # -------- fallback: small synthetic toy dataset --------------------------
    print("Real dataset not found – creating synthetic data …")

    def synth_split(n):
        syms, seqs, labels = list("ABCDEFGH"), [], []
        for _ in range(n):
            seq = "".join(random.choice(syms) for _ in range(random.randint(5, 12)))
            labels.append(int(seq.count("A") % 2 == 0))  # parity rule
            seqs.append(seq)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    from datasets import Dataset as HFDS

    out = DatasetDict()
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        out[split] = HFDS.from_dict(synth_split(n))
    return out


spr_bench = maybe_load_real_dataset()
print("Loaded splits:", spr_bench.keys())

# --------------------- vocab / encoding --------------------------------------
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 for PAD
itos = {i: ch for ch, i in stoi.items()}
pad_idx = 0
max_len = min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# --------------------- torch Dataset -----------------------------------------
class SPRTorch(Dataset):
    def __init__(self, hf_dataset):
        self.seqs, self.labels = hf_dataset["sequence"], hf_dataset["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


train_ds, val_ds, test_ds = (
    SPRTorch(spr_bench[spl]) for spl in ["train", "dev", "test"]
)


# --------------------- model --------------------------------------------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out.mean(dim=1)).squeeze(1)
        return logits


# --------------------- training / evaluation utils ---------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    if train:
        model.train()
    else:
        model.eval()
    total_loss, preds, truths = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["x"].size(0)
            preds.extend((torch.sigmoid(logits).detach() > 0.5).cpu().numpy())
            truths.extend(batch["y"].detach().cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    mcc = matthews_corrcoef(truths, preds)
    return avg_loss, mcc, preds, truths


# --------------------- hyper-parameter grid -----------------------------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
epochs = 5
best_val_mcc, best_run_key = -1.0, None

for wd in weight_decays:
    key = f"wd_{wd}"
    print(f"\n=== Training with weight_decay={wd} ===")
    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)
    # model / optim / loss
    model = CharBiLSTM(len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    # storage for this run
    run_log = {
        "epochs": [],
        "losses": {"train": [], "val": []},
        "metrics": {"train_MCC": [], "val_MCC": [], "test_MCC": None},
        "predictions": [],
        "ground_truth": [],
    }

    # ----- training -----
    for epoch in range(1, epochs + 1):
        tr_loss, tr_mcc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_mcc, _, _ = run_epoch(model, val_loader, criterion)
        run_log["epochs"].append(epoch)
        run_log["losses"]["train"].append(tr_loss)
        run_log["losses"]["val"].append(val_loss)
        run_log["metrics"]["train_MCC"].append(tr_mcc)
        run_log["metrics"]["val_MCC"].append(val_mcc)
        print(
            f"  Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_MCC={val_mcc:.4f}"
        )

    # ----- final test -----
    _, _, t_preds, t_truths = run_epoch(model, test_loader, criterion)
    test_mcc = matthews_corrcoef(t_truths, t_preds)
    run_log["metrics"]["test_MCC"] = test_mcc
    if test_mcc is not None:  # store predictions for best val run
        run_log["predictions"] = t_preds
        run_log["ground_truth"] = t_truths
    print(f"  >> Test MCC: {test_mcc:.4f}")

    experiment_data["WEIGHT_DECAY"]["SPR_BENCH"][key] = run_log

    # track best based on validation MCC (last epoch)
    if val_mcc > best_val_mcc:
        best_val_mcc, best_run_key = val_mcc, key

print(f"\nBest validation MCC {best_val_mcc:.4f} with run '{best_run_key}'")

# --------------------- save & plots ------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# simple plots for each run (loss & val MCC curves)
for key, run in experiment_data["WEIGHT_DECAY"]["SPR_BENCH"].items():
    ep = run["epochs"]
    plt.figure()
    plt.plot(ep, run["losses"]["train"], label="train")
    plt.plot(ep, run["losses"]["val"], label="val")
    plt.title(f"Loss ({key})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_{key}.png"))
    plt.close()

    plt.figure()
    plt.plot(ep, run["metrics"]["val_MCC"], label="val_MCC")
    plt.title(f"Validation MCC ({key})")
    plt.xlabel("epoch")
    plt.ylabel("MCC")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"mcc_{key}.png"))
    plt.close()
