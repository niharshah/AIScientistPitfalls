import os, pathlib, random, time, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
#                               work dir + device                             #
# --------------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------------------------------- #
#                         load (real or synthetic) data                       #
# --------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


def maybe_load_real_dataset() -> DatasetDict:
    root = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    if root.exists():
        print(f"Loading real SPR_BENCH from {root}")
        return load_spr_bench(root)
    print("Real dataset not found, building synthetic data...")
    from datasets import Dataset as HFDataset

    def synth_split(n):
        syms = list("ABCDEFGH")
        seqs, labels = [], []
        for _ in range(n):
            s = "".join(random.choice(syms) for _ in range(random.randint(5, 12)))
            labels.append(int(s.count("A") % 2 == 0))
            seqs.append(s)
        return HFDataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    return DatasetDict(
        train=synth_split(2000), dev=synth_split(500), test=synth_split(500)
    )


spr_bench = maybe_load_real_dataset()
print("Loaded splits:", spr_bench.keys())

# --------------------------------------------------------------------------- #
#                               vocab / encoding                              #
# --------------------------------------------------------------------------- #
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 for PAD
itos = {i: ch for ch, i in stoi.items()}
pad_idx = 0
max_len = min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    return ids + [pad_idx] * (max_len - len(ids))


# --------------------------------------------------------------------------- #
#                              Torch Datasets                                 #
# --------------------------------------------------------------------------- #
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


train_ds, val_ds, test_ds = (SPRTorch(spr_bench[s]) for s in ("train", "dev", "test"))


# --------------------------------------------------------------------------- #
#                             BiLSTM definition                               #
# --------------------------------------------------------------------------- #
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_sz, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        pooled = out.mean(1)
        return self.fc(pooled).squeeze(1)


# --------------------------------------------------------------------------- #
#                        hyper-parameter tuning loop                          #
# --------------------------------------------------------------------------- #
hidden_dims = [32, 64, 128, 256]
experiment_data = {"HIDDEN_DIM_TUNING": {"SPR_BENCH": {}}}

for hid in hidden_dims:
    print(f"\n===== Training with HIDDEN_DIM={hid} =====")
    model = CharBiLSTM(len(vocab), hidden=hid).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    entry = {
        "metrics": {"train_MCC": [], "val_MCC": [], "test_MCC": None},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
    epochs = 5
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tloss, tp, tt = 0.0, [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            logits = model(batch["x"])
            loss = crit(logits, batch["y"])
            loss.backward()
            optim.step()
            tloss += loss.item() * batch["x"].size(0)
            tp.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tt.extend(batch["y"].cpu().numpy())
        train_loss = tloss / len(train_ds)
        train_mcc = matthews_corrcoef(tt, tp)

        # ---- val ----
        model.eval()
        vloss, vp, vt = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["x"])
                vloss += crit(logits, batch["y"]).item() * batch["x"].size(0)
                vp.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                vt.extend(batch["y"].cpu().numpy())
        val_loss = vloss / len(val_ds)
        val_mcc = matthews_corrcoef(vt, vp)

        entry["losses"]["train"].append(train_loss)
        entry["losses"]["val"].append(val_loss)
        entry["metrics"]["train_MCC"].append(train_mcc)
        entry["metrics"]["val_MCC"].append(val_mcc)
        entry["epochs"].append(ep)
        print(
            f"Epoch {ep}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_MCC {val_mcc:.4f}"
        )

    # ---- test ----
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            tp.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tt.extend(batch["y"].cpu().numpy())
    test_mcc = matthews_corrcoef(tt, tp)
    entry["metrics"]["test_MCC"] = test_mcc
    entry["predictions"], entry["ground_truth"] = tp, tt
    print(f"HIDDEN_DIM={hid} Test MCC: {test_mcc:.4f}")

    # store
    experiment_data["HIDDEN_DIM_TUNING"]["SPR_BENCH"][str(hid)] = entry

    # ---- plots ----
    ep_range = entry["epochs"]
    plt.figure()
    plt.plot(ep_range, entry["losses"]["train"], label="train")
    plt.plot(ep_range, entry["losses"]["val"], label="val")
    plt.title(f"Loss (hid={hid})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_hid{hid}.png"))
    plt.close()

    plt.figure()
    plt.plot(ep_range, entry["metrics"]["val_MCC"], label="val_MCC")
    plt.title(f"MCC (hid={hid})")
    plt.xlabel("epoch")
    plt.ylabel("MCC")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"mcc_curve_hid{hid}.png"))
    plt.close()

# --------------------------------------------------------------------------- #
#                                save all data                                #
# --------------------------------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All experiments finished and data saved to working/experiment_data.npy")
