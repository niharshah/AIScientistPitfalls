import os, pathlib, random, json, math, time
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, Dataset as HFDataset
from sklearn.metrics import matthews_corrcoef

# ------------------- bookkeeping ---------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "batch_size": {"SPR_BENCH": {}}  # hyper-parameter being tuned  # dataset name
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------- dataset --------------------------------------------------
def load_spr_bench():
    root = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    if root.exists():
        print(f"Loading real SPR_BENCH from {root}")

        def _load(csv):
            return load_dataset(
                "csv", data_files=str(root / csv), split="train", cache_dir=".cache"
            )

        return DatasetDict(
            {
                "train": _load("train.csv"),
                "dev": _load("dev.csv"),
                "test": _load("test.csv"),
            }
        )
    print("Real dataset not found, generating synthetic data ...")

    def synth(n):
        syms = list("ABCDEFGH")
        seqs, labels = [], []
        for _ in range(n):
            s = "".join(random.choice(syms) for _ in range(random.randint(5, 12)))
            seqs.append(s)
            labels.append(int(s.count("A") % 2 == 0))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    return DatasetDict(
        {
            "train": HFDataset.from_dict(synth(2000)),
            "dev": HFDataset.from_dict(synth(500)),
            "test": HFDataset.from_dict(synth(500)),
        }
    )


spr_bench = load_spr_bench()

# ------------------- vocab / encoding ----------------------------------------
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {c: i + 1 for i, c in enumerate(vocab)}  # 0 = PAD
pad_idx, max_len = 0, min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    return ids + [pad_idx] * (max_len - len(ids))


class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.seq, self.lbl = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        return {
            "x": torch.tensor(encode(self.seq[i]), dtype=torch.long),
            "y": torch.tensor(self.lbl[i], dtype=torch.float32),
        }


train_ds, val_ds, test_ds = map(
    SPRTorch, (spr_bench["train"], spr_bench["dev"], spr_bench["test"])
)


# ------------------- model ----------------------------------------------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb=32, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(out.mean(1)).squeeze(1)


# ------------------- hyper-parameter sweep ------------------------------------
batch_sizes = [32, 64, 128, 256, 512]
epochs = 5

for bs in batch_sizes:
    key = f"bs_{bs}"
    experiment_data["batch_size"]["SPR_BENCH"][key] = {
        "metrics": {"train_MCC": [], "val_MCC": [], "test_MCC": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": list(range(1, epochs + 1)),
    }

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(512, bs * 2))
    model = CharBiLSTM(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        # ---- train ----------------------------------------------------------
        model.train()
        tloss, preds, trues = 0.0, [], []
        for b in train_loader:
            x, y = b["x"].to(device), b["y"].to(device)
            opt.zero_grad()
            logit = model(x)
            loss = crit(logit, y)
            loss.backward()
            opt.step()
            tloss += loss.item() * x.size(0)
            preds.extend(torch.sigmoid(logit).detach().cpu().numpy() > 0.5)
            trues.extend(y.cpu().numpy())
        train_loss = tloss / len(train_ds)
        train_mcc = matthews_corrcoef(trues, preds)

        # ---- validation -----------------------------------------------------
        model.eval()
        vloss, vp, vt = 0.0, [], []
        with torch.no_grad():
            for b in val_loader:
                x, y = b["x"].to(device), b["y"].to(device)
                logit = model(x)
                loss = crit(logit, y)
                vloss += loss.item() * x.size(0)
                vp.extend(torch.sigmoid(logit).cpu().numpy() > 0.5)
                vt.extend(y.cpu().numpy())
        val_loss = vloss / len(val_ds)
        val_mcc = matthews_corrcoef(vt, vp)

        d = experiment_data["batch_size"]["SPR_BENCH"][key]
        d["losses"]["train"].append(train_loss)
        d["losses"]["val"].append(val_loss)
        d["metrics"]["train_MCC"].append(train_mcc)
        d["metrics"]["val_MCC"].append(val_mcc)
        print(
            f"[bs={bs}] epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_MCC={val_mcc:.4f}"
        )

    # ---- test evaluation ----------------------------------------------------
    test_loader = DataLoader(test_ds, batch_size=min(512, bs * 2))
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for b in test_loader:
            x, y = b["x"].to(device), b["y"].to(device)
            tp.extend(torch.sigmoid(model(x)).cpu().numpy() > 0.5)
            tt.extend(y.cpu().numpy())
    test_mcc = matthews_corrcoef(tt, tp)
    d["metrics"]["test_MCC"] = test_mcc
    d["predictions"], d["ground_truth"] = tp, tt
    print(f"[bs={bs}] Test MCC: {test_mcc:.4f}")
    torch.cuda.empty_cache()

# ------------------- persist --------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
