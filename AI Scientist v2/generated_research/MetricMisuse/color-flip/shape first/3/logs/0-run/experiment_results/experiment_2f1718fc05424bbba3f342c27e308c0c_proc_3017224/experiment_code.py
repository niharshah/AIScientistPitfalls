import os, random, pathlib, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------------------------
# bookkeeping dict (required format)
experiment_data = {
    "weight_decay": {  # hyper-parameter tuned
        "SPR_BENCH": {}  # dataset name (only one here)
    }
}

# working dir -----------------------------------------------------------------------------------
work_dir = os.path.join(os.getcwd(), "working")
os.makedirs(work_dir, exist_ok=True)

# device ----------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------------------------------------------------------------------------
# helpers
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def scwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) + 1e-9
    )


def try_load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset

        def _load(csv):
            return load_dataset(
                "csv",
                data_files=str(root / csv),
                split="train",
                cache_dir=".cache_dsets",
            )

        return True, {
            sp.split(".")[0]: _load(sp) for sp in ["train.csv", "dev.csv", "test.csv"]
        }
    except Exception as e:
        print("Could not load SPR_BENCH, falling back to synthetic data.", e)
        return False, {}


# synthetic data
def make_synth_dataset(n):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(3, 10)
        s = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(s)
        labels.append(int(count_shape_variety(s) > count_color_variety(s)))
    return {"sequence": seqs, "label": labels}


# dataset wrapper
class SPRDataset(Dataset):
    def __init__(self, seqs, labs, vocab, max_len):
        self.seqs, self.labs, self.vocab, self.max_len = seqs, labs, vocab, max_len

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]
        ids += (
            [self.vocab["<pad>"]] * (self.max_len - len(ids))
            if len(ids) < self.max_len
            else []
        )
        return torch.tensor(ids[: self.max_len], dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "x": self.encode(self.seqs[idx]),
            "y": torch.tensor(self.labs[idx]),
            "raw": self.seqs[idx],
        }


# model
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.fc(h.squeeze(0))


# -----------------------------------------------------------------------------------------------
# data preparation
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_real, raw = try_load_spr_bench(SPR_PATH)
if have_real:
    tr, dev, te = [raw[s] for s in ["train", "dev", "test"]]
    train_d = {"sequence": tr["sequence"], "label": tr["label"]}
    dev_d = {"sequence": dev["sequence"], "label": dev["label"]}
    test_d = {"sequence": te["sequence"], "label": te["label"]}
else:
    train_d, dev_d, test_d = [make_synth_dataset(n) for n in (2000, 400, 400)]

all_tokens = set(tok for seq in train_d["sequence"] for tok in seq.split())
vocab = {tok: i + 2 for i, tok in enumerate(sorted(all_tokens))}
vocab["<pad>"], vocab["<unk>"] = 0, 1
pad_idx = vocab["<pad>"]
max_len = max(len(s.split()) for s in train_d["sequence"])

train_ds = SPRDataset(train_d["sequence"], train_d["label"], vocab, max_len)
dev_ds = SPRDataset(dev_d["sequence"], dev_d["label"], vocab, max_len)
test_ds = SPRDataset(test_d["sequence"], test_d["label"], vocab, max_len)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# -----------------------------------------------------------------------------------------------
EPOCHS = 5
weight_decays = [0, 1e-5, 1e-4, 1e-3]

for wd in weight_decays:
    tag = str(wd)
    record = {
        "config": {"weight_decay": wd},
        "metrics": {"val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
    # model/optim/criterion
    model = GRUClassifier(len(vocab), 64, 128, len(set(train_d["label"])), pad_idx).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    # training ----------------------------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss = n = 0
        for b in train_loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in b.items()
            }
            optimizer.zero_grad()
            loss = criterion(model(b["x"]), b["y"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * b["y"].size(0)
            n += b["y"].size(0)
        train_loss = tot_loss / n
        record["losses"]["train"].append(train_loss)
        # validation ----------------------------------------------------------------------------
        model.eval()
        val_loss = n = 0
        preds, trues, seqs = [], [], []
        with torch.no_grad():
            for b in dev_loader:
                b = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in b.items()
                }
                logits = model(b["x"])
                loss = criterion(logits, b["y"])
                val_loss += loss.item() * b["y"].size(0)
                n += b["y"].size(0)
                p = logits.argmax(1).cpu().tolist()
                preds += p
                trues += b["y"].cpu().tolist()
                seqs += b["raw"]
        val_loss /= n
        val_scwa = scwa(seqs, trues, preds)
        record["losses"]["val"].append(val_loss)
        record["metrics"]["val"].append(val_scwa)
        record["epochs"].append(epoch)
        print(
            f"[wd={wd}] epoch {epoch}: val_loss={val_loss:.4f} val_SCWA={val_scwa:.4f}"
        )
    # final test --------------------------------------------------------------------------------
    model.eval()
    preds, trues, seqs = [], [], []
    with torch.no_grad():
        for b in test_loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in b.items()
            }
            logits = model(b["x"])
            preds += logits.argmax(1).cpu().tolist()
            trues += b["y"].cpu().tolist()
            seqs += b["raw"]
    test_scwa = scwa(seqs, trues, preds)
    print(f"[wd={wd}] Test SCWA={test_scwa:.4f}")
    record["predictions"] = preds
    record["ground_truth"] = trues
    record["test_SCWA"] = test_scwa
    experiment_data["weight_decay"]["SPR_BENCH"][tag] = record

# save ------------------------------------------------------------------------------------------
np.save(os.path.join(work_dir, "experiment_data.npy"), experiment_data)
