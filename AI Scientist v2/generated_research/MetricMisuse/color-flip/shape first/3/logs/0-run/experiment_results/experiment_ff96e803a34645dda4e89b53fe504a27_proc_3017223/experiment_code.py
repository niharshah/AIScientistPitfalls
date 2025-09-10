import os, random, pathlib, numpy as np, torch, math, time
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------- reproducibility --------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ---------------------------- working dir ------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------- device -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------- helper functions -------------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def scwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) + 1e-9
    )


def try_load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset

        def _ld(name):
            return load_dataset(
                "csv",
                data_files=str(root / name),
                split="train",
                cache_dir=".cache_dsets",
            )

        return True, {sp: _ld(f"{sp}.csv") for sp in ["train", "dev", "test"]}
    except Exception as e:
        print("Could not load SPR_BENCH, using synthetic data.", e)
        return False, {}


def make_synth_dataset(n):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(3, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labels.append(int(count_shape_variety(seq) > count_color_variety(seq)))
    return {"sequence": seqs, "label": labels}


# ---------------------------- Dataset wrapper --------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs, labels, vocab, max_len):
        self.seqs, self.labels, self.vocab, self.max_len = seqs, labels, vocab, max_len

    def __len__(self):
        return len(self.seqs)

    def encode(self, s):
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in s.split()]
        ids += (
            [self.vocab["<pad>"]] * (self.max_len - len(ids))
            if len(ids) < self.max_len
            else []
        )
        return torch.tensor(ids[: self.max_len], dtype=torch.long)

    def __getitem__(self, i):
        return {
            "x": self.encode(self.seqs[i]),
            "y": torch.tensor(self.labels[i], dtype=torch.long),
            "raw": self.seqs[i],
        }


# ---------------------------- Model ------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        return self.fc(h.squeeze(0))


# ---------------------------- Load / build data -----------------------------------------------
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_real, raw = try_load_spr_bench(SPR_PATH)
if have_real:
    tr, dv, te = raw["train"], raw["dev"], raw["test"]
    train_dict = {"sequence": tr["sequence"], "label": tr["label"]}
    dev_dict = {"sequence": dv["sequence"], "label": dv["label"]}
    test_dict = {"sequence": te["sequence"], "label": te["label"]}
else:
    train_dict, dev_dict, test_dict = (
        make_synth_dataset(2000),
        make_synth_dataset(400),
        make_synth_dataset(400),
    )

all_tokens = {tok for seq in train_dict["sequence"] for tok in seq.split()}
vocab = {tok: i + 2 for i, tok in enumerate(sorted(all_tokens))}
vocab["<pad>"], vocab["<unk>"] = 0, 1
pad_idx = vocab["<pad>"]
max_len = max(len(s.split()) for s in train_dict["sequence"])

train_ds = SPRDataset(train_dict["sequence"], train_dict["label"], vocab, max_len)
dev_ds = SPRDataset(dev_dict["sequence"], dev_dict["label"], vocab, max_len)
test_ds = SPRDataset(test_dict["sequence"], test_dict["label"], vocab, max_len)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# ---------------------------- Experiment store -------------------------------------------------
experiment_data = {"emb_dim": {"SPR_BENCH": {}}}

# ---------------------------- Training / tuning loop ------------------------------------------
EPOCHS = 5
emb_dims = [32, 64, 128, 256]
hid_dim = 128
num_classes = len(set(train_dict["label"]))

for ed in emb_dims:
    print(f"\n=== Training with emb_dim={ed} ===")
    model = GRUClassifier(len(vocab), ed, hid_dim, num_classes, pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            total += loss.item() * batch["y"].size(0)
            n += batch["y"].size(0)
        tr_loss = total / n
        log["losses"]["train"].append(tr_loss)

        # ---- validation ----
        model.eval()
        val_tot, n = 0.0, 0
        val_pred, val_true, val_seq = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["x"])
                loss = criterion(logits, batch["y"])
                val_tot += loss.item() * batch["y"].size(0)
                n += batch["y"].size(0)
                p = logits.argmax(1).cpu().tolist()
                val_pred.extend(p)
                val_true.extend(batch["y"].cpu().tolist())
                val_seq.extend(batch["raw"])
        val_loss = val_tot / n
        val_scwa = scwa(val_seq, val_true, val_pred)
        log["losses"]["val"].append(val_loss)
        log["metrics"]["val"].append(val_scwa)
        log["epochs"].append(epoch)
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_SCWA={val_scwa:.4f}"
        )

    # ---- test evaluation ----
    model.eval()
    tst_pred, tst_true, tst_seq = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            p = logits.argmax(1).cpu().tolist()
            tst_pred.extend(p)
            tst_true.extend(batch["y"].cpu().tolist())
            tst_seq.extend(batch["raw"])
    test_scwa = scwa(tst_seq, tst_true, tst_pred)
    print(f"Test SCWA (emb_dim={ed}) = {test_scwa:.4f}")
    log["predictions"] = tst_pred
    log["ground_truth"] = tst_true
    log["test_SCWA"] = test_scwa

    experiment_data["emb_dim"]["SPR_BENCH"][str(ed)] = log

# ---------------------------- save all ---------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
