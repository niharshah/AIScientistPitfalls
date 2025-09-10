import os, random, pathlib, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------- misc / dirs / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------- util
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def scwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / (sum(w) + 1e-9)


def try_load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset

        def _ld(csv_name):
            return load_dataset(
                "csv",
                data_files=str(root / csv_name),
                split="train",
                cache_dir=".cache_dsets",
            )

        return True, {
            sp.split(".")[0]: _ld(sp) for sp in ["train.csv", "dev.csv", "test.csv"]
        }
    except Exception as e:
        print("Could not load SPR_BENCH, falling back to synthetic.", e)
        return False, {}


def make_synth_dataset(n):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, lbls = [], []
    for _ in range(n):
        L = random.randint(3, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        lbls.append(int(count_shape_variety(seq) > count_color_variety(seq)))
    return {"sequence": seqs, "label": lbls}


# -------------------------------------------------- dataset
class SPRDataset(Dataset):
    def __init__(self, sequences, labels, vocab, max_len):
        self.seqs, self.labels, self.vocab, self.max_len = (
            sequences,
            labels,
            vocab,
            max_len,
        )

    def __len__(self):
        return len(self.seqs)

    def _enc(self, s):
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in s.split()]
        ids = ids[: self.max_len] + [self.vocab["<pad>"]] * max(
            0, self.max_len - len(ids)
        )
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "x": self._enc(self.seqs[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


# -------------------------------------------------- model
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


# -------------------------------------------------- load / build data
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_real, raw = try_load_spr_bench(SPR_PATH)
if have_real:
    train_dict = {"sequence": raw["train"]["sequence"], "label": raw["train"]["label"]}
    dev_dict = {"sequence": raw["dev"]["sequence"], "label": raw["dev"]["label"]}
    test_dict = {"sequence": raw["test"]["sequence"], "label": raw["test"]["label"]}
else:
    train_dict = make_synth_dataset(2000)
    dev_dict = make_synth_dataset(400)
    test_dict = make_synth_dataset(400)

all_tokens = {tok for s in train_dict["sequence"] for tok in s.split()}
vocab = {t: i + 2 for i, t in enumerate(sorted(all_tokens))}
vocab["<pad>"], vocab["<unk>"] = 0, 1
pad_idx = 0
max_len = max(len(s.split()) for s in train_dict["sequence"])

train_ds = SPRDataset(train_dict["sequence"], train_dict["label"], vocab, max_len)
dev_ds = SPRDataset(dev_dict["sequence"], dev_dict["label"], vocab, max_len)
test_ds = SPRDataset(test_dict["sequence"], test_dict["label"], vocab, max_len)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# -------------------------------------------------- hyperparameter tuning: hid_dim
experiment_data = {"hid_dim": {"SPR_BENCH": {}}}
hidden_sizes = [64, 128, 256, 512]
EPOCHS = 5
for hd in hidden_sizes:
    print(f"\n=== Training with hid_dim={hd} ===")
    model = GRUClassifier(
        len(vocab),
        emb_dim=64,
        hid_dim=hd,
        num_classes=len(set(train_dict["label"])),
        pad_idx=pad_idx,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    records = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    for ep in range(1, EPOCHS + 1):
        # -------- train
        model.train()
        tloss, n = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            xb, yb = batch["x"].to(device), batch["y"].to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tloss += loss.item() * yb.size(0)
            n += yb.size(0)
        train_loss = tloss / n
        # -------- val
        model.eval()
        vloss, vn = 0, 0
        all_pred, all_true, all_seq = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                xb, yb = batch["x"].to(device), batch["y"].to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss += loss.item() * yb.size(0)
                vn += yb.size(0)
                preds = logits.argmax(1).cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(batch["y"].cpu().tolist())
                all_seq.extend(batch["raw"])
        val_loss = vloss / vn
        val_scwa = scwa(all_seq, all_true, all_pred)

        # record
        records["losses"]["train"].append(train_loss)
        records["losses"]["val"].append(val_loss)
        records["metrics"]["train"].append(float("nan"))  # train SCWA not computed
        records["metrics"]["val"].append(val_scwa)
        records["epochs"].append(ep)
        print(
            f"[hid={hd}] Epoch {ep}: val_loss={val_loss:.4f}  val_SCWA={val_scwa:.4f}"
        )

    # -------- test
    model.eval()
    all_pred, all_true, all_seq = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            xb = batch["x"].to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(batch["y"].cpu().tolist())
            all_seq.extend(batch["raw"])
    test_scwa = scwa(all_seq, all_true, all_pred)
    print(f"[hid={hd}] Test SCWA = {test_scwa:.4f}")

    records["predictions"] = all_pred
    records["ground_truth"] = all_true
    records["test_SCWA"] = test_scwa
    experiment_data["hid_dim"]["SPR_BENCH"][str(hd)] = records

    # free memory
    del model
    torch.cuda.empty_cache()

# -------------------------------------------------- save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
