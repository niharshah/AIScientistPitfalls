import os, random, pathlib, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------------------------------
# Directory for artefacts
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device ---------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------------------------------------------------------------------------
# Utility functions
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def scwa(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) + 1e-9)


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

        d = {}
        for name in ["train", "dev", "test"]:
            d[name] = _ld(name + ".csv")
        return True, d
    except Exception as e:
        print("Could not load SPR_BENCH, falling back to synthetic data.", e)
        return False, {}


# Synthetic data -------------------------------------------------------------------------------
def make_synth_dataset(n_rows):
    shapes = list("ABCDE")
    colors = list("12345")
    seqs, labels = [], []
    for _ in range(n_rows):
        L = random.randint(3, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labels.append(int(count_shape_variety(seq) > count_color_variety(seq)))
    return {"sequence": seqs, "label": labels}


# Dataset wrapper -------------------------------------------------------------------------------
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

    def encode(self, seq):
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]
        ids = (
            ids[: self.max_len] + [self.vocab["<pad>"]] * (self.max_len - len(ids))
            if len(ids) < self.max_len
            else ids[: self.max_len]
        )
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "x": self.encode(self.seqs[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


# Model -----------------------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        return self.fc(h.squeeze(0))


# ----------------------------------------------------------------------------------------------
# Load / create data once (datasets reused across LR sweep)
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_real, raw_dsets = try_load_spr_bench(SPR_PATH)
if have_real:
    train_dict = {
        "sequence": raw_dsets["train"]["sequence"],
        "label": raw_dsets["train"]["label"],
    }
    dev_dict = {
        "sequence": raw_dsets["dev"]["sequence"],
        "label": raw_dsets["dev"]["label"],
    }
    test_dict = {
        "sequence": raw_dsets["test"]["sequence"],
        "label": raw_dsets["test"]["label"],
    }
else:
    train_dict = make_synth_dataset(2000)
    dev_dict = make_synth_dataset(400)
    test_dict = make_synth_dataset(400)

# Build vocab / datasets (based on training set)
all_tokens = set(tok for seq in train_dict["sequence"] for tok in seq.split())
vocab = {tok: i + 2 for i, tok in enumerate(sorted(all_tokens))}
vocab["<pad>"] = 0
vocab["<unk>"] = 1
pad_idx = vocab["<pad>"]
max_len = max(len(s.split()) for s in train_dict["sequence"])

train_ds = SPRDataset(train_dict["sequence"], train_dict["label"], vocab, max_len)
dev_ds = SPRDataset(dev_dict["sequence"], dev_dict["label"], vocab, max_len)
test_ds = SPRDataset(test_dict["sequence"], test_dict["label"], vocab, max_len)

train_loader = lambda bs: DataLoader(train_ds, batch_size=bs, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# ----------------------------------------------------------------------------------------------
# Hyper-parameter sweep set-up
learning_rates = [3e-4, 1e-3, 3e-3]
EPOCHS = 5
experiment_data = {"learning_rate": {}}


def run_one_lr(lr_value: float):
    tag = f"lr_{lr_value:.0e}"
    print(f"\n=== Training with {tag} ===")
    exp_dict = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    model = GRUClassifier(
        len(vocab),
        emb_dim=64,
        hid_dim=128,
        n_classes=len(set(train_dict["label"])),
        pad_idx=pad_idx,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_value)

    for epoch in range(1, EPOCHS + 1):
        # Train ----------------------------------------------------------
        model.train()
        tot_loss = 0
        n = 0
        for batch in train_loader(128):
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            bs = batch["y"].size(0)
            tot_loss += loss.item() * bs
            n += bs
        train_loss = tot_loss / n
        exp_dict["losses"]["train"].append(train_loss)

        # Validation -----------------------------------------------------
        model.eval()
        val_loss = 0
        n = 0
        all_pred = []
        all_true = []
        all_seq = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["x"])
                loss = criterion(logits, batch["y"])
                bs = batch["y"].size(0)
                val_loss += loss.item() * bs
                n += bs
                preds = logits.argmax(1).cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(batch["y"].cpu().tolist())
                all_seq.extend(batch["raw"])
        val_loss /= n
        val_scwa = scwa(all_seq, all_true, all_pred)
        exp_dict["losses"]["val"].append(val_loss)
        exp_dict["metrics"]["val"].append(val_scwa)
        exp_dict["epochs"].append(epoch)
        print(f"Epoch {epoch}: val_loss={val_loss:.4f} | val_SCWA={val_scwa:.4f}")

    # Final test evaluation --------------------------------------------
    model.eval()
    all_pred = []
    all_true = []
    all_seq = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            preds = logits.argmax(1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(batch["y"].cpu().tolist())
            all_seq.extend(batch["raw"])
    test_scwa = scwa(all_seq, all_true, all_pred)
    print(f"{tag} | Test SCWA = {test_scwa:.4f}")
    exp_dict["predictions"] = all_pred
    exp_dict["ground_truth"] = all_true
    exp_dict["test_scwa"] = test_scwa
    return tag, exp_dict


# Run sweep -------------------------------------------------------------------------------------
for lr in learning_rates:
    tag, res = run_one_lr(lr)
    experiment_data["learning_rate"][tag] = res

# Save everything -------------------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll results saved to", os.path.join(working_dir, "experiment_data.npy"))
