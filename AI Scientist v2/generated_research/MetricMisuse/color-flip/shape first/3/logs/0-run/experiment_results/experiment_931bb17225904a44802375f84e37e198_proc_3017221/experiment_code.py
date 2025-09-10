import os, random, pathlib, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------------------------------------------------------
# Reproducibility helpers
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# ----------------------------------------------------------------------------------------------------------------------
# Working dir and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------------------------------------------------------------------------------------------------
# Utility functions (same as baseline)
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

        def _ld(split_csv):
            return load_dataset(
                "csv",
                data_files=str(root / split_csv),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = {}
        for sp in ["train.csv", "dev.csv", "test.csv"]:
            d[sp.split(".")[0]] = _ld(sp)
        return True, d
    except Exception as e:
        print("Could not load SPR_BENCH, falling back to synthetic data.", e)
        return False, {}


# ----------------------------------------------------------------------------------------------------------------------
# Synthetic data fallback
def make_synth_dataset(n_rows):
    shapes, colors = list("ABCDE"), list("12345")
    sequences, labels = [], []
    for _ in range(n_rows):
        L = random.randint(3, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        sequences.append(seq)
        labels.append(int(count_shape_variety(seq) > count_color_variety(seq)))
    return {"sequence": sequences, "label": labels}


# ----------------------------------------------------------------------------------------------------------------------
# Dataset wrapper
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
        ids = ids[: self.max_len] + [self.vocab["<pad>"]] * max(
            0, self.max_len - len(ids)
        )
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "x": self.encode(self.seqs[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


# ----------------------------------------------------------------------------------------------------------------------
# Model that supports variable depth
class GRUClassifier(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, hid_dim, num_classes, pad_idx, num_layers=1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)  # h: (num_layers, B, H)
        logits = self.fc(h[-1])  # last layer hidden
        return logits


# ----------------------------------------------------------------------------------------------------------------------
# Load data (real or synthetic)
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

# Vocabulary
all_tokens = set(tok for seq in train_dict["sequence"] for tok in seq.split())
vocab = {tok: i + 2 for i, tok in enumerate(sorted(all_tokens))}
vocab["<pad>"], vocab["<unk>"] = 0, 1
pad_idx = vocab["<pad>"]
max_len = max(len(seq.split()) for seq in train_dict["sequence"])

# Datasets / loaders
train_ds = SPRDataset(train_dict["sequence"], train_dict["label"], vocab, max_len)
dev_ds = SPRDataset(dev_dict["sequence"], dev_dict["label"], vocab, max_len)
test_ds = SPRDataset(test_dict["sequence"], test_dict["label"], vocab, max_len)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# ----------------------------------------------------------------------------------------------------------------------
# Experiment bookkeeping
experiment_data = {
    "num_layers": {  # hyper-parameter tuning type
        "SPR_BENCH": {  # dataset
            "per_layer": {},  # filled below
            "best_layer": None,
            "best_val_scwa": -1.0,
            "test_scwa": None,
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# ----------------------------------------------------------------------------------------------------------------------
# Training / evaluation helpers
def evaluate(model, data_loader, criterion):
    model.eval()
    tot_loss, n = 0, 0
    preds, gts, raws = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            tot_loss += loss.item() * batch["y"].size(0)
            n += batch["y"].size(0)
            pred = logits.argmax(1).cpu().tolist()
            preds.extend(pred)
            gts.extend(batch["y"].cpu().tolist())
            raws.extend(batch["raw"])
    loss = tot_loss / n
    scwa_score = scwa(raws, gts, preds)
    return loss, scwa_score, preds, gts


# ----------------------------------------------------------------------------------------------------------------------
# Hyper-parameter sweep
NUM_LAYERS_SWEEP = [1, 2, 3]
EPOCHS = 5
num_classes = len(set(train_dict["label"]))

for nl in NUM_LAYERS_SWEEP:
    print(f"\n=== Training with num_layers = {nl} ===")
    model = GRUClassifier(
        len(vocab),
        emb_dim=64,
        hid_dim=128,
        num_classes=num_classes,
        pad_idx=pad_idx,
        num_layers=nl,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # per-layer bookkeeping
    layer_record = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
    }

    best_val_scwa_layer = -1.0
    best_state_layer = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss, n = 0, 0
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
            tot_loss += loss.item() * batch["y"].size(0)
            n += batch["y"].size(0)
        train_loss = tot_loss / n

        # validation
        val_loss, val_scwa, _, _ = evaluate(model, dev_loader, criterion)

        # record
        layer_record["losses"]["train"].append(train_loss)
        layer_record["losses"]["val"].append(val_loss)
        layer_record["metrics"]["train"].append(None)  # no special train metric
        layer_record["metrics"]["val"].append(val_scwa)
        layer_record["epochs"].append(epoch)

        print(
            f"  Epoch {epoch:>2}: tr_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_SCWA={val_scwa:.4f}"
        )

        if val_scwa > best_val_scwa_layer:
            best_val_scwa_layer = val_scwa
            best_state_layer = model.state_dict()

    # save per-layer info
    layer_record["best_val_scwa"] = best_val_scwa_layer
    experiment_data["num_layers"]["SPR_BENCH"]["per_layer"][nl] = layer_record

    # update overall best
    if (
        best_val_scwa_layer
        > experiment_data["num_layers"]["SPR_BENCH"]["best_val_scwa"]
    ):
        experiment_data["num_layers"]["SPR_BENCH"][
            "best_val_scwa"
        ] = best_val_scwa_layer
        experiment_data["num_layers"]["SPR_BENCH"]["best_layer"] = nl
        experiment_data["num_layers"]["SPR_BENCH"][
            "best_state_dict"
        ] = best_state_layer  # store to reload later

# ----------------------------------------------------------------------------------------------------------------------
# Test evaluation with best depth
best_layer = experiment_data["num_layers"]["SPR_BENCH"]["best_layer"]
print(f"\nBest num_layers according to validation: {best_layer}")

best_model = GRUClassifier(
    len(vocab),
    emb_dim=64,
    hid_dim=128,
    num_classes=num_classes,
    pad_idx=pad_idx,
    num_layers=best_layer,
).to(device)
best_model.load_state_dict(
    experiment_data["num_layers"]["SPR_BENCH"]["best_state_dict"]
)

criterion = nn.CrossEntropyLoss()
test_loss, test_scwa, test_preds, test_gts = evaluate(
    best_model, test_loader, criterion
)
experiment_data["num_layers"]["SPR_BENCH"]["test_scwa"] = test_scwa
experiment_data["num_layers"]["SPR_BENCH"]["predictions"] = test_preds
experiment_data["num_layers"]["SPR_BENCH"]["ground_truth"] = test_gts

print(f"Test SCWA (best model) = {test_scwa:.4f}")

# ----------------------------------------------------------------------------------------------------------------------
# Persist results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
