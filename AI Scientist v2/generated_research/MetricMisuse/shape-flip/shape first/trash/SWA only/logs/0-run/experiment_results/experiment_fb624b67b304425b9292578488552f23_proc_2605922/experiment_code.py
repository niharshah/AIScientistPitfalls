# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, json, datetime, random, string
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------- logging dict -----------------------------
experiment_data = {
    "Remove_RNN_Branch": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# ----------------------------- device --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------- load / synth data -------------------------
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(path):
    return all(
        os.path.isfile(os.path.join(path, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )


if spr_files_exist(SPR_PATH):
    print("Loading real SPR_BENCH …")
    from datasets import load_dataset, DatasetDict

    def load_spr(root):
        d = DatasetDict()
        for sp in ["train", "dev", "test"]:
            d[sp] = load_dataset(
                "csv", data_files=os.path.join(root, f"{sp}.csv"), split="train"
            )
        return d

    ds = load_spr(SPR_PATH)
    raw_data = {
        sp: {"sequence": ds[sp]["sequence"], "label": ds[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }
else:
    print("Real dataset not found – generating synthetic SPR data.")
    shapes = list(string.ascii_uppercase[:6])  # A-F
    colours = [str(i) for i in range(4)]  # 0-3

    def rand_seq():
        ln = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + random.choice(colours) for _ in range(ln)
        )

    def rule(seq):
        us = len(set(tok[0] for tok in seq.split()))
        uc = len(set(tok[1] for tok in seq.split()))
        return int(us == uc)

    def make(n):
        xs = [rand_seq() for _ in range(n)]
        ys = [rule(s) for s in xs]
        return {"sequence": xs, "label": ys}

    raw_data = {"train": make(3000), "dev": make(600), "test": make(800)}

# ---------------------- symbolic helpers -----------------------------
PAD, UNK = "<PAD>", "<UNK>"
shape_set = sorted(
    {tok[0] for seq in raw_data["train"]["sequence"] for tok in seq.split()}
)
colour_set = sorted(
    {tok[1] for seq in raw_data["train"]["sequence"] for tok in seq.split()}
)
shape2idx = {s: i for i, s in enumerate(shape_set)}
colour2idx = {c: i for i, c in enumerate(colour_set)}
SYM_DIM = len(shape_set) + len(colour_set) + 3  # shape hist + colour hist + 3 stats


def sym_features(seq: str):
    shp = [0] * len(shape_set)
    col = [0] * len(colour_set)
    for tok in seq.split():
        if tok[0] in shape2idx:
            shp[shape2idx[tok[0]]] += 1
        if tok[1] in colour2idx:
            col[colour2idx[tok[1]]] += 1
    n_us = sum(1 for c in shp if c > 0)
    n_uc = sum(1 for c in col if c > 0)
    eq = 1 if n_us == n_uc else 0
    return shp + col + [n_us, n_uc, eq]


def count_shape_variety(sequence):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


# ----------------------- vocab / encoding ----------------------------
def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1}
    tokens = {tok for s in seqs for tok in s.split()}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(tokens))})
    return vocab


vocab = build_vocab(raw_data["train"]["sequence"])


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


# ------------------------- Dataset & Loader --------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.raw_seq = seqs
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]  # ignored
        self.S = [torch.tensor(sym_features(s), dtype=torch.float32) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"input_ids": self.X[idx], "sym": self.S[idx], "label": self.y[idx]}


def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    inp = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)  # still keep
    for i, b in enumerate(batch):
        inp[i, : len(b["input_ids"])] = b["input_ids"]
    labs = torch.stack([b["label"] for b in batch])
    syms = torch.stack([b["sym"] for b in batch])
    lens = torch.tensor([len(b["input_ids"]) for b in batch])
    return {"input_ids": inp, "lengths": lens, "sym": syms, "labels": labs}


datasets = {
    sp: SPRDataset(raw_data[sp]["sequence"], raw_data[sp]["label"])
    for sp in ["train", "dev", "test"]
}
loaders = {
    sp: DataLoader(
        datasets[sp], batch_size=64, shuffle=(sp == "train"), collate_fn=collate
    )
    for sp in ["train", "dev", "test"]
}


# ------------------------- model (sym-only) --------------------------
class SymbOnlyClassifier(nn.Module):
    def __init__(self, symb_dim, symb_hid, n_cls):
        super().__init__()
        self.symb = nn.Sequential(nn.Linear(symb_dim, symb_hid), nn.ReLU())
        self.cls = nn.Linear(symb_hid, n_cls)

    def forward(self, ids, lens, sym):
        s = self.symb(sym)
        return self.cls(s)


model = SymbOnlyClassifier(SYM_DIM, 64, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------------- evaluation fn -------------------------------
@torch.no_grad()
def evaluate(split):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    preds, gts = [], []
    for batch in loaders[split]:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["lengths"], batch["sym"])
        loss = criterion(logits, batch["labels"])
        loss_sum += loss.item() * batch["labels"].size(0)
        p = logits.argmax(-1)
        preds.extend(p.cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
        correct += (p == batch["labels"]).sum().item()
        tot += batch["labels"].size(0)
    acc = correct / tot
    swa = shape_weighted_accuracy(datasets[split].raw_seq, gts, preds)
    return acc, loss_sum / tot, swa


# ------------------------ training loop ------------------------------
best_val_loss, patience, counter, best_state = float("inf"), 3, 0, None
for epoch in range(1, 21):
    model.train()
    run_loss, run_tot = 0.0, 0
    for batch in loaders["train"]:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["lengths"], batch["sym"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["labels"].size(0)
        run_tot += batch["labels"].size(0)
    train_loss = run_loss / run_tot
    train_acc, _, train_swa = evaluate("train")
    val_acc, val_loss, val_swa = evaluate("dev")

    ed = experiment_data["Remove_RNN_Branch"]["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append({"acc": train_acc, "swa": train_swa})
    ed["metrics"]["val"].append({"acc": val_acc, "swa": val_swa})
    ed["timestamps"].append(str(datetime.datetime.now()))

    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | val_SWA={val_swa:.3f}"
    )
    if val_loss < best_val_loss - 1e-4:
        best_val_loss, counter, best_state = (
            val_loss,
            0,
            {k: v.cpu() for k, v in model.state_dict().items()},
        )
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break

# --------------------------- final test ------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_acc, test_loss, test_swa = evaluate("test")
print(f"TEST: Acc={test_acc:.3f} | SWA={test_swa:.3f}")
ed = experiment_data["Remove_RNN_Branch"]["SPR_BENCH"]
ed["metrics"]["test"] = {"acc": test_acc, "swa": test_swa}

# --------- store predictions & gts for test split --------------------
with torch.no_grad():
    preds = []
    for batch in loaders["test"]:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["lengths"], batch["sym"])
        preds.extend(logits.argmax(-1).cpu().tolist())
ed["predictions"] = preds
ed["ground_truth"] = raw_data["test"]["label"]

# ------------------------- persist results ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as f:
    json.dump(experiment_data, f, indent=2)

# ----------------------------- plots ---------------------------------
plt.figure()
plt.plot(ed["losses"]["train"], label="train")
plt.plot(ed["losses"]["val"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curves (Remove_RNN_Branch)")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()
