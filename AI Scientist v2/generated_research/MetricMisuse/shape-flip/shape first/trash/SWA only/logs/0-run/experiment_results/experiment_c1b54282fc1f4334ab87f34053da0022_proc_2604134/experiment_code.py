import os, random, string, datetime, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- housekeeping ----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------- device ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- data load / synthetic fallback ----------------------------------
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def files_ok(path):
    return all(
        os.path.isfile(os.path.join(path, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )


if files_ok(SPR_PATH):
    from datasets import load_dataset, DatasetDict

    def load_spr(root):
        def _l(split):
            return load_dataset(
                "csv", data_files=os.path.join(root, f"{split}.csv"), split="train"
            )

        d = DatasetDict()
        for s in ["train", "dev", "test"]:
            d[s] = _l(s)
        return d

    ds_raw = load_spr(SPR_PATH)
    raw = {
        sp: {"sequence": ds_raw[sp]["sequence"], "label": ds_raw[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }
else:
    print("SPR_BENCH not found, generating synthetic toy data.")
    shapes = list(string.ascii_uppercase[:6])
    colors = [str(i) for i in range(4)]

    def rand_seq():
        ln = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(ln)
        )

    def rule(seq):  # equal num unique shapes & colors
        us = len(set(t[0] for t in seq.split()))
        uc = len(set(t[1] for t in seq.split()))
        return int(us == uc)

    def make(n):
        seqs = [rand_seq() for _ in range(n)]
        return {"sequence": seqs, "label": [rule(s) for s in seqs]}

    raw = {"train": make(2000), "dev": make(400), "test": make(600)}

# ---------- helper functions -------------------------------------------------
PAD, UNK = "<PAD>", "<UNK>"


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split()))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def build_vocab(seqs):
    tokens = {tok for s in seqs for tok in s.split()}
    vocab = {PAD: 0, UNK: 1}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(tokens))})
    return vocab


vocab = build_vocab(raw["train"]["sequence"])
vocab_size = len(vocab)


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


# ---------- dataset ---------------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.var_shapes = [count_shape_variety(s) for s in seqs]
        self.var_colors = [count_color_variety(s) for s in seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.X[idx],
            "shape_var": torch.tensor(self.var_shapes[idx], dtype=torch.float),
            "color_var": torch.tensor(self.var_colors[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lengths = [len(b["input_ids"]) for b in batch]
    maxlen = max(lengths)
    input_ids = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
    labels = torch.stack([b["label"] for b in batch])
    shape_var = torch.stack([b["shape_var"] for b in batch])
    color_var = torch.stack([b["color_var"] for b in batch])
    return {
        "input_ids": input_ids,
        "lengths": torch.tensor(lengths),
        "shape_var": shape_var,
        "color_var": color_var,
        "labels": labels,
    }


datasets = {
    sp: SPRDataset(raw[sp]["sequence"], raw[sp]["label"])
    for sp in ["train", "dev", "test"]
}
loaders = {
    sp: DataLoader(
        datasets[sp], batch_size=64, shuffle=(sp == "train"), collate_fn=collate
    )
    for sp in ["train", "dev", "test"]
}


# ---------- model -----------------------------------------------------------
class NeuroSymbolic(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hid=128, symb_dim=16, classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.rnn = nn.GRU(embed_dim, hid, batch_first=True)
        self.symb = nn.Sequential(nn.Linear(2, symb_dim), nn.ReLU())
        self.fc = nn.Linear(hid + symb_dim, classes)

    def forward(self, ids, lengths, symb_feats):
        emb = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        symb = self.symb(symb_feats)
        concat = torch.cat([h.squeeze(0), symb], dim=1)
        return self.fc(concat)


num_classes = len(set(raw["train"]["label"]))
model = NeuroSymbolic(vocab_size).to(device)

# weighted loss focusing on shape variety
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- evaluation ------------------------------------------------------
@torch.no_grad()
def eval_split(split, full=False):
    model.eval()
    total = 0
    loss_sum = 0
    correct = 0
    for batch in loaders[split]:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits = model(
            batch["input_ids"],
            batch["lengths"],
            torch.stack([batch["shape_var"], batch["color_var"]], dim=1),
        )
        loss = criterion(logits, batch["labels"])
        weights = batch["shape_var"]  # weight by shape variety
        loss_sum += (loss * weights).sum().item()
        total += weights.sum().item()
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
    acc = correct / len(datasets[split])
    avg_loss = loss_sum / total
    if not full:
        return acc, avg_loss
    # full metrics
    seqs = datasets[split].seqs
    y_true = datasets[split].labels
    # batched prediction for memory safety
    all_preds = []
    for i in range(0, len(seqs), 128):
        chunk = [encode(s) for s in seqs[i : i + 128]]
        lens = torch.tensor([len(c) for c in chunk])
        mx = lens.max()
        inp = torch.full((len(chunk), mx), vocab[PAD], dtype=torch.long)
        for j, row in enumerate(chunk):
            inp[j, : len(row)] = torch.tensor(row)
        sv = torch.tensor(
            [count_shape_variety(s) for s in seqs[i : i + 128]], dtype=torch.float
        )
        cv = torch.tensor(
            [count_color_variety(s) for s in seqs[i : i + 128]], dtype=torch.float
        )
        with torch.no_grad():
            logit = model(
                inp.to(device), lens.to(device), torch.stack([sv, cv], 1).to(device)
            )
            all_preds.extend(logit.argmax(-1).cpu().tolist())
    swa = shape_weighted_accuracy(seqs, y_true, all_preds)
    return acc, avg_loss, swa, all_preds, y_true


# ---------- training --------------------------------------------------------
best_val = float("inf")
patience = 3
patience_cntr = 0
best_state = None
max_epochs = 20
for epoch in range(1, max_epochs + 1):
    model.train()
    running_loss = 0
    denom = 0
    for batch in loaders["train"]:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(
            batch["input_ids"],
            batch["lengths"],
            torch.stack([batch["shape_var"], batch["color_var"]], dim=1),
        )
        loss = criterion(logits, batch["labels"])
        weights = batch["shape_var"]
        weighted_loss = (loss * weights).mean()
        weighted_loss.backward()
        optimizer.step()
        running_loss += weighted_loss.item() * len(batch["labels"])
        denom += len(batch["labels"])
    train_loss = running_loss / denom
    train_acc, _ = eval_split("train")
    val_acc, val_loss = eval_split("dev")
    # log
    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append({"acc": train_acc})
    ed["metrics"]["val"].append({"acc": val_acc})
    ed["timestamps"].append(str(datetime.datetime.now()))
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    # early stopping
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        patience_cntr = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        patience_cntr += 1
        if patience_cntr >= patience:
            print("Early stopping.")
            break

# ---------- final evaluation -----------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_acc, _, test_swa, preds, gts = eval_split("test", full=True)
print(
    f"TEST results ->  Acc: {test_acc:.3f}   Shape-Weighted Accuracy (SWA): {test_swa:.3f}"
)

# save logs
ed = experiment_data["SPR_BENCH"]
ed["metrics"]["test"] = {"acc": test_acc, "swa": test_swa}
ed["predictions"] = preds
ed["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)
