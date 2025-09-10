import os, json, datetime, random, string, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------- storage dict ------------------------------
experiment_data = {
    "multi_synth_generalization": {
        "D1-D2-D3": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# ---------------- device -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ------------ build three independent corpora ----------------
def make_dataset(shapes, colours, seed, n):
    random.seed(seed)

    def rand_seq():
        ln = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + random.choice(colours) for _ in range(ln)
        )

    def rule(seq):
        us = len(set(tok[0] for tok in seq.split()))
        uc = len(set(tok[1] for tok in seq.split()))
        return int(us == uc)

    xs = [rand_seq() for _ in range(n)]
    ys = [rule(s) for s in xs]
    return {"sequence": xs, "label": ys}


all_shapes_groups = [list(string.ascii_uppercase[i : i + 6]) for i in range(0, 18, 6)]
all_colour_groups = [[str(i) for i in range(j, j + 4)] for j in range(0, 12, 4)]

D1 = make_dataset(all_shapes_groups[0], all_colour_groups[0], 111, 3000)
D2 = make_dataset(all_shapes_groups[1], all_colour_groups[1], 222, 600)
D3 = make_dataset(all_shapes_groups[2], all_colour_groups[2], 333, 800)

raw_data = {"train": D1, "dev": D2, "test": D3}

# --------- global symbol mapping (union across sets) ----------
PAD, UNK = "<PAD>", "<UNK>"
shape_set = sorted(
    {
        tok[0]
        for split in raw_data.values()
        for seq in split["sequence"]
        for tok in seq.split()
    }
)
colour_set = sorted(
    {
        tok[1]
        for split in raw_data.values()
        for seq in split["sequence"]
        for tok in seq.split()
    }
)
shape2idx = {s: i for i, s in enumerate(shape_set)}
colour2idx = {c: i for i, c in enumerate(colour_set)}
SYM_DIM = len(shape_set) + len(colour_set) + 3


def sym_features(seq: str):
    shp = [0] * len(shape_set)
    col = [0] * len(colour_set)
    for tok in seq.split():
        shp[shape2idx[tok[0]]] += 1
        col[colour2idx[tok[1]]] += 1
    n_us = sum(1 for c in shp if c > 0)
    n_uc = sum(1 for c in col if c > 0)
    eq = int(n_us == n_uc)
    return shp + col + [n_us, n_uc, eq]


# -------------- vocab from TRAIN ONLY ------------------------
def build_vocab(train_seqs):
    vocab = {PAD: 0, UNK: 1}
    tokens = {tok for s in train_seqs for tok in s.split()}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(tokens))})
    return vocab


vocab = build_vocab(raw_data["train"]["sequence"])


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


# ---------------- torch dataset ------------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.raw_seq = seqs
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.S = [torch.tensor(sym_features(s), dtype=torch.float32) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"input_ids": self.X[idx], "sym": self.S[idx], "label": self.y[idx]}


def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    inp = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        inp[i, : len(b["input_ids"])] = b["input_ids"]
    lens = torch.tensor([len(b["input_ids"]) for b in batch])
    labs = torch.stack([b["label"] for b in batch])
    syms = torch.stack([b["sym"] for b in batch])
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


# ------------------ model ------------------------------------
class NeuralSymbolicClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim, rnn_hid, sym_dim, sym_hid, n_cls):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, embed_dim, padding_idx=vocab[PAD])
        self.gru = nn.GRU(embed_dim, rnn_hid, batch_first=True)
        self.symb = nn.Sequential(nn.Linear(sym_dim, sym_hid), nn.ReLU())
        self.cls = nn.Linear(rnn_hid + sym_hid, n_cls)

    def forward(self, ids, lens, sym):
        emb = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = h.squeeze(0)
        s = self.symb(sym)
        return self.cls(torch.cat([h, s], dim=1))


model = NeuralSymbolicClassifier(len(vocab), 64, 128, SYM_DIM, 64, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------- metrics helpers --------------------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


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
    return acc, loss_sum / tot, swa, preds, gts


# ---------------- training loop -------------------------------
best_val_loss = float("inf")
patience = 3
counter = 0
best_state = None

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
    train_acc, _, train_swa, _, _ = evaluate("train")
    val_acc, val_loss, val_swa, _, _ = evaluate("dev")

    ed = experiment_data["multi_synth_generalization"]["D1-D2-D3"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append({"acc": train_acc, "swa": train_swa})
    ed["metrics"]["val"].append({"acc": val_acc, "swa": val_swa})
    ed["timestamps"].append(str(datetime.datetime.now()))

    print(f"Epoch {epoch}: val_loss={val_loss:.4f} | val_SWA={val_swa:.3f}")
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        counter = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# ------------------ final test --------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_acc, test_loss, test_swa, preds, gts = evaluate("test")
print(f"TEST: Acc={test_acc:.3f} | SWA={test_swa:.3f}")

ed = experiment_data["multi_synth_generalization"]["D1-D2-D3"]
ed["metrics"]["test"] = {"acc": test_acc, "swa": test_swa}
ed["predictions"] = preds
ed["ground_truth"] = gts

# ------------------ persist -----------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)

plt.figure()
plt.plot(ed["losses"]["train"], label="train")
plt.plot(ed["losses"]["val"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curves")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()
print("All done.")
