import os, json, datetime, random, string
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- house-keeping -------------------------------
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

# ---------------------- device --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- data loading --------------------------------
def load_real_spr(path):
    from datasets import load_dataset, DatasetDict

    def _ld(csv_name):
        return load_dataset(
            "csv", data_files=os.path.join(path, csv_name), split="train"
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _ld(f"{sp}.csv")
    return {sp: {"sequence": d[sp]["sequence"], "label": d[sp]["label"]} for sp in d}


SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(p):
    return all(
        os.path.isfile(os.path.join(p, f"{s}.csv")) for s in ["train", "dev", "test"]
    )


if spr_files_exist(SPR_PATH):
    raw = load_real_spr(SPR_PATH)
    print("Loaded real SPR_BENCH")
else:
    print("Real SPR_BENCH not found – using synthetic toy data.")
    shapes = list(string.ascii_uppercase[:6])
    colors = [str(i) for i in range(4)]

    def rand_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 9))
        )

    def rule(lbl_seq):
        us = len(set(t[0] for t in lbl_seq.split()))
        uc = len(set(t[1] for t in lbl_seq.split()))
        return int(us == uc)

    def make(n):
        seqs = [rand_seq() for _ in range(n)]
        return {"sequence": seqs, "label": [rule(s) for s in seqs]}

    raw = {"train": make(2000), "dev": make(400), "test": make(600)}

# ---------------------- helper metrics ------------------------------
PAD, UNK = "<PAD>", "<UNK>"


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


# ---------------------- vocab & encoder -----------------------------
vocab = {PAD: 0, UNK: 1}
for s in raw["train"]["sequence"]:
    for tok in s.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


# ---------------------- dataset -------------------------------------
def symbolic_feats(seq):
    us = count_shape_variety(seq)
    uc = len(set(tok[1] for tok in seq.split() if len(tok) > 1))
    ln = len(seq.split())
    eq = int(us == uc)
    return [us, uc, ln, eq]


class SPRSet(Dataset):
    def __init__(self, seqs, labels):
        self.raw = seqs
        self.labels = labels
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.sym = torch.tensor([symbolic_feats(s) for s in seqs], dtype=torch.float)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"ids": self.X[idx], "sym": self.sym[idx], "label": self.y[idx]}


def collate(batch):
    lens = [len(b["ids"]) for b in batch]
    mx = max(lens)
    ids = torch.full((len(batch), mx), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : lens[i]] = b["ids"]
    sym = torch.stack([b["sym"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    return {"ids": ids, "lens": torch.tensor(lens), "sym": sym, "labels": labels}


datasets = {sp: SPRSet(raw[sp]["sequence"], raw[sp]["label"]) for sp in raw}
loaders = {
    sp: DataLoader(
        datasets[sp], batch_size=64, shuffle=(sp == "train"), collate_fn=collate
    )
    for sp in datasets
}


# ---------------------- model ---------------------------------------
class NeuroSymbolic(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=128, sym_dim=4, classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=vocab[PAD])
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(hid * 2 + sym_dim, classes)

    def forward(self, ids, lens, sym):
        e = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h:[2,B,hid]
        h = torch.cat([h[0], h[1]], dim=-1)  # [B,hid*2]
        out = self.lin(torch.cat([h, sym], dim=-1))
        return out


classes = len(set(raw["train"]["label"]))
model = NeuroSymbolic(len(vocab), classes=classes).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------- training loop w/ early stop -----------------
best_val = np.inf
patience = 3
wait = 0
best_state = None
max_epochs = 20
for epoch in range(1, max_epochs + 1):
    model.train()
    run_loss = 0.0
    for bt in loaders["train"]:
        bt = {k: v.to(device) if torch.is_tensor(v) else v for k, v in bt.items()}
        logits = model(bt["ids"], bt["lens"], bt["sym"])
        loss = criterion(logits, bt["labels"])
        optim.zero_grad()
        loss.backward()
        optim.step()
        run_loss += loss.item() * bt["labels"].size(0)
    train_loss = run_loss / len(datasets["train"])

    # eval on val
    def eval_split(split):
        model.eval()
        total = 0
        corr = 0
        preds = []
        gts = []
        with torch.no_grad():
            for b in loaders[split]:
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
                lg = model(b["ids"], b["lens"], b["sym"])
                p = lg.argmax(-1)
                preds.extend(p.cpu().tolist())
                gts.extend(b["labels"].cpu().tolist())
                corr += (p == b["labels"]).sum().item()
                total += b["labels"].size(0)
        acc = corr / total
        swa = shape_weighted_accuracy(datasets[split].raw, gts, preds)
        return acc, swa, preds, gts

    val_acc, val_swa, _, _ = eval_split("dev")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(
        val_swa
    )  # store swa as val loss proxy
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"swa": None}
    )  # placeholder
    experiment_data["SPR_BENCH"]["metrics"]["val"].append({"swa": val_swa})
    experiment_data["SPR_BENCH"]["timestamps"].append(str(datetime.datetime.now()))
    print(f"Epoch {epoch}: validation_loss = {val_swa:.4f}")
    if val_swa < best_val - 1e-4:
        best_val = val_swa
        wait = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ---------------------- final evaluation ----------------------------
if best_state:
    model.load_state_dict(best_state)
test_acc, test_swa, test_preds, test_gts = eval_split("test")
print(f"TEST SWA = {test_swa:.4f}")

experiment_data["SPR_BENCH"]["metrics"]["test"] = {"swa": test_swa}
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as f:
    json.dump(experiment_data, f, indent=2)
