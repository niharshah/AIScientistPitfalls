import os, json, datetime, random, string, math, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------- misc --------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------------------ data loading -------------------------------
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def _spr_files(path):
    return all(
        os.path.isfile(os.path.join(path, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )


if _spr_files(SPR_PATH):
    print("Loading real SPR_BENCH …")
    from datasets import load_dataset, DatasetDict

    dss = DatasetDict()
    for sp in ["train", "dev", "test"]:
        dss[sp] = load_dataset(
            "csv", data_files=os.path.join(SPR_PATH, f"{sp}.csv"), split="train"
        )
    raw = {
        sp: {"sequence": dss[sp]["sequence"], "label": dss[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }
else:
    print("Dataset not found – generating small synthetic corpus.")
    shapes = list(string.ascii_uppercase[:6])
    cols = [str(i) for i in range(4)]

    def rand_seq():
        n = random.randint(4, 9)
        return " ".join(random.choice(shapes) + random.choice(cols) for _ in range(n))

    def rule(s):
        us = len(set(t[0] for t in s.split()))
        uc = len(set(t[1] for t in s.split()))
        return int(us == uc)

    def make(n):
        xs = [rand_seq() for _ in range(n)]
        ys = [rule(x) for x in xs]
        return {"sequence": xs, "label": ys}

    raw = {"train": make(3000), "dev": make(600), "test": make(800)}

# ---------------------- symbolic utilities ---------------------------
PAD, UNK, CLS = "<PAD>", "<UNK>", "<CLS>"
shape_set = sorted({tok[0] for seq in raw["train"]["sequence"] for tok in seq.split()})
colour_set = sorted({tok[1] for seq in raw["train"]["sequence"] for tok in seq.split()})
shape2idx = {s: i for i, s in enumerate(shape_set)}
col2idx = {c: i for i, c in enumerate(colour_set)}
SYM_DIM = len(shape_set) + len(colour_set) + 3


def sym_feats(seq: str):
    shp = [0] * len(shape_set)
    col = [0] * len(colour_set)
    for t in seq.split():
        shp[shape2idx.get(t[0], 0)] += 1
        col[col2idx.get(t[1], 0)] += 1
    n_us = sum(1 for c in shp if c > 0)
    n_uc = sum(1 for c in col if c > 0)
    eq = int(n_us == n_uc)
    return shp + col + [n_us, n_uc, eq]


def count_shape_variety(s):
    return len(set(t[0] for t in s.strip().split() if t))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


# ------------------------ vocab & encode -----------------------------
def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1, CLS: 2}
    tokens = {tok for s in seqs for tok in s.split()}
    vocab.update({t: i + 3 for i, t in enumerate(sorted(tokens))})
    return vocab


vocab = build_vocab(raw["train"]["sequence"])


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


# --------------------------- dataset ---------------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.raw = seqs
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.S = [torch.tensor(sym_feats(s), dtype=torch.float32) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"ids": self.X[idx], "sym": self.S[idx], "label": self.y[idx]}


def collate(batch):
    maxlen = max(len(b["ids"]) for b in batch) + 1  # +1 for CLS
    inp = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    pos = torch.arange(0, maxlen, dtype=torch.long).unsqueeze(0).expand(len(batch), -1)
    for i, b in enumerate(batch):
        inp[i, 0] = vocab[CLS]
        inp[i, 1 : 1 + len(b["ids"])] = b["ids"]
    lengths = torch.tensor([len(b["ids"]) + 1 for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    syms = torch.stack([b["sym"] for b in batch])
    return {"ids": inp, "pos": pos, "len": lengths, "sym": syms, "labels": labels}


data = {
    sp: SPRDataset(raw[sp]["sequence"], raw[sp]["label"])
    for sp in ["train", "dev", "test"]
}
loaders = {
    sp: DataLoader(data[sp], batch_size=64, shuffle=(sp == "train"), collate_fn=collate)
    for sp in ["train", "dev", "test"]
}


# --------------------------- model -----------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x, positions):
        return x + self.pe[positions]


class RuleAwareTransformer(nn.Module):
    def __init__(self, vocab_sz, d_model, nhead, layers, sym_dim, cls_dim, n_cls):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_sz, d_model, padding_idx=vocab[PAD])
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=0.1, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.symb = nn.Sequential(nn.Linear(sym_dim, cls_dim), nn.ReLU())
        self.cls_head = nn.Linear(d_model + cls_dim, n_cls)

    def forward(self, ids, pos, src_key_padding_mask, sym):
        x = self.token_emb(ids)
        x = self.pos_enc(x, pos)
        x = self.encoder(
            x.transpose(0, 1), src_key_padding_mask=src_key_padding_mask
        ).transpose(0, 1)
        cls_vec = x[:, 0, :]  # [B,d_model]
        sym_vec = self.symb(sym)  # [B,cls_dim]
        fused = torch.cat([cls_vec, sym_vec], dim=1)
        return self.cls_head(fused)


model = RuleAwareTransformer(
    len(vocab), d_model=128, nhead=4, layers=2, sym_dim=SYM_DIM, cls_dim=64, n_cls=2
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


# ----------------------- evaluation ----------------------------------
@torch.no_grad()
def evaluate(split):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    preds, gts = [], []
    for batch in loaders[split]:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        mask = batch["ids"] == vocab[PAD]
        logits = model(batch["ids"], batch["pos"], mask, batch["sym"])
        loss = criterion(logits, batch["labels"])
        loss_sum += loss.item() * batch["labels"].size(0)
        p = logits.argmax(-1)
        preds.extend(p.cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
        correct += (p == batch["labels"]).sum().item()
        tot += batch["labels"].size(0)
    swa = shape_weighted_accuracy(data[split].raw, gts, preds)
    return correct / tot, loss_sum / tot, swa, preds


# ----------------------- training loop -------------------------------
best_val = float("inf")
patience = 3
waits = 0
best_state = None
for epoch in range(1, 16):
    model.train()
    tloss_sum, tsamp = 0, 0
    for batch in loaders["train"]:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        mask = batch["ids"] == vocab[PAD]
        logits = model(batch["ids"], batch["pos"], mask, batch["sym"])
        loss = criterion(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tloss_sum += loss.item() * batch["labels"].size(0)
        tsamp += batch["labels"].size(0)
    tr_loss = tloss_sum / tsamp
    tr_acc, _, tr_swa, _ = evaluate("train")
    val_acc, val_loss, val_swa, _ = evaluate("dev")
    exp = experiment_data["SPR_BENCH"]
    exp["losses"]["train"].append(tr_loss)
    exp["losses"]["val"].append(val_loss)
    exp["metrics"]["train"].append({"acc": tr_acc, "swa": tr_swa})
    exp["metrics"]["val"].append({"acc": val_acc, "swa": val_swa})
    exp["timestamps"].append(str(datetime.datetime.now()))
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_SWA = {val_swa:.3f}")
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        waits = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        waits += 1
        if waits >= patience:
            print("Early stopping.")
            break

# ------------------------ test run -----------------------------------
if best_state:
    model.load_state_dict(best_state)
test_acc, test_loss, test_swa, test_preds = evaluate("test")
print(f"TEST: Acc={test_acc:.3f} | SWA={test_swa:.3f}")

experiment_data["SPR_BENCH"]["metrics"]["test"] = {"acc": test_acc, "swa": test_swa}
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = raw["test"]["label"]
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)
