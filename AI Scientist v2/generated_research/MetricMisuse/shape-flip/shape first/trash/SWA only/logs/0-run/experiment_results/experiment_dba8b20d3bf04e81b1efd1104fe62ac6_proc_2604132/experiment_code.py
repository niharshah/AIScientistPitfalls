import os, json, datetime, random, string, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------------------- device ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- dataset utils --------------------------------
def spr_files_exist(path):
    return all(
        os.path.isfile(os.path.join(path, f"{s}.csv")) for s in ["train", "dev", "test"]
    )


def load_real_spr(path):
    from datasets import load_dataset, DatasetDict

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = load_dataset(
            "csv",
            data_files=os.path.join(path, f"{split}.csv"),
            split="train",
            cache_dir=os.path.join(path, ".cache_hf"),
        )
    return {
        sp: {"sequence": dset[sp]["sequence"], "label": dset[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }


def make_synthetic():
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

    return {"train": make(4000), "dev": make(800), "test": make(800)}


SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")
raw_data = load_real_spr(SPR_PATH) if spr_files_exist(SPR_PATH) else make_synthetic()
print({k: len(v["sequence"]) for k, v in raw_data.items()})

# ---------------------- symbolic feats --------------------------------
PAD, UNK = "<PAD>", "<UNK>"
shape_set = sorted(
    {tok[0] for seq in raw_data["train"]["sequence"] for tok in seq.split()}
)
colour_set = sorted(
    {tok[1] for seq in raw_data["train"]["sequence"] for tok in seq.split()}
)
shape2idx = {s: i for i, s in enumerate(shape_set)}
colour2idx = {c: i for i, c in enumerate(colour_set)}

SYM_DIM = len(shape_set) + len(colour_set) + 3  # histograms + stats


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


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


# ---------------------- vocab & encoding ------------------------------
def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1}
    tokens = {tok for s in seqs for tok in s.split()}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(tokens))})
    return vocab


vocab = build_vocab(raw_data["train"]["sequence"])


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


# ---------------------- PyTorch dataset -------------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.raw_seq = seqs
        self.X_ids = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.S = [torch.tensor(sym_features(s), dtype=torch.float32) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"ids": self.X_ids[idx], "sym": self.S[idx], "label": self.y[idx]}


def collate(batch):
    maxlen = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = b["ids"]
    lengths = torch.tensor([len(b["ids"]) for b in batch])
    syms = torch.stack([b["sym"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"ids": ids, "lengths": lengths, "sym": syms, "labels": labels}


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


# ---------------------- model -----------------------------------------
class TransformerNS(nn.Module):
    def __init__(self, vocab_sz, d_model, nhead, nlayers, sym_dim, n_cls, pad_idx):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_sz, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(256, d_model)  # max_len 256
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.sym_proj = nn.Sequential(nn.Linear(sym_dim, d_model), nn.ReLU())
        self.cls_head = nn.Linear(d_model * 2, n_cls)
        self.sym_head = nn.Linear(d_model, sym_dim)
        self.d_model = d_model
        self.pad_idx = pad_idx

    def forward(self, ids, lengths, sym):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(ids) + self.pos_emb(pos)
        mask = ids.eq(self.pad_idx)  # (B,L)
        x = x.transpose(0, 1)  # (L,B,D) for encoder
        enc = self.encoder(x, src_key_padding_mask=mask)  # (L,B,D)
        enc = enc.transpose(0, 1)  # (B,L,D)
        # mean pooling over valid tokens
        mask_inv = (~mask).unsqueeze(-1)  # (B,L,1)
        pooled = (enc * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1)
        sym_embed = self.sym_proj(sym)  # (B,D)
        fused = torch.cat([pooled, sym_embed], dim=1)
        logits = self.cls_head(fused)
        sym_pred = self.sym_head(pooled)
        return logits, sym_pred


model = TransformerNS(len(vocab), 128, 4, 2, SYM_DIM, 2, vocab[PAD]).to(device)
ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# ---------------------- eval -----------------------------------------
@torch.no_grad()
def evaluate(split):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    preds, gts = [], []
    for batch in loaders[split]:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits, sym_pred = model(batch["ids"], batch["lengths"], batch["sym"])
        loss = ce_loss(logits, batch["labels"]) + 0.1 * mse_loss(sym_pred, batch["sym"])
        loss_sum += loss.item() * batch["labels"].size(0)
        p = logits.argmax(-1)
        preds.extend(p.cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
        correct += (p == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    acc = correct / total
    swa = shape_weighted_accuracy(datasets[split].raw_seq, gts, preds)
    return acc, loss_sum / total, swa, preds, gts


# ---------------------- training loop --------------------------------
best_val_loss = float("inf")
patience = 3
wait = 0
best_state = None
for epoch in range(1, 21):
    model.train()
    tr_loss_sum = 0
    tr_total = 0
    for batch in loaders["train"]:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits, sym_pred = model(batch["ids"], batch["lengths"], batch["sym"])
        loss = ce_loss(logits, batch["labels"]) + 0.1 * mse_loss(sym_pred, batch["sym"])
        loss.backward()
        optimizer.step()
        tr_loss_sum += loss.item() * batch["labels"].size(0)
        tr_total += batch["labels"].size(0)
    train_loss = tr_loss_sum / tr_total
    train_acc, _, train_swa, _, _ = evaluate("train")
    val_acc, val_loss, val_swa, _, _ = evaluate("dev")

    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append({"acc": train_acc, "swa": train_swa})
    ed["metrics"]["val"].append({"acc": val_acc, "swa": val_swa})
    ed["timestamps"].append(str(datetime.datetime.now()))
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_SWA = {val_swa:.3f}")

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        wait = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ---------------------- test -----------------------------------------
if best_state:
    model.load_state_dict(best_state)
test_acc, test_loss, test_swa, preds, gts = evaluate("test")
print(f"TEST: Acc={test_acc:.3f} | SWA={test_swa:.3f}")
experiment_data["SPR_BENCH"]["metrics"]["test"] = {"acc": test_acc, "swa": test_swa}
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# ---------------------- save -----------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)
